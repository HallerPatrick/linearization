from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2 import Qwen2Config, Qwen2PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from transformers.activations import ACT2FN

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.feature_map import RebasedFeatureMap
from fla.models.utils import Cache, FLAGenerationMixin
from fla.layers.gla import GatedLinearAttention as FlaGatedLinearAttention

from .utils import unroll_value_projection

logger = logging.get_logger(__name__)


class Qwen2GLAConfig(Qwen2Config):
    model_type = "qwen2_gla"

    def __init__(
        self,
        attn_mode: str = "chunk",
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        feature_map: Optional[str] = None,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        **qwen2_kwargs,
    ):
        self.attn_mode = attn_mode
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_output_gate = use_output_gate
        self.feature_map = feature_map
        self.clamp_min = clamp_min
        self.fuse_norm = fuse_norm
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.gate_logit_normalizer = gate_logit_normalizer
        self.gate_low_rank_dim = gate_low_rank_dim
        super().__init__(**qwen2_kwargs)


class GatedLinearAttention(FlaGatedLinearAttention):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        feature_map: Optional[str] = None,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        layer_idx: int = None,
    ):
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_output_gate = use_output_gate

        expand_k = 1.0
        expand_v = 1.0

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx

        if feature_map == "rebased":
            feature_dim = self.key_dim // self.num_heads
            self.feature_map_fn = RebasedFeatureMap(feature_dim, True, True, True)
        else:
            self.feature_map_fn = ACT2FN[feature_map] if feature_map is not None else None

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not supported mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')

        self.gk_proj = nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                     nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True))

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
            self.gate_fn = ACT2FN[gate_fn]

        self.gate_logit_normalizer = gate_logit_normalizer
        self.g_norm_swish_gate = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        diagnostics=None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states),
                                            cu_seqlens=cu_seqlens,
                                            cache=conv_state_q,
                                            output_final_state=use_cache)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states),
                                            cu_seqlens=cu_seqlens,
                                            cache=conv_state_k,
                                            output_final_state=use_cache)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states),
                                            cu_seqlens=cu_seqlens,
                                            cache=conv_state_v,
                                            output_final_state=use_cache)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        gk = self.gk_proj(hidden_states)

        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        if self.num_kv_groups > 1:
            k, v, gk = (repeat(x, 'b t (h d) -> b t (h g) d', h=self.num_kv_heads, g=self.num_kv_groups) for x in (k, v, gk))
        else:
            k, v, gk = (rearrange(x, 'b t (h d) -> b t h d', h=self.num_kv_heads) for x in (k, v, gk))

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))

        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        attentions = None
        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q, k=k, v=v, gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(
                q=q, k=k, v=v, g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q, k=k, v=v, g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if diagnostics is not None:
            alpha = torch.exp(gk).clamp(0.0, 1.0)
            beta = alpha.mean(-1)
            diagnostics.record_gate(self.layer_idx, alpha, beta)
            if isinstance(recurrent_state, torch.Tensor):
                prev_state = last_state["recurrent_state"] if last_state is not None else None
                if not isinstance(prev_state, torch.Tensor):
                    prev_state = None
                diagnostics.record_state(self.layer_idx, recurrent_state, prev_state)

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b t (h d) -> b t h d', h=self.num_heads)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, 'b t h d -> b t (h d)')
            else:
                o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if diagnostics is not None:
            diagnostics.record_output(self.layer_idx, o)

        return o, attentions, past_key_values


class Qwen2GLADecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GatedLinearAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            feature_map=config.feature_map,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            use_output_gate=config.use_output_gate,
            gate_fn=config.hidden_act,
            elementwise_affine=config.elementwise_affine,
            norm_eps=config.norm_eps,
            clamp_min=config.clamp_min,
            fuse_norm=config.fuse_norm,
            layer_idx=layer_idx
        )
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        diagnostics=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attentions, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            diagnostics=diagnostics,
            **kwargs
        )

        if output_attentions:
            return hidden_states, attentions

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)
        return outputs


class Qwen2GLAModel(Qwen2PreTrainedModel):
    config_class = Qwen2GLAConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2GLADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        diagnostics=None,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    **kwargs
                )
            else:
                hidden_states, attn, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    diagnostics=diagnostics,
                    **kwargs
                )

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )


class Qwen2GLAPreTrainedModel(Qwen2PreTrainedModel):
    config_class = Qwen2GLAConfig

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2GLAForCausalLM(Qwen2GLAPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen2GLAConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2GLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def freeze_it(self):
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.model.layers:
            for param in layer.self_attn.parameters():
                param.requires_grad = True

    def copy_from_teacher(self, teacher, copy_qkv=True):
        assert len(self.model.layers) == len(teacher.model.layers)

        self.model.embed_tokens.weight.data.copy_(teacher.get_input_embeddings().weight.data)
        self.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
        self.lm_head.weight.data.copy_(teacher.get_output_embeddings().weight.data)

        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.layers):
            self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict())
            self_layer.input_layernorm.load_state_dict(teacher_layer.input_layernorm.state_dict())
            self_layer.post_attention_layernorm.load_state_dict(teacher_layer.post_attention_layernorm.state_dict())

            if copy_qkv:
                self_layer.self_attn.q_proj.load_state_dict(teacher_layer.self_attn.q_proj.state_dict())
                self_layer.self_attn.o_proj.load_state_dict(teacher_layer.self_attn.o_proj.state_dict())

                n_rep = self_layer.self_attn.v_proj.weight.shape[0] // teacher_layer.self_attn.v_proj.weight.shape[0]
                v_proj_unrolled = unroll_value_projection(teacher_layer.self_attn.v_proj, n_rep)
                k_proj_unrolled = unroll_value_projection(teacher_layer.self_attn.k_proj, n_rep)
                self_layer.self_attn.v_proj.load_state_dict(v_proj_unrolled.state_dict())
                self_layer.self_attn.k_proj.load_state_dict(k_proj_unrolled.state_dict())

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        diagnostics=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            diagnostics=diagnostics,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
AutoConfig.register(Qwen2GLAConfig.model_type, Qwen2GLAConfig)
AutoModel.register(Qwen2GLAConfig, Qwen2GLAModel)
AutoModelForCausalLM.register(Qwen2GLAConfig, Qwen2GLAForCausalLM)

StudentConfig = Qwen2GLAConfig
StudentModel = Qwen2GLAForCausalLM
