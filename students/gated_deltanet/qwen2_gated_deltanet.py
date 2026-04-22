import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2 import Qwen2Config, Qwen2PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

from fla.models.utils import Cache, FLAGenerationMixin
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm

from .utils import unroll_value_projection

logger = logging.get_logger(__name__)


class Qwen2GatedDeltaNetConfig(Qwen2Config):
    model_type = "qwen2_gated_deltanet"

    def __init__(
        self,
        attn_mode: str = "chunk",
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        norm_eps: float = 1e-5,
        use_elu: bool = False,
        **qwen2_kwargs,
    ):
        self.attn_mode = attn_mode
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.norm_eps = norm_eps
        self.use_elu = use_elu
        super().__init__(**qwen2_kwargs)


class GatedDeltaNet(nn.Module):

    def __init__(
        self,
        d_model: int = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        mode: str = 'chunk',
        use_gate: bool = False,
        use_output_norm: bool = True,
        use_elu: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        qk_activation: str = 'silu',
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        **kwargs
    ):
        super().__init__()

        self.mode = mode
        self.qk_activation = qk_activation

        expand_k = 1.0
        expand_v = 1.0

        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_output_norm = use_output_norm
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.norm_first = norm_first
        self.layer_idx = layer_idx

        self.silu = nn.SiLU()

        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        if norm_first:
            self.norm = RMSNorm(self.hidden_size, eps=norm_eps)

        self.use_norm = True

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)

        if self.use_norm:
            self.q_norm = RMSNorm(self.head_qk_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_qk_dim, eps=norm_eps)

        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        self.use_elu = use_elu
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )
        else:
            raise UserWarning(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) <= 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 and not self.training else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')

        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(
                hidden_states, "b s ... -> (b s) ..."),
                indices
            ).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        q, k, v = map(lambda x: rearrange(x, '... (h d) -> ... h d', h=self.num_heads), (q, k, v))

        if self.use_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = l2_norm(q)
        k = l2_norm(k)

        beta = self.b_proj(hidden_states).sigmoid()

        if self.allow_neg_eigval:
            beta = beta * 2.

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


class Qwen2GatedDeltaNetDecoderLayer(GradientCheckpointingLayer):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GatedDeltaNet(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            use_gate=config.use_gate,
            num_heads=config.num_attention_heads,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx
        )
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_self_attn: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_items_in_batch=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if output_self_attn:
            output_attentions = False

        hidden_states, attentions, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if output_attentions:
            return hidden_states, attentions

        if output_self_attn:
            return hidden_states, attentions

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)
        return outputs


class Qwen2GatedDeltaNetModel(Qwen2PreTrainedModel):
    config_class = Qwen2GatedDeltaNetConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2GatedDeltaNetDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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
                    output_attentions,
                    use_cache,
                    **kwargs
                )
            else:
                hidden_states, attn, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
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


class Qwen2GatedDeltaNetPreTrainedModel(Qwen2PreTrainedModel):
    config_class = Qwen2GatedDeltaNetConfig
    supports_gradient_checkpointing = True
    base_model_prefix = "model"

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


class Qwen2GatedDeltaNetForCausalLM(Qwen2GatedDeltaNetPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen2GatedDeltaNetConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2GatedDeltaNetModel(config)
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

                if hasattr(teacher_layer.self_attn, 'q_norm'):
                    if self_layer.self_attn.q_norm.weight.shape == teacher_layer.self_attn.q_norm.weight.shape:
                        self_layer.self_attn.q_norm.load_state_dict(teacher_layer.self_attn.q_norm.state_dict())
                        self_layer.self_attn.k_norm.load_state_dict(teacher_layer.self_attn.k_norm.state_dict())

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
AutoConfig.register(Qwen2GatedDeltaNetConfig.model_type, Qwen2GatedDeltaNetConfig)
AutoModel.register(Qwen2GatedDeltaNetConfig, Qwen2GatedDeltaNetModel)
AutoModelForCausalLM.register(Qwen2GatedDeltaNetConfig, Qwen2GatedDeltaNetForCausalLM)

StudentConfig = Qwen2GatedDeltaNetConfig
StudentModel = Qwen2GatedDeltaNetForCausalLM
