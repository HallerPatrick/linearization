import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2 import Qwen2Config, Qwen2PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm

from fla.models.utils import Cache, FLAGenerationMixin
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution

from .utils import unroll_value_projection

logger = logging.get_logger(__name__)


def naive_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = (-A_log.view(H, 1).float().exp() * F.softplus(g.float())).to(output_dtype)
    return g


class Qwen2KDAConfig(Qwen2Config):
    model_type = "qwen2_kda"

    def __init__(
        self,
        attn_mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        **qwen2_kwargs,
    ):
        self.attn_mode = attn_mode
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.norm_eps = norm_eps
        super().__init__(**qwen2_kwargs)


class KimiDeltaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}."
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}."
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim, kernel_size=conv_size, bias=conv_bias, activation="silu",
            )

        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=True)
        self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, elementwise_affine=True)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        diagnostics=None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2

        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states), cache=conv_state_q,
                output_final_state=use_cache, cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states), cache=conv_state_k,
                output_final_state=use_cache, cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states), cache=conv_state_v,
                output_final_state=use_cache, cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states).sigmoid()

        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k, g = (repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads) for x in (q, k, g))
            beta = repeat(beta, "... h -> ... (h g)", g=self.num_v_heads // self.num_heads)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None

        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=recurrent_state, output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_recurrent":
            g = fused_kda_gate(g=g, A_log=self.A_log, dt_bias=self.dt_bias)
            o, recurrent_state = fused_recurrent_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state, output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if diagnostics is not None:
            g_eff = naive_kda_gate(g=g, A_log=self.A_log, dt_bias=self.dt_bias)
            alpha = torch.exp(g_eff).clamp(0.0, 1.0)
            diagnostics.record_gate(self.layer_idx, alpha, beta)
            beta_k2 = beta.float() * (k.float() ** 2).sum(-1)
            diagnostics.record_beta_k2(self.layer_idx, beta_k2)
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
                offset=q_len,
            )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if diagnostics is not None:
            diagnostics.record_output(self.layer_idx, o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


class Qwen2KDADecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = KimiDeltaAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_self_attn: bool = False,
        diagnostics=None,
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
            diagnostics=diagnostics,
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


class Qwen2KDAModel(Qwen2PreTrainedModel):
    config_class = Qwen2KDAConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2KDADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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


class Qwen2KDAPreTrainedModel(Qwen2PreTrainedModel):
    config_class = Qwen2KDAConfig

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, KimiDeltaAttention) and next(module.parameters()).device.type == "meta":
            with torch.no_grad():
                if not getattr(module.A_log, '_is_hf_initialized', False):
                    module.A_log.copy_(nn.init.uniform_(module.A_log, a=1, b=16).log())
                if not getattr(module.dt_bias, '_is_hf_initialized', False):
                    dt = torch.exp(
                        nn.init.uniform_(module.dt_bias) * (math.log(0.1) - math.log(0.001)) + math.log(0.001),
                    ).clamp(min=1e-4)
                    inv_dt = dt + torch.log(-torch.expm1(-dt))
                    module.dt_bias.copy_(inv_dt)
                module.dt_bias._is_hf_initialized = True
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2KDAForCausalLM(Qwen2KDAPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen2KDAConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2KDAModel(config)
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
AutoConfig.register(Qwen2KDAConfig.model_type, Qwen2KDAConfig)
AutoModel.register(Qwen2KDAConfig, Qwen2KDAModel)
AutoModelForCausalLM.register(Qwen2KDAConfig, Qwen2KDAForCausalLM)

StudentConfig = Qwen2KDAConfig
StudentModel = Qwen2KDAForCausalLM
