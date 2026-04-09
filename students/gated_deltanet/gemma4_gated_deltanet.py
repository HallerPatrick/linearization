import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextAttention,
    Gemma4TextMLP,
    Gemma4RMSNorm,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
)

from fla.models.utils import Cache, FLAGenerationMixin
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm

from .utils import unroll_value_projection

logger = logging.get_logger(__name__)


class Gemma4GatedDeltaNetConfig(Gemma4TextConfig):
    model_type = "gemma4_gated_deltanet"

    def __init__(
        self,
        attn_mode: str = "chunk",
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        norm_eps: float = 1e-6,
        hidden_size_per_layer_input: int = 0,
        enable_moe_block: bool = False,
        **gemma4_kwargs,
    ):
        self.attn_mode = attn_mode
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.norm_eps = norm_eps
        super().__init__(
            hidden_size_per_layer_input=hidden_size_per_layer_input,
            enable_moe_block=enable_moe_block,
            **gemma4_kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        # Gemma4 is a multimodal model; unwrap the text sub-config
        if hasattr(cfg, "text_config"):
            cfg = cfg.text_config
        text_dict = cfg.to_dict()
        text_dict.pop("model_type", None)
        text_dict.update(kwargs)
        return cls(**text_dict)


class GatedDeltaNet(nn.Module):

    def __init__(
        self,
        hidden_size: int = 1024,
        head_dim: int = None,
        num_heads: int = 4,
        mode: str = "chunk",
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx

        # When head_dim is explicitly provided (e.g. Gemma4), key/value dims are
        # derived from it rather than from hidden_size, so projections match the teacher.
        if head_dim is not None:
            self.key_dim = num_heads * head_dim
            self.value_dim = num_heads * head_dim
            self.head_qk_dim = head_dim
            self.head_v_dim = head_dim
        else:
            self.key_dim = hidden_size
            self.value_dim = hidden_size
            self.head_qk_dim = hidden_size // num_heads
            self.head_v_dim = hidden_size // num_heads

        assert self.key_dim % num_heads == 0
        assert self.value_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_norm = RMSNorm(self.head_qk_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_qk_dim, eps=norm_eps)

        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, activation="silu")
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, activation="silu")
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, activation="silu")

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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) <= 2

        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if q_len <= 64 and not self.training else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens")

        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states), cache=conv_state_q,
            output_final_state=use_cache, cu_seqlens=cu_seqlens
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states), cache=conv_state_k,
            output_final_state=use_cache, cu_seqlens=cu_seqlens
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states), cache=conv_state_v,
            output_final_state=use_cache, cu_seqlens=cu_seqlens
        )

        q, k, v = map(lambda x: rearrange(x, "... (h d) -> ... h d", h=self.num_heads), (q, k, v))
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = l2_norm(q)
        k = l2_norm(k)

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        cu_seqlens = kwargs.get("cu_seqlens", None)

        if mode == "chunk":
            o, recurrent_state = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=q.shape[1],
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


class Gemma4GatedDeltaNetDecoderLayer(GradientCheckpointingLayer):

    def __init__(self, config: Gemma4GatedDeltaNetConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.layer_idx = layer_idx

        if self.layer_type == "sliding_attention":
            self.self_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.num_attention_heads,
                mode=config.attn_mode,
                use_gate=config.use_gate,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
            )
        else:
            self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)
            # Own rotary_emb so this layer can compute position embeddings when
            # called directly (e.g. hidden-to-hidden alignment) without a full
            # model forward pass providing them.
            self.rotary_emb = Gemma4TextRotaryEmbedding(config)

        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "sliding_attention":
            hidden_states, _, past_key_values = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            if position_embeddings is None:
                if position_ids is None:
                    position_ids = torch.arange(
                        hidden_states.shape[1], device=hidden_states.device
                    ).unsqueeze(0)
                position_embeddings = self.rotary_emb(hidden_states, position_ids, "full_attention")
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class Gemma4GatedDeltaNetPreTrainedModel(PreTrainedModel):
    config_class = Gemma4GatedDeltaNetConfig
    supports_gradient_checkpointing = True
    base_model_prefix = "model"
    _no_split_modules = ["Gemma4GatedDeltaNetDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Embedding, Gemma4TextScaledWordEmbedding)):
            module.weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "padding_idx") and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Gemma4GatedDeltaNetModel(Gemma4GatedDeltaNetPreTrainedModel):
    config_class = Gemma4GatedDeltaNetConfig

    def __init__(self, config: Gemma4GatedDeltaNetConfig):
        super().__init__(config)
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx,
            embed_scale=config.hidden_size ** 0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma4GatedDeltaNetDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.unique_layer_types = list(set(config.layer_types))
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0) + past_seen_tokens

        # Keep the original 2D padding mask for GatedDeltaNet unpadding;
        # full attention layers use the computed 4D causal mask.
        padding_mask = attention_mask if not isinstance(attention_mask, dict) else None

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": position_ids[0],
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds

        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in self.unique_layer_types
        }

        for i, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            # GatedDeltaNet handles masking via 2D padding_mask / cu_seqlens;
            # full_attention uses the 4D causal mask.
            layer_mask = padding_mask if layer_type == "sliding_attention" else causal_mask_mapping[layer_type]

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    None,
                    position_embeddings[layer_type],
                    layer_mask,
                    position_ids,
                    past_key_values,
                    use_cache,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    per_layer_input=None,
                    position_embeddings=position_embeddings[layer_type],
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Gemma4GatedDeltaNetForCausalLM(Gemma4GatedDeltaNetPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Gemma4GatedDeltaNetConfig

    def __init__(self, config: Gemma4GatedDeltaNetConfig):
        super().__init__(config)
        self.model = Gemma4GatedDeltaNetModel(config)
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

    def get_decoder(self):
        return self.model

    def freeze_it(self):
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.model.layers:
            if layer.layer_type == "sliding_attention":
                for param in layer.self_attn.parameters():
                    param.requires_grad = True

    def copy_from_teacher(self, teacher, copy_qkv=True):

        teacher_model = teacher.model.language_model
        assert len(self.model.layers) == len(teacher_model.layers)

        self.model.embed_tokens.weight.data.copy_(teacher_model.embed_tokens.weight.data)
        self.model.norm.weight.data.copy_(teacher_model.norm.weight.data)
        self.lm_head.weight.data.copy_(teacher.lm_head.weight.data)

        config = self.config
        for self_layer, teacher_layer in zip(self.model.layers, teacher_model.layers):
            self_layer.input_layernorm.load_state_dict(teacher_layer.input_layernorm.state_dict())
            self_layer.post_attention_layernorm.load_state_dict(teacher_layer.post_attention_layernorm.state_dict())
            self_layer.pre_feedforward_layernorm.load_state_dict(teacher_layer.pre_feedforward_layernorm.state_dict())
            self_layer.post_feedforward_layernorm.load_state_dict(teacher_layer.post_feedforward_layernorm.state_dict())
            self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict())
            self_layer.layer_scalar.copy_(teacher_layer.layer_scalar)

            if copy_qkv:
                if self_layer.layer_type == "full_attention":
                    self_layer.self_attn.load_state_dict(teacher_layer.self_attn.state_dict())
                else:
                    # q_proj and o_proj shapes match directly
                    self_layer.self_attn.q_proj.load_state_dict(teacher_layer.self_attn.q_proj.state_dict())
                    self_layer.self_attn.o_proj.load_state_dict(teacher_layer.self_attn.o_proj.state_dict())

                    # k_proj and v_proj: unroll GQA from num_key_value_heads to num_attention_heads
                    n_rep = config.num_attention_heads // config.num_key_value_heads
                    head_dim = config.head_dim
                    self_layer.self_attn.k_proj.load_state_dict(
                        unroll_value_projection(teacher_layer.self_attn.k_proj, n_rep, head_dim).state_dict()
                    )
                    self_layer.self_attn.v_proj.load_state_dict(
                        unroll_value_projection(teacher_layer.self_attn.v_proj, n_rep, head_dim).state_dict()
                    )

                    # q_norm and k_norm: same head_dim, copy directly
                    self_layer.self_attn.q_norm.weight.data.copy_(teacher_layer.self_attn.q_norm.weight.data)
                    self_layer.self_attn.k_norm.weight.data.copy_(teacher_layer.self_attn.k_norm.weight.data)

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
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
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
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

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
AutoConfig.register(Gemma4GatedDeltaNetConfig.model_type, Gemma4GatedDeltaNetConfig)
AutoModel.register(Gemma4GatedDeltaNetConfig, Gemma4GatedDeltaNetModel)
AutoModelForCausalLM.register(Gemma4GatedDeltaNetConfig, Gemma4GatedDeltaNetForCausalLM)

StudentConfig = Gemma4GatedDeltaNetConfig
StudentModel = Gemma4GatedDeltaNetForCausalLM
