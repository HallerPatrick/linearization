"""
Grouped Gated DeltaNet – Experiment 2
======================================
Interpolates between Gated DeltaNet (scalar gate per head) and Kimi / KDA
(channel-wise gate per head) by varying the gate group size G:

  G = head_k_dim  →  one scalar per head  =  Gated DeltaNet behaviour
  G = 1           →  one value per channel =  Kimi / KDA behaviour
  1 < G < head_k_dim  →  channels in groups of G share a decay value

The ONLY structural difference from KDA's KimiDeltaAttention is:
  - The gate projection ``g_proj`` outputs ``num_heads * (head_k_dim // G)``
    values instead of ``num_heads * head_k_dim``.
  - These values are repeated G times (broadcast) before being fed to the
    same KDA kernel used by Kimi.

All other components (q/k/v projections, short convolution, beta gate,
output norm, etc.) are identical to KDA and are copied verbatim.
"""
import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
import transformers.models.llama.modeling_llama as llama
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

from fla.models.kda import KDAConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate
from fla.modules import RMSNorm, ShortConvolution

# Reuse the naive_kda_gate from the kda student for diagnostics
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from students.kda.llama_kda import naive_kda_gate
from students.kda.utils import unroll_value_projection

logger = logging.get_logger(__name__)


class LlamaGroupedGDNConfig(KDAConfig):
    """Extends KDAConfig with a gate_group_size parameter."""

    def __init__(self, *args, gate_group_size=None, intermediate_size=1024,
                 rms_norm_eps=1e-6, mlp_bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        # gate_group_size: number of channels sharing one gate value.
        # head_k_dim = full head (= GDN), 1 = per-channel (= Kimi).
        # None → defaults to head_k_dim at build time.
        self.gate_group_size = gate_group_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.mlp_bias = mlp_bias


class GroupedGatedDeltaNet(nn.Module):
    """
    Grouped-gate variant of the delta-rule token mixer.

    Args
    ----
    gate_group_size : int
        Number of adjacent key-dimension channels that share a single decay
        gate value.  ``head_k_dim`` → scalar per head (GDN); ``1`` → full
        per-channel (Kimi).  Must divide ``head_k_dim`` evenly.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 128,
        num_heads: int = 16,
        gate_group_size: int = None,
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
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = head_dim  # expand_v = 1 always
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = num_heads * self.head_v_dim
        self.layer_idx = layer_idx

        # ── Gate group size ──────────────────────────────────────────────────
        # Default: scalar per head (GDN behaviour)
        if gate_group_size is None:
            gate_group_size = self.head_k_dim
        assert self.head_k_dim % gate_group_size == 0, (
            f"gate_group_size={gate_group_size} must divide head_k_dim={self.head_k_dim}"
        )
        self.gate_group_size = gate_group_size
        self.num_gate_values = self.head_k_dim // gate_group_size  # per head
        self.gate_dim = num_heads * self.num_gate_values  # total output dim

        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        # ── Q / K / V projections (identical to KDA) ─────────────────────────
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size,
                bias=conv_bias, activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size,
                bias=conv_bias, activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim, kernel_size=conv_size,
                bias=conv_bias, activation="silu",
            )

        # ── Gate projection (only this differs from KDA) ─────────────────────
        # KDA:          f_proj: hidden → head_v_dim → key_dim
        # GroupedGDN:   f_proj: hidden → head_v_dim → gate_dim
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=False),
        )

        # ── Beta (retention strength) – identical to KDA ──────────────────────
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ── Eigenvalue parameters – same as KDA ──────────────────────────────
        self.A_log = nn.Parameter(
            torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(self.gate_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

        # ── Output ────────────────────────────────────────────────────────────
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=True)
        self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, elementwise_affine=True)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def _broadcast_gate(self, g: torch.Tensor) -> torch.Tensor:
        """Expand grouped gate values to full key dimension.

        g : [..., H, num_gate_values]  →  [..., H, head_k_dim]
        """
        if self.gate_group_size == 1:
            return g  # already per-channel
        return g.repeat_interleave(self.gate_group_size, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        diagnostics=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:

        if attention_mask is not None:
            assert len(attention_mask.shape) <= 2
            assert attention_mask.shape[-1] == hidden_states.shape[1]

        batch_size, q_len, _ = hidden_states.shape
        mode = self.mode

        last_state = (
            past_key_values[self.layer_idx]
            if past_key_values is not None and len(past_key_values) > self.layer_idx
            else None
        )
        conv_state_q = conv_state_k = conv_state_v = None
        if last_state is not None and "conv_state" in last_state:
            cs = last_state["conv_state"]
            if cs is not None:
                conv_state_q, conv_state_k, conv_state_v = cs

        if self.use_short_conv:
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states), cache=conv_state_q,
                output_final_state=use_cache,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states), cache=conv_state_k,
                output_final_state=use_cache,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states), cache=conv_state_v,
                output_final_state=use_cache,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # Reshape to heads
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # Gate: project → grouped → broadcast to full key dim
        # g_raw shape: [..., gate_dim] where gate_dim = num_heads * num_gate_values
        g_raw = self.f_proj(hidden_states)
        g_grouped = rearrange(g_raw, "... (h d) -> ... h d",
                              d=self.num_gate_values)                       # [..., H, G_vals]

        # Pre-compute the log-domain gate (used by fused_recurrent and diagnostics)
        g_eff_grouped = naive_kda_gate(g=g_grouped, A_log=self.A_log,
                                       dt_bias=self.dt_bias)               # [..., H, G_vals]
        g_eff = self._broadcast_gate(g_eff_grouped)                        # [..., H, head_k_dim]

        beta = self.b_proj(hidden_states).sigmoid()                        # [..., H]
        if self.allow_neg_eigval:
            beta = beta * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None

        if mode == "chunk":
            # chunk_kda expects raw g of shape [..., H, head_k_dim] + dt_bias of
            # shape [num_heads * head_k_dim].  Broadcast both to full head_k_dim.
            g_raw_full = self._broadcast_gate(g_grouped)                   # [..., H, head_k_dim]
            dt_bias_full = self.dt_bias.reshape(self.num_heads, self.num_gate_values) \
                               .repeat_interleave(self.gate_group_size, dim=-1) \
                               .flatten()                                   # [num_heads * head_k_dim]
            o, recurrent_state = chunk_kda(
                q=q, k=k, v=v, g=g_raw_full, beta=beta,
                A_log=self.A_log, dt_bias=dt_bias_full,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
            )
        elif mode == "fused_recurrent":
            # fused_recurrent_kda expects pre-computed log-domain g, full head_k_dim
            o, recurrent_state = fused_recurrent_kda(
                q=q, k=k, v=v, g=g_eff, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"mode={mode} not supported.")

        if diagnostics is not None:
            # alpha = exp(g_eff), clamped to [0,1]
            alpha = g_eff.exp().clamp(0.0, 1.0)  # [..., H, head_k_dim]
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

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d",
                                      d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        if diagnostics is not None:
            diagnostics.record_output(self.layer_idx, o)

        return o, None, past_key_values


# ── Decoder layer ──────────────────────────────────────────────────────────────

class GroupedGDNDecoderLayer(GradientCheckpointingLayer):

    def __init__(self, config: LlamaGroupedGDNConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        head_dim = config.hidden_size // config.num_attention_heads
        gate_group_size = getattr(config, "gate_group_size", None) or head_dim

        self.self_attn = GroupedGatedDeltaNet(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_heads=config.num_attention_heads,
            gate_group_size=gate_group_size,
            mode=config.attn_mode,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
        )
        self.mlp = llama.LlamaMLP(config)
        self.input_layernorm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = llama.LlamaRMSNorm(config.hidden_size,
                                                           eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings=None,
        output_self_attn: bool = False,
        diagnostics=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attentions, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            diagnostics=diagnostics,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attentions, past_key_values


# ── Model body ─────────────────────────────────────────────────────────────────

class GroupedGDNModel(llama.LlamaPreTrainedModel):
    config_class = LlamaGroupedGDNConfig

    def __init__(self, config: LlamaGroupedGDNConfig):
        super().__init__(config)
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GroupedGDNDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (
            self.config.use_cache if not self.training else False
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attn, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                diagnostics=diagnostics,
                **kwargs,
            )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(x for x in [hidden_states, past_key_values,
                                      all_hidden_states, all_attns] if x is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


# ── CausalLM head ──────────────────────────────────────────────────────────────

class GroupedGDNPreTrainedModel(PreTrainedModel):
    config_class = LlamaGroupedGDNConfig

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


class GroupedGDNForCausalLM(GroupedGDNPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}
    config_class = LlamaGroupedGDNConfig

    def __init__(self, config: LlamaGroupedGDNConfig):
        super().__init__(config)
        self.model = GroupedGDNModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def freeze_it(self):
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.model.layers:
            for param in layer.self_attn.parameters():
                param.requires_grad = True

    def copy_from_teacher(self, teacher, copy_qkv=True):
        """Copy non-mixer params from teacher (same pattern as KDA)."""
        assert len(self.model.layers) == len(teacher.model.layers)
        self.model.embeddings.weight.data.copy_(teacher.get_input_embeddings().weight.data)
        self.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
        self.lm_head.weight.data.copy_(teacher.get_output_embeddings().weight.data)

        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.layers):
            self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict())
            self_layer.input_layernorm.load_state_dict(
                teacher_layer.input_layernorm.state_dict())
            self_layer.post_attention_layernorm.load_state_dict(
                teacher_layer.post_attention_layernorm.state_dict())
            if copy_qkv:
                try:
                    self_layer.self_attn.q_proj.load_state_dict(
                        teacher_layer.self_attn.q_proj.state_dict())
                    self_layer.self_attn.o_proj.load_state_dict(
                        teacher_layer.self_attn.o_proj.state_dict())
                    n_rep = (self_layer.self_attn.v_proj.weight.shape[0]
                             // teacher_layer.self_attn.v_proj.weight.shape[0])
                    v_unrolled = unroll_value_projection(teacher_layer.self_attn.v_proj, n_rep)
                    k_unrolled = unroll_value_projection(teacher_layer.self_attn.k_proj, n_rep)
                    self_layer.self_attn.v_proj.load_state_dict(v_unrolled.state_dict())
                    self_layer.self_attn.k_proj.load_state_dict(k_unrolled.state_dict())
                except Exception as e:
                    logger.warning(f"copy_from_teacher: could not copy QKV – {e}")

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
        diagnostics=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            diagnostics=diagnostics,
            **kwargs,
        )
        hidden_states = outputs[0]
        if num_logits_to_keep > 0:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:])
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

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


# ── Public exports (used by train.py dynamic import) ──────────────────────────
StudentConfig = LlamaGroupedGDNConfig
StudentModel = GroupedGDNForCausalLM
