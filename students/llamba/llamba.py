# Copyright (c) 2024, Kevin Li, Aviv Bick.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.utils import ModelOutput, logging
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_llamba import LlambaConfig
from .discrete_mamba import DiscreteMamba2
from .modeling_llama import LlamaMLP, LlamaRMSNorm

logger = logging.get_logger(__name__)


def _get_linear(mod, names):
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise AttributeError(f"Could not find any of {names} on module {type(mod)}")


def _weight_bias(linear: torch.nn.Linear):
    W = linear.weight.data
    b = linear.bias.data if linear.bias is not None else None
    return W, b


def _expand_kv_heads(W_kv, num_heads, kv_heads, head_dim):
    # W_kv: [kv_heads*head_dim, hidden]
    if kv_heads == num_heads:
        return W_kv
    if head_dim is None or head_dim * kv_heads != W_kv.shape[0]:
        assert W_kv.shape[0] % kv_heads == 0, (
            f"W_kv rows={W_kv.shape[0]} must be divisible by kv_heads={kv_heads}"
        )
        head_dim = W_kv.shape[0] // kv_heads
    assert num_heads % kv_heads == 0, f"num_heads={num_heads} not divisible by kv_heads={kv_heads}"
    rep = num_heads // kv_heads
    W = W_kv.view(kv_heads, head_dim, -1)               # [kv, hd, hidden]
    W = W.repeat_interleave(rep, dim=0)                 # [num_heads, hd, hidden]
    return W.reshape(num_heads * head_dim, -1)          # [num_heads*hd, hidden]


def _make_P(ssm_state_size, head_dim, device, dtype):
    P = torch.zeros(ssm_state_size, head_dim, device=device, dtype=dtype)
    diag = min(ssm_state_size, head_dim)
    P[:diag, :diag] = torch.eye(diag, device=device, dtype=dtype)
    return P


def _qk_to_grouped_bc(W, num_heads, head_dim, n_groups, ssm_state_size):
    # W: [num_heads*head_dim, hidden] -> [n_groups*ssm_state_size, hidden]
    hidden = W.shape[1]
    if head_dim is None or head_dim * num_heads != W.shape[0]:
        assert W.shape[0] % num_heads == 0, (
            f"W rows={W.shape[0]} must be divisible by num_heads={num_heads}"
        )
        head_dim = W.shape[0] // num_heads
    assert num_heads % n_groups == 0, f"num_heads={num_heads} must be divisible by n_groups={n_groups}"
    heads_per_group = num_heads // n_groups

    W_h = W.view(num_heads, head_dim, hidden)  # [H, Dh, D]
    W_g = W_h.view(n_groups, heads_per_group, head_dim, hidden).mean(dim=1)  # [G, Dh, D]

    P = _make_P(ssm_state_size, head_dim, device=W.device, dtype=W.dtype)    # [S, Dh]
    W_bc = torch.einsum("sd,gdh->gsh", P, W_g)                               # [G, S, D]
    return W_bc.reshape(n_groups * ssm_state_size, hidden)                   # [G*S, D]


def _copy_attn_to_discrete_mamba(mixer: DiscreteMamba2, teacher_attn):
    """
    mixer: DiscreteMamba2 (student)
    teacher_attn: attention module (teacher)
    """
    # ---- pull teacher projections
    q_proj = _get_linear(teacher_attn, ["q_proj", "Wq"])
    k_proj = _get_linear(teacher_attn, ["k_proj", "Wk"])
    v_proj = _get_linear(teacher_attn, ["v_proj", "Wv"])
    o_proj = _get_linear(teacher_attn, ["o_proj", "Wo"])

    Wq, bq = _weight_bias(q_proj)
    Wk, bk = _weight_bias(k_proj)
    Wv, bv = _weight_bias(v_proj)
    Wo, bo = _weight_bias(o_proj)

    # ---- student mixer hyperparams
    d_model = mixer.d_model
    d_inner = mixer.d_inner
    d_state = mixer.d_state
    n_qk_heads = mixer.n_qk_heads
    n_v_heads = mixer.n_v_heads
    headdim = mixer.headdim

    # ---- teacher kv heads (for GQA/MQA) + teacher head dim inference
    kv_heads = getattr(teacher_attn, "num_key_value_heads", None) or getattr(teacher_attn, "n_kv_heads", None)
    teacher_num_heads = getattr(teacher_attn, "num_heads", None) or getattr(teacher_attn, "num_attention_heads", None)
    if teacher_num_heads is None:
        teacher_num_heads = n_v_heads # assume same as student if not found
    
    teacher_head_dim = Wq.shape[0] // teacher_num_heads
    if kv_heads is None:
        kv_heads = teacher_num_heads
    
    # ---- move teacher weights onto student dtype/device
    device = mixer.in_proj.weight.device
    dtype  = mixer.in_proj.weight.dtype
    Wq = Wq.to(device=device, dtype=dtype)
    Wk = Wk.to(device=device, dtype=dtype)
    Wv = Wv.to(device=device, dtype=dtype)

    # Expand KV to full heads if needed
    Wk_full = _expand_kv_heads(Wk, num_heads=teacher_num_heads, kv_heads=kv_heads, head_dim=teacher_head_dim)
    Wv_full = _expand_kv_heads(Wv, num_heads=teacher_num_heads, kv_heads=kv_heads, head_dim=teacher_head_dim)

    # ---- build B/C blocks (K/Q mapping)
    # n_qk_heads in DiscreteMamba2 acts like n_groups in Mamba2Mixer logic
    W_B = _qk_to_grouped_bc(Wk_full, num_heads=teacher_num_heads, head_dim=teacher_head_dim,
                            n_groups=n_qk_heads, ssm_state_size=d_state)
    W_C = _qk_to_grouped_bc(Wq,      num_heads=teacher_num_heads, head_dim=teacher_head_dim,
                            n_groups=n_qk_heads, ssm_state_size=d_state)

    # ---- V block into hidden stream (x)
    # Need [d_inner, d_model]
    if Wv_full.shape[0] != d_inner:
        if Wv_full.shape[0] > d_inner:
            Wv_block = Wv_full[:d_inner, :]
        else:
            Wv_block = torch.zeros(d_inner, Wv_full.shape[1], device=device, dtype=dtype)
            Wv_block[:Wv_full.shape[0], :] = Wv_full
    else:
        Wv_block = Wv_full

    # ---- copy into student mixer.in_proj
    # Layout: [xBC, z, A_log]
    # xBC = [x, B, C]
    # x: d_inner
    # B: n_qk_heads * d_state
    # C: n_qk_heads * d_state
    # z: d_inner
    # A_log: n_v_heads
    
    o_x = 0
    o_B = d_inner
    o_C = o_B + n_qk_heads * d_state
    o_z = o_C + n_qk_heads * d_state
    o_A = o_z + d_inner

    with torch.no_grad():
        # Copy V -> x
        mixer.in_proj.weight[o_x : o_x + d_inner, :].copy_(Wv_block)
        # Copy K -> B
        mixer.in_proj.weight[o_B : o_B + n_qk_heads * d_state, :].copy_(W_B)
        # Copy Q -> C
        mixer.in_proj.weight[o_C : o_C + n_qk_heads * d_state, :].copy_(W_C)
        
        # Zero out z and A_log to be safe, or leave them (A_log usually initialized specifically)
        # mixer.in_proj.weight[o_z : o_z + d_inner, :].zero_()
        # mixer.in_proj.weight[o_A : o_A + n_v_heads, :].zero_()

        if mixer.in_proj.bias is not None:
            mixer.in_proj.bias[o_x : o_z].zero_()

        # copy output projection
        Wo = Wo.to(device=mixer.out_proj.weight.device, dtype=mixer.out_proj.weight.dtype)
        if mixer.out_proj.weight.shape == Wo.shape:
            mixer.out_proj.weight.copy_(Wo)
        elif mixer.out_proj.weight.shape == Wo.t().shape:
            mixer.out_proj.weight.copy_(Wo.t())

        if mixer.out_proj.bias is not None and bo is not None:
            mixer.out_proj.bias.copy_(bo.to(device=mixer.out_proj.bias.device, dtype=mixer.out_proj.bias.dtype))


class LlambaPreTrainedModel(PreTrainedModel):
    config_class = LlambaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]

    def _init_weights(self, module):
        if any(p.device.type == "meta" for p in module.parameters(recurse=False)):
            return
        if isinstance(module, DiscreteMamba2):
            # --- A_log ---
            A = torch.empty(module.n_v_heads, dtype=torch.float32).uniform_(0, 16)
            with torch.no_grad():
                module.A_log.copy_(torch.log(A))
            module.A_log._no_weight_decay = True

            # --- D ---
            nn.init.ones_(module.D)
            module.D._no_weight_decay = True

            # --- dt_bias ---
            dt = torch.exp(
                torch.rand(module.n_v_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min),
            ).clamp(min=self.config.time_step_floor)

            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True

            # Special init for in_proj that predicts dt
            # in_proj layout: [xBC, z, dt]
            # We want the weights for the dt part to be zero so that dt_bias dominates initially
            dt_offset = 2 * module.d_inner + 2 * module.n_qk_heads * module.d_state
            with torch.no_grad():
                module.in_proj.weight[dt_offset:, :].zero_()

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Block(nn.Module):
    def __init__(self, config: LlambaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Mixer
        self.mixer = DiscreteMamba2(
            d_model=self.config.d_model,
            layer_idx=layer_idx,
            **config.ssm_cfg,
        )

        # Other components
        self.input_layernorm = LlamaRMSNorm(
            hidden_size=self.config.d_model, eps=config.norm_epsilon
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            hidden_size=self.config.d_model, eps=config.norm_epsilon
        )
        self.mlp = LlamaMLP(
            hidden_size=self.config.d_model,
            **config.mlp_cfg,
        )

    def forward(
        self,
        hidden_states: Tensor,
        inference_params=None,
        attention_mask=None,
        output_attentions=None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        # Apply Mixer
        mixer_outputs = self.mixer(
            hidden_states,
            inference_params=inference_params,
        )

        if output_attentions:
            return mixer_outputs["hidden_states"],
        
        hidden_states = mixer_outputs["hidden_states"].to(residual.dtype) + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states,


class LlambaModel(LlambaPreTrainedModel):
    def __init__(self, config: LlambaConfig):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList(
            [Block(config=config, layer_idx=i) for i in range(config.n_layer)]
        )

        self.final_layernorm = LlamaRMSNorm(
            hidden_size=config.d_model,
            eps=config.norm_epsilon,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inference_params=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            hidden_states = self.embedding(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states,  = layer(
                hidden_states,
                inference_params=inference_params,
                attention_mask=attention_mask,
            )

        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class LlambaLMHeadModel(LlambaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}
    # _tied_weights_keys = ["lm_head.weight", "model.embedding.weight"]

    def __init__(self, config: LlambaConfig, **kwargs):
        super().__init__(config)
        self.model = LlambaModel(config)
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=config.lm_head_bias
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def tie_weights(self, **kwargs):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.model.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.model.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inference_params=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inference_params=inference_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self, input_ids, inference_params=None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        if inference_params is not None:
            if inference_params.seqlen_offset > 0:
                input_ids = input_ids[:, -1].unsqueeze(-1)
        
        if inputs_embeds is not None and inference_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "inference_params": inference_params,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def copy_from_teacher(self, teacher, copy_qkv=True):
        def _copy_param(dst, src):
            if dst.shape != src.shape:
                logger.warning(f"Shape mismatch: {dst.shape} vs {src.shape}")
                return False
            dst.data.copy_(src.data)
            return True

        with torch.no_grad():
            # Embeddings
            if hasattr(teacher, "get_input_embeddings"):
                t_embed = teacher.get_input_embeddings().weight
                _copy_param(self.model.embedding.weight, t_embed)

            # Final norm
            if hasattr(teacher, "model") and hasattr(teacher.model, "norm"):
                _copy_param(self.model.final_layernorm.weight, teacher.model.norm.weight)

            # LM head (if untied)
            if not self.config.tie_embeddings and hasattr(teacher, "get_output_embeddings"):
                t_lm = teacher.get_output_embeddings().weight
                _copy_param(self.lm_head.weight, t_lm)

            # Per-layer MLP + norms + optionally Mamba mixer
            if hasattr(teacher, "model") and hasattr(teacher.model, "layers"):
                for i, teacher_layer in enumerate(teacher.model.layers):
                    if i >= len(self.model.layers):
                        break
                    self_layer = self.model.layers[i]
                    if hasattr(teacher_layer, "mlp"):
                        self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict(), strict=False)
                    if hasattr(teacher_layer, "input_layernorm"):
                        self_layer.input_layernorm.load_state_dict(
                            teacher_layer.input_layernorm.state_dict(), strict=False
                        )
                    if hasattr(teacher_layer, "post_attention_layernorm"):
                        self_layer.post_attention_layernorm.load_state_dict(
                            teacher_layer.post_attention_layernorm.state_dict(), strict=False
                        )
                    
                    if copy_qkv and hasattr(teacher_layer, "self_attn"):
                        _copy_attn_to_discrete_mamba(self_layer.mixer, teacher_layer.self_attn)


from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register(LlambaConfig.model_type, LlambaConfig)
AutoModelForCausalLM.register(LlambaConfig, LlambaLMHeadModel)

# For backward compatibility within this project
StudentConfig = LlambaConfig
StudentModel = LlambaLMHeadModel
