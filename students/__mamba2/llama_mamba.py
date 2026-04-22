from typing import Optional
import math

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import Placement, Replicate

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import transformers.models.llama.modeling_llama as llama
from transformers.modeling_utils import PreTrainedModel

from fla.layers.mamba2 import is_fast_path_available
from fla.models.mamba2.modeling_mamba2 import Mamba2 as FLAMamba2
from fla.modules import FusedLinearCrossEntropyLoss, FusedCrossEntropyLoss
from fla.models.mamba2 import Mamba2Config

def tensor_to_dtensor(
    tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    current_placement: Placement | list[Placement],
    desired_placement: Placement | list[Placement] | None = None,
    run_check: bool = False,
):
    if isinstance(tensor, DTensor):
        return tensor

    if isinstance(current_placement, Placement):
        current_placement = [current_placement]

    dtensor = DTensor.from_local(tensor, device_mesh=device_mesh, run_check=run_check, placements=current_placement)

    if desired_placement is not None:
        if isinstance(desired_placement, Placement):
            desired_placement = [desired_placement]

        dtensor = dtensor.redistribute(device_mesh=device_mesh, placements=desired_placement, async_op=True)

    return dtensor

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

def _copy_attn_to_mamba2_mixer(mixer, teacher_attn):
    """
    mixer: Mamba2Mixer (student)
    teacher_attn: attention module (teacher)
    """
    # ---- pull teacher projections (most HF Llama-style modules have q_proj/k_proj/v_proj/o_proj)
    q_proj = _get_linear(teacher_attn, ["q_proj", "Wq"])
    k_proj = _get_linear(teacher_attn, ["k_proj", "Wk"])
    v_proj = _get_linear(teacher_attn, ["v_proj", "Wv"])
    o_proj = _get_linear(teacher_attn, ["o_proj", "Wo"])

    Wq, bq = _weight_bias(q_proj)
    Wk, bk = _weight_bias(k_proj)
    Wv, bv = _weight_bias(v_proj)
    Wo, bo = _weight_bias(o_proj)

    # ---- student mixer hyperparams
    # We assume mixer has these attrs; if not, infer from shapes.
    hidden_size = mixer.in_proj.in_features
    proj_out = mixer.in_proj.out_features

    num_heads = getattr(mixer, "num_heads", None)
    head_dim  = getattr(mixer, "head_dim", None)
    n_groups  = getattr(mixer, "n_groups", None)
    ssm_state_size = getattr(mixer, "ssm_state_size", None)
    intermediate_size = getattr(mixer, "intermediate_size", None)
    conv_dim = getattr(mixer, "conv_dim", None)

    # Fallback inference from shapes if attrs missing
    if intermediate_size is None:
        # from your forward: conv_dim packs (hidden_states + B + C) where hidden_states is intermediate_size
        # Often intermediate_size == hidden_size in this config
        intermediate_size = hidden_size
    if conv_dim is None or n_groups is None or ssm_state_size is None:
        # We can recover (conv_dim) only if we know groups/state; but in HF Mamba2Mixer these should exist.
        raise AttributeError("mixer is missing conv_dim / n_groups / ssm_state_size attributes needed for slicing.")
    if num_heads is None:
        raise AttributeError("mixer is missing num_heads attribute.")
    if head_dim is None:
        # common: intermediate_size == num_heads*head_dim
        assert intermediate_size % num_heads == 0
        head_dim = intermediate_size // num_heads

    # ---- teacher kv heads (for GQA/MQA) + teacher head dim inference
    kv_heads = getattr(teacher_attn, "num_key_value_heads", None) or getattr(teacher_attn, "n_kv_heads", None)
    teacher_num_heads = getattr(teacher_attn, "num_heads", None) or getattr(teacher_attn, "num_attention_heads", None)
    if teacher_num_heads is None:
        teacher_num_heads = num_heads
    assert Wq.shape[0] % teacher_num_heads == 0, (
        f"Wq rows={Wq.shape[0]} must be divisible by teacher_num_heads={teacher_num_heads}"
    )
    teacher_head_dim = Wq.shape[0] // teacher_num_heads
    if kv_heads is None:
        kv_heads = num_heads
    if Wk.shape[0] != num_heads * head_dim and Wk.shape[0] % teacher_head_dim == 0:
        kv_heads = Wk.shape[0] // teacher_head_dim

    # ---- move teacher weights onto student dtype/device
    device = mixer.in_proj.weight.device
    dtype  = mixer.in_proj.weight.dtype
    Wq = Wq.to(device=device, dtype=dtype)
    Wk = Wk.to(device=device, dtype=dtype)
    Wv = Wv.to(device=device, dtype=dtype)

    # Expand KV to full heads if needed
    qk_head_dim = head_dim
    if head_dim * num_heads != Wq.shape[0]:
        qk_head_dim = teacher_head_dim
    Wk_full = _expand_kv_heads(Wk, num_heads=num_heads, kv_heads=kv_heads, head_dim=qk_head_dim)
    Wv_full = _expand_kv_heads(Wv, num_heads=num_heads, kv_heads=kv_heads, head_dim=qk_head_dim)

    # ---- compute packed offsets (matches your forward split)
    # projected_states.split([d_mlp, d_mlp, intermediate_size, conv_dim, num_heads], dim=-1)
    d_mlp = (proj_out - (intermediate_size + conv_dim + num_heads)) // 2
    assert 2 * d_mlp + intermediate_size + conv_dim + num_heads == proj_out, (
        f"in_proj layout mismatch: got out={proj_out}, inferred d_mlp={d_mlp}, "
        f"intermediate={intermediate_size}, conv_dim={conv_dim}, num_heads={num_heads}"
    )

    o_mlp1 = 0
    o_mlp2 = o_mlp1 + d_mlp
    o_z    = o_mlp2 + d_mlp
    o_hbc  = o_z    + intermediate_size
    o_dt   = o_hbc  + conv_dim

    groups_time_state_size = n_groups * ssm_state_size
    h_hidden = o_hbc
    h_B      = h_hidden + intermediate_size
    h_C      = h_B      + groups_time_state_size

    # ---- build B/C blocks
    W_B = _qk_to_grouped_bc(Wk_full, num_heads=num_heads, head_dim=qk_head_dim,
                            n_groups=n_groups, ssm_state_size=ssm_state_size)
    W_C = _qk_to_grouped_bc(Wq,      num_heads=num_heads, head_dim=qk_head_dim,
                            n_groups=n_groups, ssm_state_size=ssm_state_size)

    # ---- V block into hidden stream
    # Need [intermediate_size, hidden_size]
    if Wv_full.shape[0] != intermediate_size:
        if Wv_full.shape[0] > intermediate_size:
            Wv_block = Wv_full[:intermediate_size, :]
        else:
            Wv_block = torch.zeros(intermediate_size, Wv_full.shape[1], device=device, dtype=dtype)
            Wv_block[:Wv_full.shape[0], :] = Wv_full
    else:
        Wv_block = Wv_full

    # ---- copy into student mixer
    with torch.no_grad():
        # keep dt and gate “neutral” (let dt_bias / A_log init dominate early)
        mixer.in_proj.weight[o_z:o_z+intermediate_size, :].zero_()
        mixer.in_proj.weight[o_dt:o_dt+num_heads, :].zero_()
        if mixer.in_proj.bias is not None:
            mixer.in_proj.bias[o_z:o_z+intermediate_size].zero_()
            mixer.in_proj.bias[o_dt:o_dt+num_heads].zero_()

        # write V, B, C
        mixer.in_proj.weight[h_hidden:h_hidden+intermediate_size, :].copy_(Wv_block)
        mixer.in_proj.weight[h_B:h_B+groups_time_state_size, :].copy_(W_B)
        mixer.in_proj.weight[h_C:h_C+groups_time_state_size, :].copy_(W_C)

        if mixer.in_proj.bias is not None:
            # safest: zero these biases
            mixer.in_proj.bias[h_hidden:h_hidden+intermediate_size].zero_()
            mixer.in_proj.bias[h_B:h_B+groups_time_state_size].zero_()
            mixer.in_proj.bias[h_C:h_C+groups_time_state_size].zero_()

        # copy output projection if compatible
        Wo = Wo.to(device=mixer.out_proj.weight.device, dtype=mixer.out_proj.weight.dtype)
        if mixer.out_proj.weight.shape == Wo.shape:
            mixer.out_proj.weight.copy_(Wo)
            if mixer.out_proj.bias is not None and bo is not None:
                mixer.out_proj.bias.copy_(bo.to(device=mixer.out_proj.bias.device, dtype=mixer.out_proj.bias.dtype))
        elif mixer.out_proj.weight.shape == Wo.t().shape:
            mixer.out_proj.weight.copy_(Wo.t())
            if mixer.out_proj.bias is not None and bo is not None:
                mixer.out_proj.bias.copy_(bo.to(device=mixer.out_proj.bias.device, dtype=mixer.out_proj.bias.dtype))
        # else: leave out_proj as-is

class LlamaMamba2Config(Mamba2Config):
    model_type = "llama_mamba2"

    def __init__(self, rms_norm_eps: float = 1e-6, mlp_bias: bool = False,
                 intermediate_size: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.rms_norm_eps = rms_norm_eps
        self.mlp_bias = mlp_bias
        self.intermediate_size = intermediate_size

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Mamba2Mixer(FLAMamba2):
    def forward(
        self,
        hidden_states,
        cache_params = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
        dtype = hidden_states.dtype
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)

    @classmethod
    def from_config(cls, config, layer_idx=None):
        expand = 1
        return cls(
            num_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            head_dim = config.hidden_size // (config.num_attention_heads),
            state_size=128,
            n_groups=1,
            expand=expand,
            layer_idx=layer_idx,
            rms_norm=False,
            # hidden_act=None
            # **config.to_dict(),
        )


class Mamba2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.self_attn = Mamba2Mixer.from_config(config, layer_idx=layer_idx)
        # config.layer_norm_epsilon = 1e-5
        # config.num_heads = config.num_attention_heads
        # config.state_size = 128
        # config.expand = 1
        # config.fuse_cross_entropy = False
        # config.chunk_size = config.num_heads
        # self.self_attn = HFMamba2Mixer(config, layer_idx=layer_idx)
        # self.self_attn = Mamba2Mixer(config, layer_idx=layer_idx)

        self.mlp = llama.LlamaMLP(config)
        self.input_layernorm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(
        self,
        hidden_states,
        cache_params = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,
        use_cache: Optional[bool] = None,
        output_attentions=False,
        **kwargs
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # outputs = self.mixer(
        # hidden_states = self.self_attn(
        hidden_states = self.self_attn(
            hidden_states,
            # cache_params=cache_params,
            # cache_position=cache_position,
            attention_mask=attention_mask,
            # return_mixer_matrix=output_attentions,
        )

        if output_attentions:
            return hidden_states, None, None

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, None)

    def _forward(
        self,
        hidden_states,
        cache_params = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,
        use_cache: Optional[bool] = None,
        output_attentions=False,
        **kwargs
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # outputs = self.mixer(
        # hidden_states = self.self_attn(
        attn_output = self.self_attn(
            hidden_states,
            # cache_params=cache_params,
            # cache_position=cache_position,
            attention_mask=attention_mask,
            # return_mixer_matrix=output_attentions,
        )

        if output_attentions:
            return hidden_states, None, None

        # hidden_states = residual + hidden_states

        attn_output = self.post_attention_layernorm(attn_output)

        residual = hidden_states

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states + attn_output

        return (hidden_states, None)


class LlamaMambaPreTrainedModel(PreTrainedModel):
    config_class = LlamaMamba2Config
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, Mamba2Mixer):

            # --- A_log ---
            A = torch.empty(module.num_heads, dtype=torch.float32).uniform_(0, 16)
            with torch.no_grad():
                A_log = torch.log(A)
                if isinstance(module.A_log, DTensor):
                    A_log = tensor_to_dtensor(
                        tensor=A_log,
                        device_mesh=module.A_log.device_mesh,
                        current_placement=[Replicate()] * len(module.A_log.placements),
                        desired_placement=module.A_log.placements,
                        run_check=True,
                    )

                module.A_log.copy_(A_log)

            module.A_log._no_weight_decay = True

            # --- D ---
            nn.init.ones_(module.D)
            module.D._no_weight_decay = True

            # --- dt_bias ---
            dt = torch.exp(
                torch.rand(self.config.num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min),
            ).clamp(min=self.config.time_step_floor)

            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                if isinstance(module.dt_bias, DTensor):
                    inv_dt = tensor_to_dtensor(
                        tensor=inv_dt,
                        device_mesh=module.dt_bias.device_mesh,
                        current_placement=[Replicate()] * len(module.dt_bias.placements),
                        desired_placement=module.dt_bias.placements,
                        run_check=True,
                    )

                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True
        # if isinstance(module, nn.Linear):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.bias is not None:
        #         module.bias.data.zero_()
        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()


class Mamba2Model(LlamaMambaPreTrainedModel):
    
    config_class = LlamaMamba2Config

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values = None,
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

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        # if use_cache and not isinstance(past_key_values, Cache):
        #     past_key_values = Cache.from_legacy_cache(past_key_values)

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
                hidden_states, attn  = layer( 
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



class Mamba2ForCausalLM(LlamaMambaPreTrainedModel):
    _tied_weights_keys = {}
    config_class = LlamaMamba2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = Mamba2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_it(self):
        # Freeze everything except the token mixer
        for param in self.parameters():
            param.requires_grad = False

        for layer in self.model.layers:
            for param in layer.self_attn.parameters():
                param.requires_grad = True

    def _copy_from_teacher(self, teacher, a=False, b=False):
        assert len(self.model.layers) == len(teacher.model.layers)

        self.model.embeddings.weight.data.copy_(teacher.get_input_embeddings().weight.data)
        self.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
        self.lm_head.weight.data.copy_(teacher.get_output_embeddings().weight.data)

        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.layers):
            # self_layer.self_attn.load_state_dict(teacher_layer.self_attn.state_dict())
            self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict())
            self_layer.input_layernorm.load_state_dict(teacher_layer.input_layernorm.state_dict())
            self_layer.post_attention_layernorm.load_state_dict(teacher_layer.post_attention_layernorm.state_dict())

    def copy_from_teacher(self, teacher, a=False, b=False):
        assert len(self.model.layers) == len(teacher.model.layers)

        self.model.embeddings.weight.data.copy_(teacher.get_input_embeddings().weight.data)
        self.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
        self.lm_head.weight.data.copy_(teacher.get_output_embeddings().weight.data)

        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.layers):
            # Copy MLP and norms as before
            self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict())
            self_layer.input_layernorm.load_state_dict(teacher_layer.input_layernorm.state_dict())
            self_layer.post_attention_layernorm.load_state_dict(teacher_layer.post_attention_layernorm.state_dict())

            # NEW: copy teacher attention projections -> student Mamba2Mixer
            _copy_attn_to_mamba2_mixer(self_layer.self_attn, teacher_layer.self_attn)


    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_logits_to_keep: Optional[int] = 0,
        **kwargs,  # for now we need this for generation
        ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            # cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:])

        loss = None
        if labels is not None:
            if self.config.fuse_cross_entropy:
                if fuse_linear_and_cross_entropy:
                    loss_fct = FusedLinearCrossEntropyLoss()
                else:
                    loss_fct = FusedCrossEntropyLoss(inplace_backward=True)
            else:
                loss_fct = nn.CrossEntropyLoss()
            # Enable model parallelism
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = loss_fct(hidden_states.view(-1, self.config.hidden_size),
                                labels.view(-1),
                                self.lm_head.weight,
                                self.lm_head.bias)
            else:
                loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            # cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
        )

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
AutoConfig.register(LlamaMamba2Config.model_type, LlamaMamba2Config)
AutoModel.register(LlamaMamba2Config, Mamba2Model)
AutoModelForCausalLM.register(LlamaMamba2Config, Mamba2ForCausalLM)
# AutoModelForSequenceClassification.register(Mamba2Config, Mamba2ForSequenceClassification)

StudentConfig = LlamaMamba2Config
StudentModel = Mamba2ForCausalLM
