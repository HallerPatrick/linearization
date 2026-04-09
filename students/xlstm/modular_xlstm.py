from typing import Optional
import torch

import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config, OPTConfig, Gemma3TextConfig, LlamaConfig
from transformers.modeling_rope_utils import rope_config_validation


from xlstm.xlstm_large.model import (
    BackendModeType,
    ChunkwiseKernelType,
    DtypeType,
    SequenceKernelType,
    StepKernelType,
    WeightModeType,
    round_up_to_next_multiple_of,
    xLSTMLargeConfig,
    mLSTMStateType,
    soft_cap,
    # mLSTMLayer,
    mLSTMLayerConfig as _mLSTMLayerConfig,
    mLSTMBackendConfig,
    xLSTMLargeConfig,
    mLSTMLayerStateType,
    mLSTMBackend,
    MultiHeadLayerNorm,
)

from models.swa import causal_sliding_window_attention

def standardize_rope_params(config, rope_theta: float | dict[str, float] | None = None):
    """
    Helper to standardize the config's rope params field by ensuring the params are defined for each
    later type. For old model the fn will duplicate a single rope param in each layer type (backward compatibility)
    """
    rope_parameters = getattr(config, "rope_parameters", None)
    layer_types = getattr(config, "layer_types", None)
    if rope_theta is None:
        rope_theta = getattr(config, "rope_theta", None)

    # Case 1: one RoPE theat = one RoPE param per model without nesting
    if not isinstance(rope_theta, dict):
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        else:
            # BC: if there is a 'type' field, copy it it to 'rope_type'.
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            rope_theta = rope_parameters.get("rope_theta") or rope_theta
            rope_parameters.update({"rope_theta": rope_theta, "rope_type": rope_type})
        config.rope_parameters = rope_parameters

    # Case 2: different RoPE for each layer as nested dict
    else:
        rope_parameters_per_layer_type = {}
        for layer_type in layer_types:
            if rope_parameters is None:
                rope_parameters_per_layer_type[layer_type] = {
                    "rope_type": "default",
                    "rope_theta": rope_theta[layer_type],
                }
            else:
                is_field_in_new_format = any(layer_type in rope_parameters for layer_type in layer_types)
                if not is_field_in_new_format:
                    curr_rope_type = rope_parameters.get("rope_type", rope_parameters.get("type"))
                    rope_parameters_per_layer_type[layer_type] = {
                        **rope_parameters,
                        "rope_type": curr_rope_type,
                        "rope_theta": rope_theta[layer_type],
                    }
                else:
                    curr_rope_type = rope_parameters[layer_type].get(
                        "rope_type", rope_parameters[layer_type].get("type")
                    )
                    rope_parameters_per_layer_type[layer_type] = {
                        **rope_parameters[layer_type],
                        "rope_type": curr_rope_type,
                        "rope_theta": rope_theta[layer_type],
                    }
            config.rope_parameters = rope_parameters_per_layer_type

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class xQwen3Config(Qwen3Config):
    model_type = "xqwen3"
    def __init__(
        self,
        chunkwise_kernel: ChunkwiseKernelType = "chunkwise--native_autograd",
        sequence_kernel: SequenceKernelType = "native_sequence__native",
        step_kernel: StepKernelType = "native",
        # nedded to enable generation
        mode: BackendModeType = "inference",
        chunk_size: int = 64,
        # needed to be true for generation
        return_last_states: bool = True,
        autocast_kernel_dtype: DtypeType = "bfloat16",
        eps: float = 1e-6,
        inference_state_dtype: DtypeType = "float32",
        # feedforward
        ffn_proj_factor: float = 2.667,
        ffn_round_up_to_multiple_of: int = 64,
        # capping
        gate_soft_cap: float = 15.0,
        output_logit_soft_cap: float = 30.0,
        # weights
        weight_mode: WeightModeType = "single",
        **qwen3_kwargs,
    ):
        self.chunkwise_kernel = chunkwise_kernel
        self.sequence_kernel = sequence_kernel
        self.step_kernel = step_kernel
        self.mode = mode
        self.chunk_size = chunk_size
        self.return_last_states = return_last_states
        self.autocast_kernel_dtype = autocast_kernel_dtype
        self.eps = eps
        self.inference_state_dtype = inference_state_dtype
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of
        self.gate_soft_cap = gate_soft_cap
        self.output_logit_soft_cap = output_logit_soft_cap
        self.weight_mode = weight_mode
        super().__init__(**qwen3_kwargs)

class xLlamaConfig(LlamaConfig):
    model_type = "xllama"
    def __init__(
        self,
        chunkwise_kernel: ChunkwiseKernelType = "chunkwise--native_autograd",
        sequence_kernel: SequenceKernelType = "native_sequence__native",
        step_kernel: StepKernelType = "native",
        # nedded to enable generation
        mode: BackendModeType = "inference",
        chunk_size: int = 64,
        # needed to be true for generation
        return_last_states: bool = True,
        autocast_kernel_dtype: DtypeType = "bfloat16",
        eps: float = 1e-6,
        inference_state_dtype: DtypeType = "float32",
        # feedforward
        ffn_proj_factor: float = 2.667,
        ffn_round_up_to_multiple_of: int = 64,
        # capping
        gate_soft_cap: float = 15.0,
        output_logit_soft_cap: float = 30.0,
        # weights
        weight_mode: WeightModeType = "single",
        rope_parameters = None,
        apply_rope = False,
        **kwargs,
    ):
        self.chunkwise_kernel = chunkwise_kernel
        self.sequence_kernel = sequence_kernel
        self.step_kernel = step_kernel
        self.mode = mode
        self.chunk_size = chunk_size
        self.return_last_states = return_last_states
        self.autocast_kernel_dtype = autocast_kernel_dtype
        self.eps = eps
        self.inference_state_dtype = inference_state_dtype
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of
        self.gate_soft_cap = gate_soft_cap
        self.output_logit_soft_cap = output_logit_soft_cap
        self.weight_mode = weight_mode
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters
        rope_theta = kwargs.get("rope_theta", 10000.0)
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)
        self.apply_rope = apply_rope
        super().__init__(**kwargs)
        self.num_heads = self.num_attention_heads

class xOPTConfig(OPTConfig):
    model_type = "xopt"
    def __init__(
        self,
        chunkwise_kernel: ChunkwiseKernelType = "chunkwise--native_autograd",
        sequence_kernel: SequenceKernelType = "native_sequence__native",
        step_kernel: StepKernelType = "native",
        # nedded to enable generation
        mode: BackendModeType = "inference",
        chunk_size: int = 64,
        # needed to be true for generation
        return_last_states: bool = True,
        autocast_kernel_dtype: DtypeType = "bfloat16",
        eps: float = 1e-6,
        inference_state_dtype: DtypeType = "float32",
        # feedforward
        ffn_proj_factor: float = 2.667,
        ffn_round_up_to_multiple_of: int = 64,
        # capping
        gate_soft_cap: float = 15.0,
        output_logit_soft_cap: float = 30.0,
        # weights
        weight_mode: WeightModeType = "single",
        rope_parameters = None,
        **qwen3_kwargs,
    ):
        self.chunkwise_kernel = chunkwise_kernel
        self.sequence_kernel = sequence_kernel
        self.step_kernel = step_kernel
        self.mode = mode
        self.chunk_size = chunk_size
        self.return_last_states = return_last_states
        self.autocast_kernel_dtype = autocast_kernel_dtype
        self.eps = eps
        self.inference_state_dtype = inference_state_dtype
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of
        self.gate_soft_cap = gate_soft_cap
        self.output_logit_soft_cap = output_logit_soft_cap
        self.weight_mode = weight_mode
        self.rope_parameters = rope_parameters
        super().__init__(**qwen3_kwargs)

class LAOPTConfig(xOPTConfig):
    model_type = "laopt"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "laopt"

class xGemma3Config(Gemma3TextConfig):
    model_type = "xgemma3"
    def __init__(
        self,
        chunkwise_kernel: ChunkwiseKernelType = "chunkwise--native_autograd",
        sequence_kernel: SequenceKernelType = "native_sequence__native",
        step_kernel: StepKernelType = "native",
        # nedded to enable generation
        mode: BackendModeType = "inference",
        chunk_size: int = 64,
        # needed to be true for generation
        return_last_states: bool = True,
        autocast_kernel_dtype: DtypeType = "bfloat16",
        eps: float = 1e-6,
        inference_state_dtype: DtypeType = "float32",
        # feedforward
        ffn_proj_factor: float = 2.667,
        ffn_round_up_to_multiple_of: int = 64,
        # capping
        gate_soft_cap: float = 15.0,
        output_logit_soft_cap: float = 30.0,
        # weights
        weight_mode: WeightModeType = "single",
        **qwen3_kwargs,
    ):
        self.chunkwise_kernel = chunkwise_kernel
        self.sequence_kernel = sequence_kernel
        self.step_kernel = step_kernel
        self.mode = mode
        self.chunk_size = chunk_size
        self.return_last_states = return_last_states
        self.autocast_kernel_dtype = autocast_kernel_dtype
        self.eps = eps
        self.inference_state_dtype = inference_state_dtype
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of
        self.gate_soft_cap = gate_soft_cap
        self.output_logit_soft_cap = output_logit_soft_cap
        self.weight_mode = weight_mode
        super().__init__(**qwen3_kwargs)

class mLSTMLayerConfig(_mLSTMLayerConfig):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 8,
        head_dim: int = 64,
        use_bias: bool = False,
        norm_eps: float = 1e-6,
        norm_reduction_force_float32: bool = True,
        qk_dim_factor: float = 0.5,
        v_dim_factor: float = 1.0,
        num_key_value_heads: int = 1,
        gate_soft_cap: float = 15.0,
        weight_mode: WeightModeType = "single",
        mlstm_backend: mLSTMBackendConfig = mLSTMBackendConfig(),
        apply_rope: bool = False,
    ):
        super().__init__(
            embedding_dim=hidden_size,
            num_heads=num_heads,
            use_bias=use_bias,
            norm_eps=norm_eps,
            norm_reduction_force_float32=norm_reduction_force_float32,
            qk_dim_factor=qk_dim_factor,
            v_dim_factor=v_dim_factor,
            gate_soft_cap=gate_soft_cap,
            weight_mode=weight_mode,
            mlstm_backend=mlstm_backend,
        )
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.apply_rope = apply_rope

    @classmethod
    def from_xgemma3_config(cls, config: xGemma3Config):
        return cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            use_bias=config.attention_bias,
            norm_eps=config.rms_norm_eps,
            norm_reduction_force_float32=True,
            qk_dim_factor=1,
            v_dim_factor=1,
            num_key_value_heads=config.num_key_value_heads,
            gate_soft_cap=config.gate_soft_cap,
            weight_mode=config.weight_mode,
            mlstm_backend=mLSTMBackendConfig(
                chunkwise_kernel=config.chunkwise_kernel,
                sequence_kernel=config.sequence_kernel,
                step_kernel=config.step_kernel,
                mode=config.mode,
                chunk_size=config.chunk_size,
                return_last_states=config.return_last_states,
                autocast_kernel_dtype=config.autocast_kernel_dtype,
                eps=config.eps,
                inference_state_dtype=config.inference_state_dtype,
            )
        )

    @classmethod
    def from_xqwen3_config(cls, config: xQwen3Config):
        return cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            use_bias=config.attention_bias,
            norm_eps=config.rms_norm_eps,
            norm_reduction_force_float32=True,
            qk_dim_factor=1,
            v_dim_factor=1,
            num_key_value_heads=config.num_key_value_heads,
            gate_soft_cap=config.gate_soft_cap,
            weight_mode=config.weight_mode,
            mlstm_backend=mLSTMBackendConfig(
                chunkwise_kernel=config.chunkwise_kernel,
                sequence_kernel=config.sequence_kernel,
                step_kernel=config.step_kernel,
                mode=config.mode,
                chunk_size=config.chunk_size,
                return_last_states=config.return_last_states,
                autocast_kernel_dtype=config.autocast_kernel_dtype,
                eps=config.eps,
                inference_state_dtype=config.inference_state_dtype,
            )
        )

    @classmethod
    def from_xopt_config(cls, config: xOPTConfig):
        return cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            use_bias=config.enable_bias,
            # norm_eps=config.rms_norm_eps,
            norm_reduction_force_float32=True,
            qk_dim_factor=1,
            v_dim_factor=1,
            num_key_value_heads=config.num_attention_heads,
            gate_soft_cap=config.gate_soft_cap,
            weight_mode=config.weight_mode,
            mlstm_backend=mLSTMBackendConfig(
                chunkwise_kernel=config.chunkwise_kernel,
                sequence_kernel=config.sequence_kernel,
                step_kernel=config.step_kernel,
                mode=config.mode,
                chunk_size=config.chunk_size,
                return_last_states=config.return_last_states,
                autocast_kernel_dtype=config.autocast_kernel_dtype,
                eps=config.eps,
                inference_state_dtype=config.inference_state_dtype,
            )
        )

    @classmethod
    def from_xllama_config(cls, config: xLlamaConfig):
        return cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            use_bias=config.attention_bias,
            norm_eps=config.rms_norm_eps,
            norm_reduction_force_float32=True,
            qk_dim_factor=1,
            v_dim_factor=1,
            num_key_value_heads=config.num_key_value_heads,
            gate_soft_cap=config.gate_soft_cap,
            weight_mode=config.weight_mode,
            apply_rope=getattr(config, "apply_rope", False),
            mlstm_backend=mLSTMBackendConfig(
                chunkwise_kernel=config.chunkwise_kernel,
                sequence_kernel=config.sequence_kernel,
                step_kernel=config.step_kernel,
                mode=config.mode,
                chunk_size=config.chunk_size,
                return_last_states=config.return_last_states,
                autocast_kernel_dtype=config.autocast_kernel_dtype,
                eps=config.eps,
                inference_state_dtype=config.inference_state_dtype,
            )
        )

class mLSTMIntraLayer(nn.Module):
    def __init__(self, config: mLSTMLayerConfig, intra_layer_swa: bool = False):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_heads // config.num_key_value_heads
        self.intra_layer_swa = intra_layer_swa

        self.v_dim = int(config.embedding_dim * config.v_dim_factor)
        self.qk_dim = int(config.embedding_dim * config.qk_dim_factor)

        if self.config.weight_mode == "single":
            self.q = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads * self.head_dim,
                bias=self.config.use_bias,
            )
            self.k = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=config.num_key_value_heads * self.head_dim,
                bias=self.config.use_bias,
            )
            self.v = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=config.num_key_value_heads * self.head_dim,
                bias=self.config.use_bias,
            )

            self.ogate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.head_dim * self.config.num_heads,
                bias=self.config.use_bias,
            )
            self.igate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads,
                bias=True,
            )
            self.fgate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads,
                bias=True,
            )
        elif self.config.weight_mode == "fused":
            self.qkv_opreact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=2 * self.qk_dim + 2 * self.v_dim,
                bias=self.config.use_bias,
            )
            self.ifgate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=2 * self.config.num_heads,
                bias=True,
            )

        self.ogate_act_fn = nn.Sigmoid()
        self.mlstm_backend = mLSTMBackend(config=self.config.mlstm_backend)

        self.multihead_norm = MultiHeadLayerNorm(
            num_heads=self.config.num_heads,
            head_dim=self.head_dim,
            eps=self.config.norm_eps,
            use_weight=True,
            use_bias=self.config.use_bias,
            force_float32_reductions=self.config.norm_reduction_force_float32,
        )
        self.out_proj = nn.Linear(
            in_features=self.config.num_heads * self.head_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.use_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: mLSTMLayerStateType | None = None,
        output_attentions: bool = False,
        diagnostics=None,
        **kwargs,
    ) -> tuple[torch.Tensor, mLSTMLayerStateType | None, torch.Tensor | None]:
        assert hidden_states.ndim == 3, f"Input must have shape [B, S, D], got {hidden_states.shape}"
        B, S, _ = hidden_states.shape
        if self.config.weight_mode == "single":
            q = self.q(hidden_states)
            k = self.k(hidden_states)
            v = self.v(hidden_states)

            o_preact = self.ogate_preact(hidden_states)
            i_preact = soft_cap(
                self.igate_preact(hidden_states), cap_value=self.config.gate_soft_cap
            )
            f_preact = soft_cap(
                self.fgate_preact(hidden_states), cap_value=self.config.gate_soft_cap
            )

        elif self.config.weight_mode == "fused":
            qkv_opreact = self.qkv_opreact(hidden_states)
            q, k, v, o_preact = torch.tensor_split(
                qkv_opreact,
                (
                    self.qk_dim,
                    2 * self.qk_dim,
                    2 * self.qk_dim + self.v_dim,
                ),
                dim=-1,
            )

            if_preact = soft_cap(
                self.ifgate_preact(hidden_states), cap_value=self.config.gate_soft_cap
            )
            i_preact, f_preact = torch.tensor_split(
                if_preact, (self.config.num_heads,), dim=-1
            )

        q = q.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, S, self.config.num_key_value_heads, -1).transpose(1, 2)
        v = v.reshape(B, S, self.config.num_key_value_heads, -1).transpose(1, 2)

        if self.intra_layer_swa:
            sq, sk, sv = q, k, v
            cos, sin = kwargs["position_embeddings"]
            sq, sk = apply_rotary_pos_emb(sq, sk, cos, sin)

        if hasattr(self.config, "apply_rope") \
                and self.config.apply_rope \
                and "position_embeddings" in kwargs \
                and kwargs["position_embeddings"] is not None:
            cos, sin = kwargs["position_embeddings"]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        i_preact = i_preact.transpose(1, 2)
        f_preact = f_preact.transpose(1, 2)
        if state is None:
            c_initial, n_initial, m_initial = None, None, None
        else:
            c_initial, n_initial, m_initial = state

        if output_attentions:
            # TODO: Might use stable version? https://github.com/NX-AI/mlstm_kernels/blob/main/mlstm_kernels/torch/parallel/native_stablef/fw.py#L15
            _device = q.device
            B, NH, S, DHQK = q.shape
            assert k.shape == (B, NH, S, DHQK)
            assert i_preact.shape == (B, NH, S)
            assert f_preact.shape == (B, NH, S)

            vecLogSigF = F.logsigmoid(f_preact)  # (B, NH, S)

            vecLogSigF_cumsum = vecLogSigF.cumsum(-1)

            matLogSigF = (
                vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]
            )

            ltr = torch.tril(
                torch.ones(
                    (S, S),
                    dtype=torch.bool,
                    device=_device,
                )
            )

            matLogSigF_mask = torch.where(ltr, matLogSigF, -float("inf"))

            # TODO: Based on the paper, i = i_preact.exp(), but this is missing here
            # maybe also put it in log space?
            matLogD = matLogSigF_mask + i_preact[:, :, None, :]

            vecM, _ = torch.max(matLogD, dim=-1, keepdim=True)  # (B, NH, S, 1)
            matLogD_stabilized = matLogD - vecM

            matD = torch.exp(matLogD_stabilized)  # (B, NH, S, S)

            matS = (q @ k.transpose(-2, -1)) * (DHQK**-0.5)  # (B, NH, S, S)
            matCtilde = matS * matD  # (B, NH, S, S)
            vecN = torch.maximum(
                matCtilde.sum(dim=-1, keepdim=True).abs(), torch.exp(-vecM)
            )  # (B, NH, S, 1)
            # (B, NH, S, S)
            eps: float = 1e-6
            C = matCtilde / (vecN + eps)
            return None, None, C
        
        if c_initial is not None and c_initial.device != q.device:
            c_initial = c_initial.to(q.device)
            n_initial = n_initial.to(q.device)
            m_initial = m_initial.to(q.device)

        h, state = self.mlstm_backend(
            q=q,
            k=k,
            v=v,
            i=i_preact,
            f=f_preact,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
        )
        if diagnostics is not None and hasattr(self, "layer_idx"):
            alpha = torch.sigmoid(f_preact)
            beta = torch.sigmoid(i_preact)
            diagnostics.record_gate(self.layer_idx, alpha, beta)
            if isinstance(state, tuple) and isinstance(state[0], torch.Tensor):
                prev_state = c_initial if isinstance(c_initial, torch.Tensor) else None
                diagnostics.record_state(self.layer_idx, state[0], prev_state)
        expected_h_shape = (
            B,
            self.config.num_heads,
            S,
            self.config.head_dim,
        )
        assert h.shape == expected_h_shape, (
            f"Got {h.shape}, expected {expected_h_shape}"
        )

        if self.intra_layer_swa:
            attn_output = causal_sliding_window_attention(
                sq,
                sk,
                sv,
                window_size=512,
                with_memory=True
            ).transpose(1, 2).reshape(B, S, -1)

        h = h.transpose(1, 2)
        h_norm = self.multihead_norm(h)
        h_norm = h_norm.reshape(B, S, -1)

        if self.intra_layer_swa:
            hidden_state = 0.5 * attn_output + 0.5 * h_norm
            # hidden_state = attn_output
            # norm_m = torch.norm(h_norm, dim=-1).mean().item()
            # norm_s = torch.norm(attn_output, dim=-1).mean().item()

            # cos_m = F.cosine_similarity(h_norm, hidden_state, dim=-1).mean().item()
            # cos_s = F.cosine_similarity(attn_output, hidden_state, dim=-1).mean().item()
            # print(
            #     f"Intra-layer SWA stats: ||h_norm||={norm_m:.4f}, ||swa_attn||={norm_s:.4f}, "
            #     f"cos(h_norm, hidden_state)={cos_m:.4f}, cos(swa_attn, hidden_state)={cos_s:.4f}"
            #                 )

            h_norm = hidden_state

        h_out = self.ogate_act_fn(o_preact) * h_norm

        y = self.out_proj(h_out)
        return y, state, None

class mLSTMLayer(nn.Module):
    def __init__(self, config: mLSTMLayerConfig, intra_layer_swa: bool = False):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_heads // config.num_key_value_heads

        self.v_dim = int(config.embedding_dim * config.v_dim_factor)
        self.qk_dim = int(config.embedding_dim * config.qk_dim_factor)

        if self.config.weight_mode == "single":
            self.q = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads * self.head_dim,
                bias=self.config.use_bias,
            )
            self.k = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=config.num_key_value_heads * self.head_dim,
                bias=self.config.use_bias,
            )
            self.v = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=config.num_key_value_heads * self.head_dim,
                bias=self.config.use_bias,
            )

            self.ogate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.head_dim * self.config.num_heads,
                bias=self.config.use_bias,
            )
            self.igate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads,
                bias=True,
            )
            self.fgate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads,
                bias=True,
            )
        elif self.config.weight_mode == "fused":
            self.qkv_opreact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=2 * self.qk_dim + 2 * self.v_dim,
                bias=self.config.use_bias,
            )
            self.ifgate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=2 * self.config.num_heads,
                bias=True,
            )

        self.ogate_act_fn = nn.Sigmoid()
        self.mlstm_backend = mLSTMBackend(config=self.config.mlstm_backend)

        self.multihead_norm = MultiHeadLayerNorm(
            num_heads=self.config.num_heads,
            head_dim=self.head_dim,
            eps=self.config.norm_eps,
            use_weight=True,
            use_bias=self.config.use_bias,
            force_float32_reductions=self.config.norm_reduction_force_float32,
        )
        self.out_proj = nn.Linear(
            in_features=self.config.num_heads * self.head_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.use_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: mLSTMLayerStateType | None = None,
        output_attentions: bool = False,
        diagnostics=None,
        **kwargs,
    ) -> tuple[torch.Tensor, mLSTMLayerStateType | None, torch.Tensor | None]:
        assert hidden_states.ndim == 3, f"Input must have shape [B, S, D], got {hidden_states.shape}"
        B, S, _ = hidden_states.shape
        if self.config.weight_mode == "single":
            q = self.q(hidden_states)
            k = self.k(hidden_states)
            v = self.v(hidden_states)

            o_preact = self.ogate_preact(hidden_states)
            i_preact = soft_cap(
                self.igate_preact(hidden_states), cap_value=self.config.gate_soft_cap
            )
            f_preact = soft_cap(
                self.fgate_preact(hidden_states), cap_value=self.config.gate_soft_cap
            )

        elif self.config.weight_mode == "fused":
            qkv_opreact = self.qkv_opreact(hidden_states)
            q, k, v, o_preact = torch.tensor_split(
                qkv_opreact,
                (
                    self.qk_dim,
                    2 * self.qk_dim,
                    2 * self.qk_dim + self.v_dim,
                ),
                dim=-1,
            )

            if_preact = soft_cap(
                self.ifgate_preact(hidden_states), cap_value=self.config.gate_soft_cap
            )
            i_preact, f_preact = torch.tensor_split(
                if_preact, (self.config.num_heads,), dim=-1
            )

        q = q.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, S, self.config.num_key_value_heads, -1).transpose(1, 2)
        v = v.reshape(B, S, self.config.num_key_value_heads, -1).transpose(1, 2)

        if hasattr(self.config, "apply_rope") \
                and self.config.apply_rope \
                and "position_embeddings" in kwargs \
                and kwargs["position_embeddings"] is not None:
            cos, sin = kwargs["position_embeddings"]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        i_preact = i_preact.transpose(1, 2)
        f_preact = f_preact.transpose(1, 2)
        if state is None:
            c_initial, n_initial, m_initial = None, None, None
        else:
            c_initial, n_initial, m_initial = state

        if False:
            # TODO: Might use stable version? https://github.com/NX-AI/mlstm_kernels/blob/main/mlstm_kernels/torch/parallel/native_stablef/fw.py#L15
            _device = q.device
            B, NH, S, DHQK = q.shape
            assert k.shape == (B, NH, S, DHQK)
            assert i_preact.shape == (B, NH, S)
            assert f_preact.shape == (B, NH, S)

            vecLogSigF = F.logsigmoid(f_preact)  # (B, NH, S)

            vecLogSigF_cumsum = vecLogSigF.cumsum(-1)

            matLogSigF = (
                vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]
            )

            ltr = torch.tril(
                torch.ones(
                    (S, S),
                    dtype=torch.bool,
                    device=_device,
                )
            )

            matLogSigF_mask = torch.where(ltr, matLogSigF, -float("inf"))

            # TODO: Based on the paper, i = i_preact.exp(), but this is missing here
            # maybe also put it in log space?
            matLogD = matLogSigF_mask + i_preact[:, :, None, :]

            vecM, _ = torch.max(matLogD, dim=-1, keepdim=True)  # (B, NH, S, 1)
            matLogD_stabilized = matLogD - vecM

            matD = torch.exp(matLogD_stabilized)  # (B, NH, S, S)

            matS = (q @ k.transpose(-2, -1)) * (DHQK**-0.5)  # (B, NH, S, S)
            matCtilde = matS * matD  # (B, NH, S, S)
            vecN = torch.maximum(
                matCtilde.sum(dim=-1, keepdim=True).abs(), torch.exp(-vecM)
            )  # (B, NH, S, 1)
            # (B, NH, S, S)
            eps: float = 1e-6
            C = matCtilde / (vecN + eps)
            return None, None, C
        
        if c_initial is not None and c_initial.device != q.device:
            c_initial = c_initial.to(q.device)
            n_initial = n_initial.to(q.device)
            m_initial = m_initial.to(q.device)

        h, state = self.mlstm_backend(
            q=q,
            k=k,
            v=v,
            i=i_preact,
            f=f_preact,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
        )
        if diagnostics is not None and hasattr(self, "layer_idx"):
            alpha = torch.sigmoid(f_preact)
            beta = torch.sigmoid(i_preact)
            diagnostics.record_gate(self.layer_idx, alpha, beta)
            if isinstance(state, tuple) and isinstance(state[0], torch.Tensor):
                prev_state = c_initial if isinstance(c_initial, torch.Tensor) else None
                diagnostics.record_state(self.layer_idx, state[0], prev_state)
        expected_h_shape = (
            B,
            self.config.num_heads,
            S,
            self.config.head_dim,
        )
        assert h.shape == expected_h_shape, (
            f"Got {h.shape}, expected {expected_h_shape}"
        )

        h = h.transpose(1, 2)
        h_norm = self.multihead_norm(h)
        h_norm = h_norm.reshape(B, S, -1)
        h_out = self.ogate_act_fn(o_preact) * h_norm

        y = self.out_proj(h_out)
        return y, state, None


class xLSTMCache:
    """
    Cache for xLSTM model which does not have attention mechanism and key value states.

    Arguments:
        config (`PreTrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The batch size with which the model will be used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype

    Example:

        ```python
        >>> from transformers import AutoTokenizer, xLSTMForCausalLM, xLSTMCache

        >>> model = xLSTMForCausalLM.from_pretrained("NX-AI/xLSTM-7b")
        >>> tokenizer = xLSTMTokenizer.from_pretrained("NX-AI/xLSTM-7b")

        >>> inputs = tokenizer(text="I am an xLSTM", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_params = xLSTMCache(config=model.config, max_batch_size=1, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, cache_params=cache_params, use_cache=True)
        >>> outputs.cache_params
        xLSTMCache()
    """

    def __init__(
        self,
        config,
        max_batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.seqlen_offset = 0
        self._dtype = dtype
        self.config = config
        self.rnn_state = {
            layer: (
                torch.zeros(
                    [max_batch_size, config.num_attention_heads, config.head_dim, config.head_dim],
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros([max_batch_size, config.num_attention_heads, config.head_dim], dtype=dtype, device=device),
                torch.zeros([max_batch_size, config.num_attention_heads, 1], dtype=dtype, device=device),
            )
            for layer in range(config.num_hidden_layers)
        }

    def reset(self):
        self.rnn_state = {
            layer: (
                torch.zeros_like(self.rnn_state[layer][0]),
                torch.zeros_like(self.rnn_state[layer][1]),
                torch.zeros_like(self.rnn_state[layer][2]),
            )
            for layer in self.rnn_state
        }
