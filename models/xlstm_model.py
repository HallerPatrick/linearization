from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin

from transformers.utils import logging, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import transformers.models.llama.modeling_llama as llama


from transformers.modeling_flash_attention_utils import _flash_attention_forward
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

# from .utils import unroll_value_projection


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


logger = logging.get_logger(__name__)


def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    """Rounds up x to the next multiple of multiple_of."""
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)


class _xLlamaConfig(PretrainedConfig):
    model_type = "xsmolm"

    def __init__(
        self,
        vocab_size: int = 50304,
        embedding_dim: int = 4096,
        num_blocks: int = 32,
        num_heads: int = 8,
        use_bias: bool = False,
        norm_reduction_force_float32: bool = True,
        tie_word_embeddings: bool = False,
        add_out_norm: bool = True,
        norm_eps: float = 1e-6,
        # mlstm_layer
        qk_dim_factor: float = 0.5,
        v_dim_factor: float = 1.0,
        # mlstm backend
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
        # HF interface
        use_cache: bool = True,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        force_bos_token_insert: bool = True,
        max_inference_chunksize: int = 16384,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = embedding_dim
        self.num_blocks = num_blocks
        self.num_hidden_layers = num_blocks
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.add_out_norm = add_out_norm
        self.norm_eps = norm_eps
        self.norm_reduction_force_float32 = norm_reduction_force_float32
        # mlstm_layer
        self.qk_dim_factor = qk_dim_factor
        self.v_dim_factor = v_dim_factor
        # mlstm backend
        self.chunkwise_kernel = chunkwise_kernel
        self.sequence_kernel = sequence_kernel
        self.step_kernel = step_kernel
        self.mode = mode
        self.chunk_size = chunk_size
        self.return_last_states = return_last_states
        self.autocast_kernel_dtype = autocast_kernel_dtype
        self.eps = eps
        self.inference_state_dtype = inference_state_dtype
        # feedforward
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of
        # capping
        self.gate_soft_cap = gate_soft_cap
        self.output_logit_soft_cap = output_logit_soft_cap
        self.weight_mode = weight_mode

        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.force_bos_token_insert = force_bos_token_insert
        self.max_inference_chunksize = max_inference_chunksize

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def qk_dim(self):
        return round_up_to_next_multiple_of(
            self.embedding_dim * self.qk_dim_factor,
            multiple_of=64,
        )

    @property
    def v_dim(self):
        return round_up_to_next_multiple_of(
            self.embedding_dim * self.v_dim_factor,
            multiple_of=64,
        )

    @property
    def qk_head_dim(self):
        return self.qk_dim // self.num_heads

    @property
    def v_head_dim(self):
        return self.v_dim // self.num_heads

    def to_xlstm_block_config(self):
        return xLSTMLargeConfig(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            use_bias=self.use_bias,
            add_out_norm=self.add_out_norm,
            norm_eps=self.norm_eps,
            norm_reduction_force_float32=self.norm_reduction_force_float32,
            # mlstm_layer
            qk_dim_factor=self.qk_dim_factor,
            v_dim_factor=self.v_dim_factor,
            # mlstm backend
            chunkwise_kernel=self.chunkwise_kernel,
            sequence_kernel=self.sequence_kernel,
            step_kernel=self.step_kernel,
            mode=self.mode,
            chunk_size=self.chunk_size,
            return_last_states=self.return_last_states,
            autocast_kernel_dtype=self.autocast_kernel_dtype,
            eps=self.eps,
            inference_state_dtype=self.inference_state_dtype,
            # feedforward
            ffn_proj_factor=self.ffn_proj_factor,
            ffn_round_up_to_multiple_of=self.ffn_round_up_to_multiple_of,
            # capping
            gate_soft_cap=self.gate_soft_cap,
            output_logit_soft_cap=self.output_logit_soft_cap,
            weight_mode=self.weight_mode,
        )


class xLlamaConfig(_xLlamaConfig):
    rms_norm_eps: float = 1e-6
    mlp_bias: bool = False
    intermediate_size: int = 1024
    num_key_value_heads: int = 1


class xLSTMCache:
    """
    Cache / RNN State handler for xLSTM.

    Args:
        config: xLlamaConfig
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype
    """

    def __init__(
        self,
        config: xLlamaConfig,
        batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
    ):
        self.seqlen_offset = torch.tensor(0, dtype=torch.int64, device=device)
        self.dtype = dtype
        self.config = config
        self.rnn_state: mLSTMStateType = {
            layer: (
                torch.zeros(
                    [
                        batch_size,
                        config.num_heads,
                        config.qk_head_dim,
                        config.v_head_dim,
                    ],
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros(
                    [batch_size, config.num_heads, config.qk_head_dim],
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros(
                    [batch_size, config.num_heads, 1], dtype=dtype, device=device
                ),
            )
            for layer in range(config.num_hidden_layers)
        }
        self.rnn_state_initial = True

    def reset(self):
        self.rnn_state = {
            layer: (
                torch.zeros_like(self.rnn_state[layer][0]),
                torch.zeros_like(self.rnn_state[layer][1]),
                torch.zeros_like(self.rnn_state[layer][2]),
            )
            for layer in self.rnn_state
        }
        self.rnn_state_initial = True


class xLSTMPreTrainedModel(PreTrainedModel):
    """
    An abstract class for an interface to loading a pre-trained xLSTM model.
    """

    config_class = xLlamaConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["xLSTMBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        """Initialize the weights."""
        # TODO: this is a dummy, check with original settings.
        pass


@dataclass
class xLSTMOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor]
    cache_params: Optional[xLSTMCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class xLSTMCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[xLSTMCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class mLSTMLayerConfig(_mLSTMLayerConfig):
    def __init__(
        self,
        embedding_dim: int = 4096,
        num_heads: int = 8,
        use_bias: bool = False,
        norm_eps: float = 1e-6,
        norm_reduction_force_float32: bool = True,
        qk_dim_factor: float = 0.5,
        v_dim_factor: float = 1.0,
        num_key_value_heads: int = 1,
        gate_soft_cap: float = 15.0,
        weight_mode: WeightModeType = "single",
        mlstm_backend: mLSTMBackendConfig = mLSTMBackendConfig(),
    ):
        super().__init__(
            embedding_dim=embedding_dim,
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


class mLSTMLayer(nn.Module):
    def __init__(self, config: mLSTMLayerConfig):
        super().__init__()
        self.config = config

        self.head_dim = config.embedding_dim // config.num_heads
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
                out_features=self.config.embedding_dim,
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
            in_features=self.config.embedding_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.use_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: mLSTMLayerStateType | None = None,
        output_attentions: bool = False,
        output_attention_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, mLSTMLayerStateType | None]:
        assert x.ndim == 3, f"Input must have shape [B, S, D], got {x.shape}"
        B, S, _ = x.shape
        if self.config.weight_mode == "single":
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)

            o_preact = self.ogate_preact(x)
            i_preact = soft_cap(
                self.igate_preact(x), cap_value=self.config.gate_soft_cap
            )
            f_preact = soft_cap(
                self.fgate_preact(x), cap_value=self.config.gate_soft_cap
            )

        elif self.config.weight_mode == "fused":
            qkv_opreact = self.qkv_opreact(x)
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
                self.ifgate_preact(x), cap_value=self.config.gate_soft_cap
            )
            i_preact, f_preact = torch.tensor_split(
                if_preact, (self.config.num_heads,), dim=-1
            )

        q = q.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, S, self.config.num_key_value_heads, -1).transpose(1, 2)
        v = v.reshape(B, S, self.config.num_key_value_heads, -1).transpose(1, 2)

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
            return C

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
        expected_h_shape = (
            B,
            self.config.num_heads,
            S,
            self.v_dim // self.config.num_heads,
        )
        assert h.shape == expected_h_shape, (
            f"Got {h.shape}, expected {expected_h_shape}"
        )

        h = h.transpose(1, 2)
        h_norm = self.multihead_norm(h)
        h_norm = h_norm.reshape(B, S, -1)
        h_out = self.ogate_act_fn(o_preact) * h_norm

        y = self.out_proj(h_out)
        return y, state


class mLSTMBlock(nn.Module):
    def __init__(self, config: xLlamaConfig):
        super().__init__()
        self.config = config
        self.input_layernorm = llama.LlamaRMSNorm(
            config.embedding_dim, eps=config.norm_eps
        )
        self.token_mixer = mLSTMLayer(
            mLSTMLayerConfig(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                use_bias=config.use_bias,
                norm_eps=config.norm_eps,
                norm_reduction_force_float32=config.norm_reduction_force_float32,
                qk_dim_factor=config.qk_dim_factor,
                v_dim_factor=config.v_dim_factor,
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
                ),
            ),
        )
        self.mlp = llama.LlamaMLP(config)
        self.post_attention_layernorm = llama.LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: mLSTMStateType | None = None,
        output_attentions: bool = False,
        output_attention_hidden_states: bool = False,
        output_token_mixer: bool = False,
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if output_attention_hidden_states or output_token_mixer:
            output, _ = self.token_mixer(
                hidden_states,
                state,
                # output_attention_hidden_states=output_attention_hidden_states,
            )
            return output
        elif not output_attentions:
            hidden_states, state = self.token_mixer(hidden_states, state)
        else:
            attn = self.token_mixer(hidden_states, state, output_attentions)
            return None, attn

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, state


class xLSTMModel(xLSTMPreTrainedModel):
    config_class = xLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        # use config explicitly to mitigate unused variable tests
        xlstm_block_config = xLSTMLargeConfig(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            use_bias=config.use_bias,
            add_out_norm=config.add_out_norm,
            norm_eps=config.rms_norm_eps,
            norm_reduction_force_float32=config.norm_reduction_force_float32,
            # mlstm_layer
            qk_dim_factor=config.qk_dim_factor,
            v_dim_factor=config.v_dim_factor,
            # mlstm backend
            chunkwise_kernel=config.chunkwise_kernel,
            sequence_kernel=config.sequence_kernel,
            step_kernel=config.step_kernel,
            mode=config.mode,
            chunk_size=config.chunk_size,
            return_last_states=config.return_last_states,
            autocast_kernel_dtype=config.autocast_kernel_dtype,
            eps=config.eps,
            inference_state_dtype=config.inference_state_dtype,
            # feedforward
            ffn_proj_factor=config.ffn_proj_factor,
            ffn_round_up_to_multiple_of=config.ffn_round_up_to_multiple_of,
            # capping
            gate_soft_cap=config.gate_soft_cap,
            output_logit_soft_cap=config.output_logit_soft_cap,
            weight_mode=config.weight_mode,
        )

        xlstm_block_config.hidden_size = config.embedding_dim
        # xlstm_block_config.intermediate_size = round_up_to_next_multiple_of(
        #     xlstm_block_config.hidden_size * config.ffn_proj_factor,
        #     config.ffn_round_up_to_multiple_of,
        # )
        # # FOr large model
        xlstm_block_config.intermediate_size = config.intermediate_size
        xlstm_block_config.mlp_bias = config.use_bias
        xlstm_block_config.hidden_act = "silu"
        xlstm_block_config.rms_norm_eps = config.rms_norm_eps
        xlstm_block_config.num_key_value_heads = config.num_key_value_heads

        self.layers = nn.ModuleList(
            [mLSTMBlock(xlstm_block_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        # Not implemented yet - use pretrained model.
        pass

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embedding):
        self.embeddings = new_embedding

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[xLSTMCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, xLSTMOutput]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not self.training else False)
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = xLSTMCache(
                    self.config,
                    inputs_embeds.size(0),
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds

        if (
            not self.training
            and self.config.max_inference_chunksize < hidden_states.shape[1]
            and not output_hidden_states
        ):
            all_hidden_states = None
            offset = 0
            with torch.no_grad():
                if cache_params is None:
                    cache_params = xLSTMCache(
                        config=self.config, batch_size=hidden_states.shape[0]
                    )
                final_state = torch.zeros_like(hidden_states)
                while offset < hidden_states.shape[1]:
                    hidden_states_chunk = hidden_states[
                        :,
                        offset : min(
                            offset + self.config.max_inference_chunksize,
                            hidden_states.shape[1],
                        ),
                    ]
                for i, xlstm_block in enumerate(self.layers):
                    hidden_states_chunk, rnn_state = xlstm_block(
                        hidden_states_chunk,
                        state=cache_params.rnn_state[i],
                    )
                    for state_idx in range(len(cache_params.rnn_state[i])):
                        local_rnn_state = rnn_state[state_idx]
                        local_rnn_state = rnn_state[state_idx]
                        cache_params.rnn_state[i][state_idx].copy_(local_rnn_state)
                    cache_params.rnn_state_initial = False
                final_state[
                    :,
                    offset : min(
                        offset + self.config.max_inference_chunksize,
                        hidden_states.shape[1],
                    ),
                ] = hidden_states_chunk
                offset += self.config.max_inference_chunksize
            hidden_states = final_state
        else:
            all_hidden_states = () if output_hidden_states else None
            for i, xlstm_block in enumerate(self.layers):
                if self.gradient_checkpointing and self.training:
                    hidden_states, rnn_state = self._gradient_checkpointing_func(
                        xlstm_block.__call__,
                        hidden_states,
                        cache_params.rnn_state[i] if cache_params is not None else None,
                    )
                else:
                    hidden_states, rnn_state = xlstm_block(
                        hidden_states,
                        state=cache_params.rnn_state[i]
                        if cache_params is not None
                        else None,
                    )
                if cache_params:
                    for state_idx in range(len(cache_params.rnn_state[i])):
                        local_rnn_state = rnn_state[state_idx]
                        cache_params.rnn_state[i][state_idx].copy_(local_rnn_state)
                    cache_params.rnn_state_initial = False

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, cache_params, all_hidden_states]
                if v is not None
            )

        return xLSTMOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )

    def get_attn_map(self, input):
        attns = []
        embeddings = self.embeddings(input)
        for i, xlstm_block in enumerate(self.layers):
            _, attn = xlstm_block(embeddings, output_attentions=True, probbe=True)
            attns.append(attn)
        # _, attn =  self.layers[0](embeddings, output_attentions=True)
        return attns


@dataclass
class xLSTMCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[xLSTMCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class xLlamaForCausalLM(xLSTMPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = xLSTMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_attn_map(self, input):
        return self.model.get_attn_map(input)

    def freeze_it(self):
        # Freeze everything except the token mixer
        for param in self.parameters():
            param.requires_grad = False

        for layer in self.model.layers:
            for param in layer.parameters():
                for param in layer.token_mixer.parameters():
                    param.requires_grad = True

    def copy_from_teacher(self, teacher, copy_qkv: bool = True):
        assert len(self.model.layers) == len(teacher.model.layers)

        self.model.embeddings.weight.data.copy_(
            teacher.get_input_embeddings().weight.data
        )
        self.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
        self.lm_head.weight.data.copy_(teacher.get_output_embeddings().weight.data)

        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.layers):
            # self_layer.token_mixer.load_state_dict(teacher_layer.token_mixer.state_dict())
            self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict())
            self_layer.input_layernorm.load_state_dict(
                teacher_layer.input_layernorm.state_dict()
            )
            self_layer.post_attention_layernorm.load_state_dict(
                teacher_layer.post_attention_layernorm.state_dict()
            )

            if copy_qkv:
                self_layer.token_mixer.q.load_state_dict(
                    teacher_layer.self_attn.q_proj.state_dict()
                )
                self_layer.token_mixer.out_proj.load_state_dict(
                    teacher_layer.self_attn.o_proj.state_dict()
                )

                v_proj_unrolled = teacher_layer.self_attn.v_proj
                k_proj_unrolled = teacher_layer.self_attn.k_proj

                self_layer.token_mixer.v.load_state_dict(v_proj_unrolled.state_dict())
                self_layer.token_mixer.k.load_state_dict(k_proj_unrolled.state_dict())

                self_layer.token_mixer.igate_preact.bias.data.fill_(
                    torch.log(torch.tensor(2.0))
                )
                self_layer.token_mixer.igate_preact.bias.data.fill_(
                    -torch.log(torch.tensor(2.0))
                )

                # Init weight with small values
                init.xavier_uniform_(self_layer.token_mixer.igate_preact.weight.data)
                self_layer.token_mixer.igate_preact.weight.data *= 0.1
                init.xavier_uniform_(self_layer.token_mixer.fgate_preact.weight.data)
                self_layer.token_mixer.fgate_preact.weight.data *= 0.1

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
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return xLSTMCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params
            if outputs.cache_params is not None
            else None,
            hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[xLSTMCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Overwritten -- uses `cache_params` as opposed to `past_key_values`
        # Does not support using additional convolution states via inputs_embeds
        # as opposed to Mamba, currently.
        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            # If the first cache position is non-zero, we assume we are in generation mode.
            # Thus, the cache_params state is assumed to be the state before the last token
            # (lastly generated token), and all previous tokens are already ingested.
            # This should as well support generation from scratch with the [BOS] token inserted first.

            # if is_torchdynamo_compiling() or cache_position[0] > 0:
            if cache_params is not None:
                input_ids = input_ids[:, -1:]
                if inputs_embeds is not None:
                    inputs_embeds = inputs_embeds[:, -1:]

        attention_mask = None

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
        )
        return model_inputs


from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)

AutoConfig.register(xLlamaConfig.model_type, xLlamaConfig)
AutoModel.register(xLlamaConfig, xLSTMModel)
AutoModelForCausalLM.register(xLlamaConfig, xLlamaForCausalLM)
