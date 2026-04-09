from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.init as init

from transformers import GenerationConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateOutput
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from models.modular_xlstm import mLSTMLayerConfig, mLSTMLayer, xLlamaConfig, xLSTMCache

logger = logging.get_logger(__name__)

mLSTMLayerStateType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
mLSTMStateType = dict[int, mLSTMLayerStateType]

@dataclass
class xLSTMOutput(CausalLMOutputWithPast):
    cache_params: Optional[xLSTMCache] = None
    last_hidden_state: Optional[torch.Tensor] = None


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: xLlamaConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[xLlamaConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: xLlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: xLlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.self_attn = mLSTMLayer(mLSTMLayerConfig.from_xllama_config(config))
        self.self_attn.layer_idx = layer_idx

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        state: Optional[DynamicCache] = None,
        output_attentions: Optional[bool] = False,
        diagnostics=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, state, attn = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            state=state,
            diagnostics=diagnostics,
            **kwargs,
        )
        if diagnostics is not None:
            diagnostics.record_output(self.self_attn.layer_idx, hidden_states)
        
        if output_attentions:
            return hidden_states, attn

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, state


class xLlamaPreTrainedModel(PreTrainedModel):
    config: xLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


class xLlamaModel(xLlamaPreTrainedModel):
    def __init__(self, config: xLlamaConfig):
        super().__init__(config)
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_params: Optional[xLSTMCache] = None,
        diagnostics=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and cache_params is None:
            cache_params = xLSTMCache(
                self.config,
                inputs_embeds.shape[0],
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        max_inference_chunksize = 16384
        if not self.training:
            offset = 0
            with torch.no_grad():
                if cache_params is None:
                    cache_params = xLSTMCache(
                        self.config,
                        hidden_states.shape[0],
                    )
                final_state = torch.zeros_like(hidden_states)

                while offset < hidden_states.shape[1]:
                    hidden_states_chunk = hidden_states[
                        :, offset : min(offset + max_inference_chunksize, hidden_states.shape[1])
                    ]
                    for layer_idx, layer in enumerate(self.layers):
                        hidden_states_chunk, rnn_state = layer(
                            hidden_states_chunk,
                            position_embeddings=position_embeddings,
                            use_cache=use_cache,
                            state=cache_params.rnn_state[layer_idx],
                            diagnostics=diagnostics,
                        )
                        for state_idx in range(len(cache_params.rnn_state[layer_idx])):
                            local_rnn_state = rnn_state[state_idx]
                            cache_params.rnn_state[layer_idx][state_idx].copy_(local_rnn_state)

                        cache_params.rnn_state_initial = False
                    final_state[:, offset : min(offset + max_inference_chunksize, hidden_states.shape[1])] = hidden_states_chunk
                    offset += max_inference_chunksize
                hidden_states = final_state
        else:
            for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                hidden_states, _ = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    diagnostics=diagnostics,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        return xLSTMOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            cache_params=cache_params
        )


class xLlamaForCausalLM(xLlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = xLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        cache_params: Optional[xLSTMCache] = None,
        diagnostics=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
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
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            cache_params=cache_params,
            diagnostics=diagnostics,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return xLSTMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cache_params=outputs.cache_params,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,  # not used but needed, otherwise generate complains when passing tokenizer inputs
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[xLSTMCache] = None,
        **kwargs,
    ):
        if use_cache and cache_params is not None:
            # If the first cache position is non-zero, we assume we are in generation mode.
            # Thus, the cache_params state is assumed to be the state before the last token
            # (lastly generated token), and all previous tokens are already ingested.
            # This should as well support generation from scratch with the [BOS] token inserted first.
            input_ids = input_ids[:, -1:]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:]

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({"cache_params": cache_params, "use_cache": use_cache})

        # Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value
        
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        **kwargs,
    ):
        # First, let HF do its normal thing (past_key_values, attention_mask, cache_position, etc.)
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, **kwargs
        )

        # Then, propagate our custom cache object
        if hasattr(outputs, "cache_params") and outputs.cache_params is not None:
            model_kwargs["cache_params"] = outputs.cache_params

        return model_kwargs


    def copy_from_teacher(self, teacher, copy_qkv: bool = True):
        assert len(self.model.layers) == len(teacher.model.layers)

        self.model.embed_tokens.weight.data.copy_(
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

            if self_layer.self_attn.config.num_heads != teacher_layer.self_attn.config.num_attention_heads:
                # Different Head Configurations
                print("Different number of attention heads; skipping QKV copy.")
                copy_qkv = False

            if copy_qkv:
                self_layer.self_attn.q.load_state_dict(
                    teacher_layer.self_attn.q_proj.state_dict()
                )
                self_layer.self_attn.out_proj.load_state_dict(
                    teacher_layer.self_attn.o_proj.state_dict()
                )

                v_proj_unrolled = teacher_layer.self_attn.v_proj
                k_proj_unrolled = teacher_layer.self_attn.k_proj

                self_layer.self_attn.v.load_state_dict(v_proj_unrolled.state_dict())
                self_layer.self_attn.k.load_state_dict(k_proj_unrolled.state_dict())

                self_layer.self_attn.igate_preact.bias.data.fill_(
                    torch.log(torch.tensor(2.0))
                )
                self_layer.self_attn.igate_preact.bias.data.fill_(
                    -torch.log(torch.tensor(2.0))
                )

            # Init weight with small values
            init.xavier_uniform_(self_layer.self_attn.igate_preact.weight.data)
            self_layer.self_attn.igate_preact.weight.data *= 0.1
            init.xavier_uniform_(self_layer.self_attn.fgate_preact.weight.data)
            self_layer.self_attn.fgate_preact.weight.data *= 0.1

    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs: Optional[torch.Tensor] = None,
    #     generation_config = None,
    #     logits_processor = None,
    #     stopping_criteria = None,
    #     prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
    #     synced_gpus: Optional[bool] = None,
    #     assistant_model: Optional["PreTrainedModel"] = None,
    #     streamer = None,
    #     negative_prompt_ids: Optional[torch.Tensor] = None,
    #     negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    #     use_model_defaults: Optional[bool] = None,
    #     custom_generate: Optional[Union[str, Callable]] = None,
    #     **kwargs,
    # ):
    #
    #     if generation_config is None:
    #         generation_config = getattr(self, "generation_config", GenerationConfig())
    #     generation_config = generation_config
    #
    #     total_new_tokens = kwargs.pop("max_new_tokens", None)
    #     if total_new_tokens is None:
    #         total_new_tokens = generation_config.max_new_tokens
    #     if total_new_tokens is None:
    #         total_new_tokens = 20  # sensible default
    #
    #     step_config = generation_config
    #     step_config.max_new_tokens = 1
    #     sequences = inputs
    #     collected_scores = [] if step_config.output_scores else None
    #     collected_attentions = None  # if you want to collect them, init lists here
    #     collected_hidden_states = None
    #
    #     for step in range(total_new_tokens):
    #         # NOTE: This super() call uses the ORIGINAL GenerationMixin.generate
    #         step_output = super().generate(
    #             inputs=sequences,
    #             generation_config=step_config,
    #             logits_processor=logits_processor,
    #             stopping_criteria=stopping_criteria,
    #             prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    #             synced_gpus=synced_gpus,
    #             streamer=streamer,
    #             # **kwargs,
    #         )
    #
    #         # HF generate can return either tensor or GenerateOutput
    #         if isinstance(step_output, GenerateOutput):
    #             step_seqs = step_output.sequences
    #         else:
    #             step_seqs = step_output
    #
    #         # step_seqs already contains the full prefix + 1 new token,
    #         # so we can simply replace `sequences` with it.
    #         sequences = step_seqs
    #
    #         # ---- (Optional) hook: inspect / modify between steps ----
    #         # Example: inspect the last generated token
    #         # last_token = sequences[:, -1]
    #         # do something with last_token / sequences here
    #
    #         # Early stop on EOS if desired
    #         eos_token_id = step_config.eos_token_id
    #         if eos_token_id is not None:
    #             eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    #             finished = torch.isin(sequences[:, -1], torch.tensor(eos_ids, device=sequences.device))
    #             if finished.all():
    #                 break
    #
    #     # ---- 4. Return result ----
    #     # if not return_dict_in_generate:
    #     #     # Standard behavior: just return the sequences
    #     #     return sequences
    #
    #     # If you care about scores/attn/hidden states, you’d need to collect them
    #     # across steps above. Here we just wrap the sequences.
    #     return sequences


__all__ = [
    "xLlamaForCausalLM",
    "xLlamaModel",
    "xLlamaPreTrainedModel",
]

from transformers import AutoConfig, AutoModelForCausalLM, AutoModel

AutoConfig.register("xllama", xLlamaConfig)
AutoModel.register(xLlamaConfig, xLlamaModel)
AutoModelForCausalLM.register(xLlamaConfig, xLlamaForCausalLM)

StudentConfig = xLlamaConfig
StudentModel = xLlamaForCausalLM
