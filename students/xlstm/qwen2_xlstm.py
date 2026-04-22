from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.init as init

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging
from transformers.models.qwen2 import Qwen2PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm

from models.modular_xlstm import mLSTMLayerConfig, mLSTMLayer, xQwen2Config, xLSTMCache

logger = logging.get_logger(__name__)

mLSTMLayerStateType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
mLSTMStateType = dict[int, mLSTMLayerStateType]


@dataclass
class xQwen2LSTMOutput(CausalLMOutputWithPast):
    cache_params: Optional[xLSTMCache] = None
    last_hidden_state: Optional[torch.Tensor] = None


class Qwen2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: xQwen2Config, device=None):
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
        config: Optional[xQwen2Config] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2XLSTMDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: xQwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = mLSTMLayer(mLSTMLayerConfig.from_xqwen2_config(config))
        self.self_attn.layer_idx = layer_idx

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, state


class xQwen2PreTrainedModel(Qwen2PreTrainedModel):
    config_class = xQwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2XLSTMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

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


class xQwen2Model(xQwen2PreTrainedModel):
    def __init__(self, config: xQwen2Config):
        super().__init__(config)
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2XLSTMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

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
                        :, offset: min(offset + max_inference_chunksize, hidden_states.shape[1])
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
                    final_state[:, offset: min(offset + max_inference_chunksize, hidden_states.shape[1])] = hidden_states_chunk
                    offset += max_inference_chunksize
                hidden_states = final_state
        else:
            for decoder_layer in self.layers[:self.config.num_hidden_layers]:
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
        return xQwen2LSTMOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            cache_params=cache_params
        )


class xQwen2ForCausalLM(xQwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = xQwen2Model(config)
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
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return xQwen2LSTMOutput(
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
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[xLSTMCache] = None,
        **kwargs,
    ):
        if use_cache and cache_params is not None:
            input_ids = input_ids[:, -1:]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:]

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({"cache_params": cache_params, "use_cache": use_cache})

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
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, **kwargs
        )
        if hasattr(outputs, "cache_params") and outputs.cache_params is not None:
            model_kwargs["cache_params"] = outputs.cache_params
        return model_kwargs

    def copy_from_teacher(self, teacher, copy_qkv: bool = True):
        assert len(self.model.layers) == len(teacher.model.layers)

        self.model.embed_tokens.weight.data.copy_(teacher.get_input_embeddings().weight.data)
        self.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
        self.lm_head.weight.data.copy_(teacher.get_output_embeddings().weight.data)

        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.layers):
            self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict())
            self_layer.input_layernorm.load_state_dict(teacher_layer.input_layernorm.state_dict())
            self_layer.post_attention_layernorm.load_state_dict(teacher_layer.post_attention_layernorm.state_dict())

            if self_layer.self_attn.config.num_heads != teacher_layer.self_attn.config.num_attention_heads:
                print("Different number of attention heads; skipping QKV copy.")
                copy_qkv = False

            if copy_qkv:
                self_layer.self_attn.q.load_state_dict(teacher_layer.self_attn.q_proj.state_dict())
                self_layer.self_attn.out_proj.load_state_dict(teacher_layer.self_attn.o_proj.state_dict())
                self_layer.self_attn.v.load_state_dict(teacher_layer.self_attn.v_proj.state_dict())
                self_layer.self_attn.k.load_state_dict(teacher_layer.self_attn.k_proj.state_dict())

                self_layer.self_attn.igate_preact.bias.data.fill_(torch.log(torch.tensor(2.0)))
                self_layer.self_attn.fgate_preact.bias.data.fill_(-torch.log(torch.tensor(2.0)))

            init.xavier_uniform_(self_layer.self_attn.igate_preact.weight.data)
            self_layer.self_attn.igate_preact.weight.data *= 0.1
            init.xavier_uniform_(self_layer.self_attn.fgate_preact.weight.data)
            self_layer.self_attn.fgate_preact.weight.data *= 0.1


__all__ = [
    "xQwen2ForCausalLM",
    "xQwen2Model",
    "xQwen2PreTrainedModel",
]

from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
AutoConfig.register("xqwen2", xQwen2Config)
AutoModel.register(xQwen2Config, xQwen2Model)
AutoModelForCausalLM.register(xQwen2Config, xQwen2ForCausalLM)

StudentConfig = xQwen2Config
StudentModel = xQwen2ForCausalLM
