# coding=utf-8
# Copyright 2024
"""Qwen3 + mLSTM hybrid model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.init as init

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.models.qwen3 import Qwen3Config, Qwen3PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from models.modular_xlstm import mLSTMLayer, mLSTMLayerConfig
from xlstm.xlstm_large.model import (
    BackendModeType,
    ChunkwiseKernelType,
    DtypeType,
    SequenceKernelType,
    StepKernelType,
    WeightModeType,
)


class Qwen3XLSTMConfig(Qwen3Config):
    model_type = "qwen3_xlstm"

    def __init__(
        self,
        chunkwise_kernel: ChunkwiseKernelType = "chunkwise--native_autograd",
        sequence_kernel: SequenceKernelType = "native_sequence__native",
        step_kernel: StepKernelType = "native",
        mode: BackendModeType = "inference",
        chunk_size: int = 64,
        return_last_states: bool = True,
        autocast_kernel_dtype: DtypeType = "bfloat16",
        eps: float = 1e-6,
        inference_state_dtype: DtypeType = "float32",
        ffn_proj_factor: float = 2.667,
        ffn_round_up_to_multiple_of: int = 64,
        gate_soft_cap: float = 15.0,
        output_logit_soft_cap: float = 30.0,
        weight_mode: WeightModeType = "single",
        apply_rope: bool = False,
        qk_dim_factor: float = 1.0,
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
        self.apply_rope = apply_rope
        self.qk_dim_factor = qk_dim_factor
        super().__init__(**qwen3_kwargs)


@dataclass
class Qwen3XLSTMOutputWithPast(BaseModelOutputWithPast):
    mlstm_states: Optional[tuple[torch.Tensor, ...]] = None


class Qwen3XLSTMDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3XLSTMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        mlstm_config = mLSTMLayerConfig.from_xqwen3_config(config)
        self.self_attn = mLSTMLayer(mlstm_config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        mlstm_state: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, mlstm_state, attn = self.self_attn(
            hidden_states=hidden_states,
            state=mlstm_state,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, mlstm_state, attn


class Qwen3XLSTMPreTrainedModel(Qwen3PreTrainedModel):
    config: Qwen3XLSTMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3XLSTMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]


class Qwen3XLSTMModel(Qwen3XLSTMPreTrainedModel):
    def __init__(self, config: Qwen3XLSTMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3XLSTMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Qwen3XLSTMOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        mlstm_states = () if use_cache else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, layer_state, attn = decoder_layer(
                hidden_states,
                output_attentions=bool(output_attentions),
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if use_cache:
                mlstm_states = mlstm_states + (layer_state,)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm(hidden_states)
        return Qwen3XLSTMOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            mlstm_states=mlstm_states,
        )


class Qwen3XLSTMForCausalLM(Qwen3XLSTMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Qwen3XLSTMConfig):
        super().__init__(config)
        self.model = Qwen3XLSTMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def copy_from_teacher(self, teacher, copy_qkv: bool = True) -> None:
        if len(self.model.layers) != len(teacher.model.layers):
            raise ValueError("Teacher and student must have the same number of layers")

        self.model.embed_tokens.weight.data.copy_(teacher.get_input_embeddings().weight.data)
        self.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
        self.lm_head.weight.data.copy_(teacher.get_output_embeddings().weight.data)

        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.layers):
            self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict())
            self_layer.input_layernorm.load_state_dict(teacher_layer.input_layernorm.state_dict())
            self_layer.post_attention_layernorm.load_state_dict(
                teacher_layer.post_attention_layernorm.state_dict()
            )

            if copy_qkv:
                self_layer.self_attn.q.load_state_dict(teacher_layer.self_attn.q_proj.state_dict())
                self_layer.self_attn.out_proj.load_state_dict(teacher_layer.self_attn.o_proj.state_dict())
                self_layer.self_attn.v.load_state_dict(teacher_layer.self_attn.v_proj.state_dict())
                self_layer.self_attn.k.load_state_dict(teacher_layer.self_attn.k_proj.state_dict())

                self_layer.self_attn.igate_preact.bias.data.fill_(1.0)
                self_layer.self_attn.fgate_preact.bias.data.fill_(-1.0)
                init.xavier_uniform_(self_layer.self_attn.igate_preact.weight.data)
                self_layer.self_attn.igate_preact.weight.data *= 0.1
                init.xavier_uniform_(self_layer.self_attn.fgate_preact.weight.data)
                self_layer.self_attn.fgate_preact.weight.data *= 0.1


# AutoConfig.register(Qwen3XLSTMConfig.model_type, Qwen3XLSTMConfig)
# AutoModel.register(Qwen3XLSTMConfig, Qwen3XLSTMModel)
# AutoModelForCausalLM.register(Qwen3XLSTMConfig, Qwen3XLSTMForCausalLM)
