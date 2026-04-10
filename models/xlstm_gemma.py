from typing import Optional

from transformers.modeling_layers import GenericForSequenceClassification, GradientCheckpointingLayer

import torch
from torch import nn
from torch.nn import init

from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3RMSNorm, Gemma3Attention, Gemma3MLP, Gemma3PreTrainedModel, Gemma3TextScaledWordEmbedding,
    Gemma3RotaryEmbedding
)
from transformers.masking_utils import create_causal_mask, create_masks_for_generate, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from models.modular_xlstm import mLSTMLayerConfig, mLSTMLayer, xGemma3Config

class Gemma3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: xGemma3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        # self.attention_type = config.layer_types[layer_idx]
        self.self_attn = mLSTMLayer(mLSTMLayerConfig.from_xgemma3_config(config))
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        mlstm_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, state, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            state=mlstm_state,
            output_attentions=output_attentions,
            **kwargs,
        )

        if self_attn_weights is not None:
            return None, self_attn_weights

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        # if output_attentions:
        #     outputs += (self_attn_weights,)

        return outputs, None


class xGemma3TextModel(Gemma3PreTrainedModel):
    config: xGemma3Config
    input_modalities = "text"

    def __init__(self, config: xGemma3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values  = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # if use_cache and past_key_values is None and not self.training:
        #     past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            if self.config.use_bidirectional_attention:
                mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
                sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(self.config.sliding_window)

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        # for layer_type in self.config.layer_types:
        #     position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                # attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                # position_embeddings=position_embeddings[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class xGemma3ForCausalLM(Gemma3PreTrainedModel):
    config: xGemma3Config

    def __init__(self, config: xGemma3Config):
        super().__init__(config)
        self.model = xGemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep = 0,
        **kwargs,
    ):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

        >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def copy_from_teacher(self, teacher: "OPTForCausalLM", copy_qkv: bool = True):
        # Embeddings + LM head
        self.get_input_embeddings().weight.data.copy_(
            teacher.get_input_embeddings().weight.data
        )
        # if self.model.decoder.final_layer_norm is not None and teacher.model.decoder.final_layer_norm is not None:
        #     self.model.decoder.final_layer_norm.load_state_dict(
        #         teacher.model.decoder.final_layer_norm.state_dict()
        #     )
        self.lm_head.weight.data.copy_(teacher.lm_head.weight.data)

        # Per-layer copy
        assert len(self.model.layers) == len(teacher.model.layers)
        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.layers):
            # MLP + norms
            self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict())

            self_layer.input_layernorm.load_state_dict(
                teacher_layer.input_layernorm.state_dict()
            )
            self_layer.post_attention_layernorm.load_state_dict(
                teacher_layer.post_attention_layernorm.state_dict()
            )
            self_layer.pre_feedforward_layernorm.load_state_dict(
                teacher_layer.pre_feedforward_layernorm.state_dict()
            )
            self_layer.post_attention_layernorm.load_state_dict(
                teacher_layer.post_attention_layernorm.state_dict()
            )

            if copy_qkv:
                # Map teacher OPTAttention projections -> mLSTM projections
                self_layer.self_attn.q.load_state_dict(
                    teacher_layer.self_attn.q_proj.state_dict()
                )
                self_layer.self_attn.out_proj.load_state_dict(
                    teacher_layer.self_attn.o_proj.state_dict()
                )

                self_layer.self_attn.v.load_state_dict(
                    teacher_layer.self_attn.v_proj.state_dict()
                )
                self_layer.self_attn.k.load_state_dict(
                    teacher_layer.self_attn.k_proj.state_dict()
                )

                # Gate init as in xQwen3
                self_layer.self_attn.igate_preact.bias.data.fill_(
                    torch.log(torch.tensor(2.0))
                )
                self_layer.self_attn.fgate_preact.bias.data.fill_(
                    -torch.log(torch.tensor(2.0))
                )

                init.xavier_uniform_(self_layer.self_attn.igate_preact.weight.data)
                self_layer.self_attn.igate_preact.weight.data *= 0.1
                init.xavier_uniform_(self_layer.self_attn.fgate_preact.weight.data)
                self_layer.self_attn.fgate_preact.weight.data *= 0.1


from transformers import AutoModelForCausalLM, AutoConfig, AutoModel

AutoConfig.register("xgemma3", xGemma3Config)
AutoModel.register(xGemma3Config, xGemma3TextModel)
AutoModelForCausalLM.register(xGemma3Config, xGemma3ForCausalLM)
