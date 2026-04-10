from collections.abc import Callable
from typing import Optional, Union
from regex import A

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.init as init  # <-- NEW

from models.modular_xlstm import mLSTMLayerConfig, mLSTMLayer  # <-- NEW

from transformers.models.opt.modeling_opt import (
    OPTPreTrainedModel,
    OPTLearnedPositionalEmbedding,
)

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack

from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn

from .modular_xlstm import LAOPTConfig



class EluActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.elu(x) + 1

class LAAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LAOPTConfig,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.enable_bias = config.enable_bias
        self.layer_idx = layer_idx

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.feature_map_q = EluActivation()
        self.feature_map_k = EluActivation()

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        # attention_interface: Callable = eager_attention_forward
        #
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # attn_output, attn_weights = attention_interface(
        #     self,
        #     query_states,
        #     key_states,
        #     value_states,
        #     attention_mask,
        #     dropout=0.0 if not self.training else self.dropout,
        #     scaling=1.0,
        #     **kwargs,
        # )

        q = self.feature_map_q(query_states)
        k = self.feature_map_k(key_states)

        q = q / (q.sum(-1, True) + 1e-4)
        k = k / (k.sum(-1, True) + 1e-4)

        batch, heads, seq, dim = q.shape

        # Permute for easier matmul
        # q = q.permute(0, 2, 1, 3)  # (batch, heads, seq, dim)
        # k = k.permute(0, 2, 1, 3)  # (batch, heads, seq, dim)
        # q = q.permute(0, 2, 1, 3)  # (batch, heads, seq, dim)
        # k = k.permute(0, 2, 1, 3)  # (batch, heads, seq, dim)

        # Compute similarity
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * (dim**-0.5)  # (batch, heads, seq, seq)

        if output_attentions and attention_mask is None:
            # Causal masking
            causal_mask = torch.tril(torch.ones(seq, seq, device=q.device)).unsqueeze(0).unsqueeze(0)

            # Apply causal mask
            attn_weights = attn_weights * causal_mask  # (batch, heads, seq, seq)

        if attention_mask is not None:
            # Apply attention mask
            attn_weights = attn_weights * attention_mask

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)  # (batch, heads, seq, dim)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq, heads, dim)

        if output_attentions:
            return None, attn_weights

        # mode = "chunk"

        # if mode == 'chunk':
        #     try:
        #         attn_output, final_state = chunk_linear_attn(
        #             q=q,
        #             k=k,
        #             v=value_states,
        #             normalize=True
        #         )
        #     except:
        #         attn_output, final_state = fused_recurrent_linear_attn(
        #             q=q,
        #             k=k,
        #             v=value_states,
        #             normalize=True
        #         )
        # elif mode == 'fused_chunk':
        #     attn_output, final_state = fused_chunk_linear_attn(
        #         q=q,
        #         k=k,
        #         v=value_states,
        #         normalize=True
        #     )
        # elif mode == 'fused_recurrent':
        #     attn_output, final_state = fused_recurrent_linear_attn(
        #         q=q,
        #         k=k,
        #         v=value_states,
        #         normalize=True
        #     )
        # else:
        #     raise NotImplementedError

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

class OPTDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LAOPTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.hidden_size

        # --- mLSTM token mixer instead of attention ---
        # Assumes mLSTMLayerConfig has a from_opt_config() constructor analogous to from_xqwen3_config().
        self.self_attn = LAAttention(config=config, layer_idx=layer_idx)


        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        mlstm_state: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        mLSTM version of the decoder layer. We keep the signature compatible with OPT but ignore
        attention-related arguments internally, just like in the xQwen3 + mLSTM integration.
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE the token mixer
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # --- mLSTM token mixer ---
        hidden_states, attn = self.self_attn(
            hidden_states=hidden_states,
            state=mlstm_state,
            output_attentions=output_attentions,
            **kwargs,
        )

        # Special "record-only" mode (mirrors xQwen3DecoderLayer):
        # if the mLSTM layer returns an attention diagnostic, we short-circuit.
        if attn is not None:
            return None, attn

        # Dropout + residual
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER the token mixer
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # --- Feed-forward (unchanged) ---
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        # outputs = (hidden_states,)

        # # mLSTM can optionally return some "attn-like" diagnostics; we surface it if requested
        # if output_attentions:
        #     outputs += (attn,)

        return hidden_states, None

class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    config: LAOPTConfig
    def __init__(self, config: LAOPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if input_ids is not None:
            input_ids = input_ids.view(-1, input_ids.shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if attention_mask is None:
            seq_length = past_seen_tokens + inputs_embeds.shape[1]
            attention_mask = torch.ones(inputs_embeds.shape[0], seq_length, device=inputs_embeds.device)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        if position_ids is None:
            # position_ids = cache_position.unsqueeze(0)
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            # cut positions if `past_seen_tokens` is > 0
            position_ids = position_ids[:, past_seen_tokens:]

        pos_embeds = self.embed_positions(attention_mask, past_seen_tokens, position_ids=position_ids)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds.to(inputs_embeds.device)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
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

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LAOPTForCausalLM(OPTPreTrainedModel, GenerationMixin):
    config: LAOPTConfig
    _tied_weights_keys = {"lm_head.weight": "model.decoder.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTDecoder(config)
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ) -> Union[tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).contiguous()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

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
        assert len(self.model.layers) == len(teacher.model.decoder.layers)
        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.decoder.layers):
            # MLP + norms
            self_layer.fc1.load_state_dict(teacher_layer.fc1.state_dict())
            self_layer.fc2.load_state_dict(teacher_layer.fc2.state_dict())
            self_layer.self_attn_layer_norm.load_state_dict(
                teacher_layer.self_attn_layer_norm.state_dict()
            )
            self_layer.final_layer_norm.load_state_dict(
                teacher_layer.final_layer_norm.state_dict()
            )

            if copy_qkv:
                # Map teacher OPTAttention projections -> mLSTM projections
                self_layer.self_attn.q_proj.load_state_dict(
                    teacher_layer.self_attn.q_proj.state_dict()
                )
                self_layer.self_attn.out_proj.load_state_dict(
                    teacher_layer.self_attn.out_proj.state_dict()
                )

                self_layer.self_attn.v_proj.load_state_dict(
                    teacher_layer.self_attn.v_proj.state_dict()
                )
                self_layer.self_attn.k_proj.load_state_dict(
                    teacher_layer.self_attn.k_proj.state_dict()
                )

                # Gate init as in xQwen3
                # self_layer.self_attn.igate_preact.bias.data.fill_(
                #     torch.log(torch.tensor(2.0))
                # )
                # self_layer.self_attn.fgate_preact.bias.data.fill_(
                #     -torch.log(torch.tensor(2.0))
                # )

                # init.xavier_uniform_(self_layer.self_attn.igate_preact.weight.data)
                # self_layer.self_attn.igate_preact.weight.data *= 0.1
                # init.xavier_uniform_(self_layer.self_attn.fgate_preact.weight.data)
                # self_layer.self_attn.fgate_preact.weight.data *= 0.1


from transformers import AutoModelForCausalLM, AutoConfig, AutoModel

AutoConfig.register("laopt", LAOPTConfig)
AutoModel.register(LAOPTConfig, OPTDecoder)
AutoModelForCausalLM.register(LAOPTConfig, LAOPTForCausalLM)

