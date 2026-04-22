from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2 import Qwen2Config, Qwen2PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm

from fla.modules import ShortConvolution
from fla.modules.feature_map import DPFPFeatureMap, HadamardFeatureMap, HedgehogFeatureMap, T2RFeatureMap
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn

from .utils import unroll_value_projection

logger = logging.get_logger(__name__)


class Qwen2LAConfig(Qwen2Config):
    model_type = "qwen2_la"

    def __init__(
        self,
        attn_mode: str = "chunk",
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        feature_map: str = "elementwise_product",
        tie_feature_map_qk: bool = False,
        norm_q: bool = False,
        norm_k: bool = False,
        do_feature_map_norm: bool = False,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        **qwen2_kwargs,
    ):
        self.attn_mode = attn_mode
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.feature_map = feature_map
        self.tie_feature_map_qk = tie_feature_map_qk
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.do_feature_map_norm = do_feature_map_norm
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        super().__init__(**qwen2_kwargs)


class EluActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.elu(x) + 1


class LinearAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        feature_map: str = 'elementwise_product',
        tie_feature_map_qk: bool = False,
        output_norm: str = 'rmsnorm',
        norm_q: bool = False,
        norm_k: bool = False,
        do_feature_map_norm: bool = False,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.mode = mode
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.do_feature_map_norm = do_feature_map_norm

        self.norm = None

        if feature_map == 'hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_k_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_k_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_k_dim)
        elif feature_map == 't2r':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_k_dim)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_k_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_k_dim)
        elif feature_map == 'elementwise_product':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(head_dim=self.head_k_dim)
            else:
                self.feature_map_q = HadamardFeatureMap(head_dim=self.head_k_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.head_k_dim)
        elif feature_map == 'dpfp':
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_k_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_k_dim)
        elif feature_map == 'elu':
            self.feature_map_q = EluActivation()
            self.feature_map_k = EluActivation()
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'identity':
            self.feature_map_q = nn.Identity()
            self.feature_map_k = nn.Identity()
        else:
            raise NotImplementedError(f"Not supported feature map `{feature_map}`.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')

        self.norm_q = norm_q
        self.norm_k = norm_k

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mode = self.mode
        if self.use_short_conv:
            q, _ = self.q_conv1d(x=self.q_proj(hidden_states))
            k, _ = self.k_conv1d(x=self.k_proj(hidden_states))
            v, _ = self.v_conv1d(x=self.v_proj(hidden_states))
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(k, '... (h d) -> ... h d', h=self.num_heads)
        v = rearrange(v, '... (h d) -> ... h d', h=self.num_heads)

        q = self.feature_map_q(q)
        k = self.feature_map_k(k)

        if self.norm_q:
            q = q / (q.sum(-1, True) + 1e-4)
        if self.norm_k:
            k = k / (k.sum(-1, True) + 1e-4)

        scale = self.head_k_dim ** -0.5

        if mode == 'chunk':
            o, final_state = chunk_linear_attn(
                q=q, k=k, v=v,
                normalize=self.do_feature_map_norm,
                scale=scale,
            )
        elif mode == 'fused_chunk':
            o, final_state = fused_chunk_linear_attn(
                q=q, k=k, v=v,
                normalize=self.do_feature_map_norm,
                scale=scale,
            )
        elif mode == 'fused_recurrent':
            o, final_state = fused_recurrent_linear_attn(
                q=q, k=k, v=v,
                normalize=self.do_feature_map_norm,
                scale=scale,
            )
        else:
            raise NotImplementedError

        o = rearrange(o, '... h d -> ... (h d)')
        o = self.o_proj(o)

        return o, None


class Qwen2LADecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LinearAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            feature_map=config.feature_map,
            tie_feature_map_qk=config.tie_feature_map_qk,
            norm_q=config.norm_q,
            norm_k=config.norm_k,
            do_feature_map_norm=config.do_feature_map_norm,
            elementwise_affine=config.elementwise_affine,
            norm_eps=config.norm_eps,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            conv_bias=config.conv_bias,
            layer_idx=layer_idx
        )
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attentions = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )

        if output_attentions:
            return hidden_states, attentions

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


class Qwen2LAPreTrainedModel(Qwen2PreTrainedModel):
    config_class = Qwen2LAConfig
    supports_gradient_checkpointing = True
    base_model_prefix = "model"

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


class Qwen2LAModel(Qwen2LAPreTrainedModel):
    config_class = Qwen2LAConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2LADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    **kwargs
                )
            else:
                hidden_states, attn = layer(
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


class Qwen2LAForCausalLM(Qwen2LAPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen2LAConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2LAModel(config)
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

    def freeze_it(self):
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.model.layers:
            for param in layer.self_attn.parameters():
                param.requires_grad = True

    def copy_from_teacher(self, teacher, copy_qkv=True):
        assert len(self.model.layers) == len(teacher.model.layers)

        self.model.embed_tokens.weight.data.copy_(teacher.get_input_embeddings().weight.data)
        self.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
        self.lm_head.weight.data.copy_(teacher.get_output_embeddings().weight.data)

        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.layers):
            if copy_qkv:
                self_layer.self_attn.q_proj.load_state_dict(teacher_layer.self_attn.q_proj.state_dict())
                n_rep = self_layer.self_attn.v_proj.weight.shape[0] // teacher_layer.self_attn.v_proj.weight.shape[0]
                v_proj_unrolled = unroll_value_projection(teacher_layer.self_attn.v_proj, n_rep)
                k_proj_unrolled = unroll_value_projection(teacher_layer.self_attn.k_proj, n_rep)
                self_layer.self_attn.k_proj.load_state_dict(k_proj_unrolled.state_dict())
                self_layer.self_attn.v_proj.load_state_dict(v_proj_unrolled.state_dict())
                self_layer.self_attn.o_proj.load_state_dict(teacher_layer.self_attn.o_proj.state_dict())

            self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict())
            self_layer.input_layernorm.load_state_dict(teacher_layer.input_layernorm.state_dict())
            self_layer.post_attention_layernorm.load_state_dict(teacher_layer.post_attention_layernorm.state_dict())

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

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


from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
AutoConfig.register(Qwen2LAConfig.model_type, Qwen2LAConfig)
AutoModel.register(Qwen2LAConfig, Qwen2LAModel)
AutoModelForCausalLM.register(Qwen2LAConfig, Qwen2LAForCausalLM)

StudentConfig = Qwen2LAConfig
StudentModel = Qwen2LAForCausalLM
