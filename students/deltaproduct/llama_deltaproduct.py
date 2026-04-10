from typing import Optional, Tuple, Union, Dict, List
import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin

from einops import rearrange, repeat
from transformers.utils import logging
import transformers.models.llama.modeling_llama as llama
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from fla.modules import RMSNorm, ShortConvolution
from fla.models.utils import Cache
# from fla.layers.gla import GatedLinearAttention as FlaGatedLinearAttention
from fla.layers.gated_deltaproduct import GatedDeltaProduct as FlaGatedDeltaProduct
from fla.models.gated_deltaproduct import GatedDeltaProductConfig
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gated_delta_product import chunk_gated_delta_product
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule


from .utils import unroll_value_projection

logger = logging.get_logger(__name__)

class LlamaGatedDeltaProductConfig(GatedDeltaProductConfig):
    model_type = "llama_gdp"

    def __init__(self, rms_norm_eps: float = 1e-6, mlp_bias: bool = False,
                 intermediate_size: int = 1024, expand_k: float = 1.0,
                 expand_v: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.rms_norm_eps = rms_norm_eps
        self.mlp_bias = mlp_bias
        self.intermediate_size = intermediate_size
        self.expand_k = expand_k
        self.expand_v = expand_v

class GatedDeltaProduct(nn.Module):
    """
    Generalized version of GatedDoubleDeltaNet that supports arbitrary number of householder transformations.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        mode: str = 'chunk',
        use_output_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        use_forget_gate: bool = True,
        allow_neg_eigval: bool = True,
        num_householder: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.mode = mode

        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_forget_gate = use_forget_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.num_householder = num_householder
        self.use_output_gate = use_output_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_v_heads = num_heads

        self.head_k_dim = self.head_dim
        self.head_v_dim = self.head_dim
        self.key_dim = hidden_size
        self.value_dim = hidden_size
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        # if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
        #     raise ValueError(
        #         f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
        #         f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
        #     )
        # if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
        #     raise ValueError(
        #         f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
        #     )
        #
        # if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
        #     raise ValueError(
        #         f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
        #         f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
        #     )
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        # self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        # self.k_proj = nn.Linear(hidden_size, self.key_dim * num_householder, bias=False)
        # self.v_proj = nn.Linear(hidden_size, self.value_dim * num_householder, bias=False)
        # self.b_proj = nn.Linear(hidden_size, self.num_v_heads * num_householder, bias=False)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size * num_householder, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size * num_householder, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads * num_householder, bias=False)

        if self.use_forget_gate:
            self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
            A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True
            # hard coded for now
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min),
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim * num_householder,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim * num_householder,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing.",
            )
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs 
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... t (n h d) -> ... (t n) h d', n=self.num_householder, d=self.head_k_dim)
        v = rearrange(v, '... t (n h d) -> ... (t n) h d', n=self.num_householder, d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.

        beta = rearrange(beta, '... t (n h) -> ... (t n) h', n=self.num_householder)
        if self.use_forget_gate:
            g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)
        else:
            g = None

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_product(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                num_householder=self.num_householder,
                use_qk_l2norm_in_kernel=True,
            )

        elif mode == 'fused_recurrent':
            if self.use_forget_gate:
                g_new = g.new_zeros(g.shape[0], g.shape[1], self.num_householder, g.shape[2])
                g_new[:, :, 0] = g
                g = rearrange(g_new, '... t n h -> ... (t n) h')

            q_new = q.new_zeros(q.shape[0], q.shape[1], self.num_householder, q.shape[2], q.shape[3])
            q_new[:, :, -1] = q
            q = rearrange(q_new, '... t n h d-> ... (t n) h d')
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens * self.num_householder if cu_seqlens is not None else None,
                use_qk_l2norm_in_kernel=True,
            )
            o = rearrange(o, '... (t n) h d -> ... t n h d', n=self.num_householder)[..., -1, :, :].contiguous()

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_output_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values

class GatedDeltaProductDecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GatedDeltaProduct(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_attention_heads,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                use_output_gate=config.use_output_gate,
                gate_fn=config.hidden_act,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx
            )
        self.mlp = llama.LlamaMLP(config)
        self.input_layernorm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        past_key_values = None,
        output_attention_hidden_states: bool = False,
        output_token_mixer: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if output_token_mixer:
            output_attentions = False

        hidden_states, attentions, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if output_token_mixer:
            return hidden_states

        if output_attentions:
            return hidden_states, attentions

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, None)
        # if output_attentions:
        #     outputs += (self_attn_weights,)

        return outputs

class GatedDeltaProductModel(llama.LlamaPreTrainedModel):
    config_class = LlamaGatedDeltaProductConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GatedDeltaProductDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
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

class LlamaGatedDeltaProductPreTrainedModel(PreTrainedModel):
    config_class = LlamaGatedDeltaProductConfig
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



class GatedDeltaProductForCausalLM(LlamaGatedDeltaProductPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    config_class = LlamaGatedDeltaProductConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = GatedDeltaProductModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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

    def freeze_it(self):
        # Freeze everything except the token mixer
        for param in self.parameters():
            param.requires_grad = False

        for layer in self.model.layers:
            for param in layer.token_mixer.parameters():
                param.requires_grad = True

    def copy_from_teacher(self, teacher, copy_qkv=True):
        assert len(self.model.layers) == len(teacher.model.layers)

        self.model.embeddings.weight.data.copy_(teacher.get_input_embeddings().weight.data)
        self.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
        self.lm_head.weight.data.copy_(teacher.get_output_embeddings().weight.data)

        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.layers):
            # self_layer.token_mixer.load_state_dict(teacher_layer.token_mixer.state_dict())
            self_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict())
            self_layer.input_layernorm.load_state_dict(teacher_layer.input_layernorm.state_dict())
            self_layer.post_attention_layernorm.load_state_dict(teacher_layer.post_attention_layernorm.state_dict())

            if copy_qkv:
                self_layer.self_attn.q_proj.load_state_dict(teacher_layer.self_attn.q_proj.state_dict())
                self_layer.self_attn.o_proj.load_state_dict(teacher_layer.self_attn.o_proj.state_dict())

                n_rep = self_layer.self_attn.v_proj.weight.shape[0] // teacher_layer.self_attn.v_proj.weight.shape[0]

                v_proj_unrolled = unroll_value_projection(teacher_layer.self_attn.v_proj, n_rep)
                k_proj_unrolled = unroll_value_projection(teacher_layer.self_attn.k_proj, n_rep)

                self_layer.self_attn.v_proj.load_state_dict(v_proj_unrolled.state_dict())
                self_layer.self_attn.k_proj.load_state_dict(k_proj_unrolled.state_dict())

                # self_layer.token_mixer.igate_preact.bias.data.fill_(torch.log(torch.tensor(2.0)))
                # self_layer.token_mixer.igate_preact.bias.data.fill_(-torch.log(torch.tensor(2.0)))
                #
                # # Init weight with small values
                # init.xavier_uniform_(self_layer.token_mixer.igate_preact.weight.data)
                # self_layer.token_mixer.igate_preact.weight.data *= 0.1
                # init.xavier_uniform_(self_layer.token_mixer.fgate_preact.weight.data)
                # self_layer.token_mixer.fgate_preact.weight.data *= 0.1


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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


from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
AutoConfig.register(LlamaGatedDeltaProductConfig.model_type, LlamaGatedDeltaProductConfig)
AutoModel.register(LlamaGatedDeltaProductConfig, GatedDeltaProductModel)
AutoModelForCausalLM.register(LlamaGatedDeltaProductConfig, GatedDeltaProductForCausalLM)

StudentConfig = LlamaGatedDeltaProductConfig
StudentModel = GatedDeltaProductForCausalLM
