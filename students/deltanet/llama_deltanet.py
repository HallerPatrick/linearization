from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel

from einops import rearrange 
from transformers.utils import logging
import transformers.models.llama.modeling_llama as llama
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from fla.models.delta_net import DeltaNetConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.layers.delta_net import (
    DeltaNet as FlaDeltaNet,
    chunk_delta_rule,
    fused_recurrent_delta_rule,
    sum_norm, elu_p1
)
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
# from fla.ops.delta_rule.parallel import naive_delta_rule_parallel
# from fla.ops.delta_rule.parallel import  parallel_delta_rule
from .utils import unroll_value_projection

logger = logging.get_logger(__name__)

def parallel_deltanet_with_attention(q, k, v, beta, mask=None):
    """
    Parallel DeltaNet with attention weights extraction.
    q, k, v: (B, L, H, D)
    beta: (B, L, H)
    mask: optional causal mask (L, L)
    Returns:
        out: (B, L, H, D)
        attn_weights: (B, H, L, L)
    """
    B, L, H, D = q.shape
    device = q.device

    # Normalize q, k
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)

    # (B, L, H, D) -> (B, H, L, D)
    k_ = k.permute(0, 2, 1, 3)
    q_ = q.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)
    beta_ = beta.permute(0, 2, 1)  # (B, H, L)

    # Compute T matrix from UT transform
    # KKt = torch.einsum('bhld,bhme->bhle', k_, k_)  # (B, H, L, L)
    KKt = torch.einsum('bhld,bhmd->bhlm', k_, k_)  # (B, H, L, L)
    tril = torch.tril(torch.ones(L, L, device=device)).unsqueeze(0).unsqueeze(0)
    tril_KKt = KKt * tril

    T = torch.linalg.solve(
        torch.eye(L, device=device, dtype=beta_.dtype).unsqueeze(0).unsqueeze(0) + tril_KKt * beta_.unsqueeze(-1),
        torch.diag_embed(beta_).to(torch.float32)
    )  # (B, H, L, L)

    # Compute pseudo values
    # U = torch.einsum('bhij,bhjd->bhid', T, v_)  # (B, H, L, D)
    # W = torch.einsum('bhij,bhjd->bhid', T, k_)  # (B, H, L, D)

    # Compute S[t] = initial state assumed to be 0
    # S = torch.zeros(B, H, D, device=device)

    # # Compute output: O = q @ S^T + (qK^T ⊙ M)(U - WS^T)
    # qS = torch.einsum('bhld,bhd->bhl', q_, S)  # (B, H, L)

    raw_attn = torch.matmul(q_, k_.transpose(-1, -2))  # (B, H, L, L)

    if mask is not None:
        raw_attn = raw_attn.masked_fill(mask == 0, 0)

    # Apply T to get final attention matrix
    attn_weights = torch.matmul(raw_attn, T.to(torch.bfloat16))  # (B, H, L, L)

    # Apply attention mask
    tril = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
    attn_weights = torch.where(tril, attn_weights, -float("inf"))

    return None, attn_weights

    # # Final output computation
    # update = torch.matmul(attn_weights, v_ - torch.einsum('bhd,bhld->bhld', S, W))  # (B, H, L, D)
    # out = qS.unsqueeze(-1) + update  # (B, H, L, D)
    #
    # return out.permute(0, 2, 1, 3), attn_weights  # (B, L, H, D), (B, H, L, L)


def naive_delta_rule_parallel(q, k, v, beta, BM=128, BN=32):
    B, H, S, D = q.shape
    device = q.device
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

    diag_beta = torch.diag_embed(beta)
    KKt = k @ k.transpose(-1, -2)
    L = torch.tril(diag_beta @ KKt, diagonal=-1)
    I = torch.eye(S, device=beta.device)[None, None]
    T = torch.inverse(I + L).to(diag_beta.dtype) @ diag_beta
    attn = torch.matmul(q, k.transpose(-1, -2))  # (B, H, L, L)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)
    attn = attn.masked_fill(mask, 0)
    attn = attn @ T  # (B, H, L, L)
    return None, attn


def _naive_delta_rule_parallel(q, k, v, beta, BM=128, BN=32):
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    b, h, l, d_k = q.shape
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    # compute (I - tri(diag(beta) KK^T))^{-1}
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BN), [q, k, v, k_beta])
    mask = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=0)
    T = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BN):
        T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    T = T + torch.eye(BN, dtype=q.dtype, device=q.device)

    mask2 = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=1)
    A_local = (q @ k.transpose(-1, -2)).masked_fill(mask2, 0) @ T
    o_intra = A_local @ v

    # apply cumprod transition matrices on k to the last position within the chunk
    k = k - ((k @ k.transpose(-1, -2)).masked_fill(mask, 0) @ T).transpose(-1, -2) @ k_beta
    # apply cumprod transition matrices on q to the first position within the chunk
    q = q - A_local @ k_beta
    o_intra = A_local @ v

    A = torch.zeros(b, h, l, l, device=q.device)

    q, k, v, k_beta, o_intra = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, v, k_beta, o_intra])
    o = torch.empty_like(v)
    for i in range(0, l, BM):
        q_i = q[:, :, i:i+BM]
        o_i = o_intra[:, :, i:i+BM]
        # intra block
        for j in range(i + BM - 2 * BN, i-BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            mask = torch.arange(i, i+BM) >= (j + BN)
            A_ij = A_ij.masked_fill_(~mask[:, None].to(A_ij.device), 0)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i += A_ij @ v[:, :, j:j+BN]
        # inter block
        for j in range(i - BN, -BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i += A_ij @ v[:, :, j:j+BN]
        o[:, :, i:i+BM] = o_i

    for i in range(0, l//BN):
        A[:, :, i*BN:i*BN+BN, i*BN:i*BN+BN] = A_local[:, :, i]

    return o, A

class LlamaDeltaNetConfig(DeltaNetConfig):
    rms_norm_eps: float = 1e-6
    mlp_bias: bool = False
    intermediate_size: int = 1024

class DeltaNet(nn.Module):

    def __init__(
        self,
        d_model: int = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        mode: str = 'chunk',
        use_beta: bool = False,
        use_gate: bool = False,
        use_output_norm: bool = True,
        use_elu: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        qk_norm: str = 'l2',
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        **kwargs
    ):
        super().__init__()

        self.mode = mode
        self.qk_norm = qk_norm
        
        expand_k = 1.0
        expand_v = 1.0

        # assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_output_norm = use_output_norm
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.norm_first = norm_first
        self.layer_idx = layer_idx

        self.silu = nn.SiLU()

        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        if norm_first:
            self.norm = RMSNorm(self.hidden_size, eps=norm_eps)

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.use_beta = use_beta
        self.use_elu = use_elu
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )
        else:
            pass
            # raise UserWarning(
            #     "ShortConvolution is crucial to the performance. "
            #     "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            # )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.apply(self._initialize_weights)

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
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # change to inference mode.
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        # mode = self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            position_ids = kwargs.get('position_ids', None)
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_q,
                                            output_final_state=use_cache)
                                            # seq_idx=position_ids)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_k,
                                            output_final_state=use_cache)
                                            # seq_idx=position_ids)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_v,
                                            output_final_state=use_cache)
                                            # seq_idx=position_ids)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = F.silu(self.v_proj(hidden_states))

        q, k, v = map(lambda x: rearrange(x, '... (h d) -> ... h d', h=self.num_heads), (q, k, v))

        if self.qk_norm == 'sum':
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)

        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = q.new_ones(q.shape[0], q.shape[1], q.shape[2])

        if self.allow_neg_eigval:
            beta = beta * 2.

        # dealing with padding
        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        # if output_attentions:
        #     q, k, v = map(lambda x: rearrange(x, 'b t h d -> b h t d'), [q, k, v])
        #     beta = rearrange(beta, 'b t h -> b h t')
        #     o, attentions = naive_delta_rule_parallel(
        #     # o, attentions = parallel_deltanet_with_attention(
        #     # o, attentions = parallel_delta_rule(
        #         q=q,
        #         k=k,
        #         v=v,
        #         beta=beta,
        #         # output_attentions=True
        #     )
        #     return None, attentions, None
        
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                # offsets=cu_seqlens,
                use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                # offsets=cu_seqlens,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values



class DeltaNetDecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DeltaNet(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                use_gate=config.use_gate,
                num_heads=config.num_heads,
                use_beta=config.use_beta,
                # num_kv_heads=config.num_kv_heads,
                # feature_map=config.feature_map,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                # conv_bias=config.conv_bias,
                # allow_neg_eigval=config.allow_neg_eigval,
                qk_norm=config.qk_norm,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx
                # use_output_gate=config.use_output_gate,
                # gate_fn=config.hidden_act,
                # elementwise_affine=config.elementwise_affine,
                # clamp_min=config.clamp_min,
                # fuse_norm=config.fuse_norm,
            )
        self.mlp = llama.LlamaMLP(config)
        self.input_layernorm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        past_key_values = None,
        output_self_attn: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if output_self_attn:
            output_attentions = False
        
        hidden_states, attentions, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )

        if output_attentions:
            return hidden_states, attentions

        if output_self_attn:
            return hidden_states, attentions

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)
        # if output_attentions:
        #     outputs += (self_attn_weights,)

        return outputs

class DeltaNetModel(llama.LlamaPreTrainedModel):
    config_class = LlamaDeltaNetConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeltaNetDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

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
                hidden_states, attn, past_key_values = layer( 
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

class LlamaDeltaNetPreTrainedModel(PreTrainedModel):
    config_class = LlamaDeltaNetConfig
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



class DeltaNetForCausalLM(LlamaDeltaNetPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    config_class = LlamaDeltaNetConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = DeltaNetModel(config)
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
            for param in layer.self_attn.parameters():
                param.requires_grad = True

    def copy_from_teacher(self, teacher, copy_qkv=True):
        assert len(self.model.layers) == len(teacher.model.layers)

        self.model.embeddings.weight.data.copy_(teacher.get_input_embeddings().weight.data)
        self.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
        self.lm_head.weight.data.copy_(teacher.get_output_embeddings().weight.data)

        for self_layer, teacher_layer in zip(self.model.layers, teacher.model.layers):
            # self_layer.self_attn.load_state_dict(teacher_layer.self_attn.state_dict())
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

class DeltaNetForSequenceClassification(llama.LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeltaNetModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
AutoConfig.register(LlamaDeltaNetConfig.model_type, LlamaDeltaNetConfig)
AutoModel.register(LlamaDeltaNetConfig, DeltaNetModel)
AutoModelForCausalLM.register(LlamaDeltaNetConfig, DeltaNetForCausalLM)

StudentConfig = LlamaDeltaNetConfig
StudentModel = DeltaNetForCausalLM
