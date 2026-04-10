import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowCausalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, T, C)
        """
        B, T, C = x.size()
        H, D = self.num_heads, self.head_dim
        W = self.window_size

        # Project and reshape
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

        # Pad k and v on the left with zeros for causality
        pad = torch.zeros(B, H, W - 1, D, device=x.device, dtype=x.dtype)
        k_padded = torch.cat([pad, k], dim=2)  # (B, H, T + W - 1, D)
        v_padded = torch.cat([pad, v], dim=2)

        # Unfold to get sliding window chunks
        k_windows = k_padded.unfold(dimension=2, size=W, step=1)  # (B, H, T, W, D)
        v_windows = v_padded.unfold(dimension=2, size=W, step=1)  # (B, H, T, W, D)

        # Compute attention scores
        q = q.unsqueeze(3)  # (B, H, T, 1, D)
        scores = torch.matmul(q, k_windows.transpose(-2, -1)).squeeze(3)  # (B, H, T, W)
        scores = scores / (D ** 0.5)

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, W)

        # Apply attention weights
        context = torch.einsum('bhtw,bhtwd->bhtd', attn_weights, v_windows)  # (B, H, T, D)

        # Reshape back
        out = context.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.out_proj(out)


def sliding_window_attention(q, k, v, window_size=64):
    """
    q: (B, H, T, D)
    k: (B, H, T, D)
    v: (B, H, T, D)
    """
    B, H, T, D = q.size()
    W = window_size
    # Pad k and v on the left with zeros for causality
    pad = torch.zeros(B, H, W - 1, D, device=q.device, dtype=q.dtype)
    k_padded = torch.cat([pad, k], dim=2)  # (B, H, T + W - 1, D)
    v_padded = torch.cat([pad, v], dim=2)
    # Unfold to get sliding window chunks
    k_windows = k_padded.unfold(dimension=2, size=W, step=1)  # (B, H, T, W, D)
    v_windows = v_padded.unfold(dimension=2, size=W, step=1)  # (B, H, T, W, D)

    # Compute attention scores
    q = q.unsqueeze(3)  # (B, H, T, 1, D)
    # scores = torch.matmul(q, k_windows.transpose(-2, -1)).squeeze(3)  # (B, H, T, W)
    scores = torch.einsum("bht1d,bhtwd->bhtw", q, k_windows)  # (B, H, T, W)
    scores = scores / (D ** 0.5)
    scores = scores / (D ** 0.5)
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, W)
    # Apply attention weights
    context = torch.einsum('bhtw,bhtwd->bhtd', attn_weights, v_windows)  # (B, H, T, D)
    return context


def sliding_window_attention2(q, k, v, window_size=64):
    B, H, T, D = q.size()

    attn_output = torch.zeros_like(q)

    for t in range(T):
        start = max(0, t - window_size)
        q_t = q[:, :, t]  # (B, H, D)
        k_slice = k[:, :, start:t+1]  # (B, H, W, D)
        v_slice = v[:, :, start:t+1]  # (B, H, W, D)

        # (B, H, 1, D) x (B, H, D, W) → (B, H, 1, W)
        scores = torch.einsum("bhd,bhwd->bhw", q_t, k_slice) / (D ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, W)

        # (B, H, W) x (B, H, W, D) → (B, H, D)
        context = torch.einsum("bhw,bhwd->bhd", attn_weights, v_slice)
        attn_output[:, :, t] = context

    # Reshape back to (B, T, C)
    attn_output = attn_output.transpose(1, 2)
    return attn_output

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

_compiled_flex_attention = torch.compile(flex_attention)
_block_mask_cache = {}


def get_block_mask(seq_len, window_size, with_memory=False, mem_window: int = 4):
    # Cache based on (seq_len, window_size) tuple
    key = (seq_len, window_size)
    if key not in _block_mask_cache:
        if with_memory:
            def swa_with_memory(b, h, q_idx, kv_idx):
                """ Sliding window causal attention with memory.

                Add mask so model always attents to first m tokens in the sequence.

                """
                causal_mask = q_idx >= kv_idx
                window_mask = (q_idx - kv_idx) <= window_size
                memory_mask = kv_idx < mem_window
                return (causal_mask & window_mask) | memory_mask

            sliding_window_mask = swa_with_memory
        else:
            def sliding_window_mask(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)

        block_mask = create_block_mask(
            sliding_window_mask,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            _compile=True
        )
        _block_mask_cache[key] = block_mask
    return _block_mask_cache[key]


def causal_sliding_window_attention(query, key, value, window_size=64, with_memory=False, mem_window: int = 4):
    """
    Perform causal sliding window attention using cached mask and compiled function.
    """
    seq_len = query.size(2)
    block_mask = get_block_mask(seq_len, window_size, with_memory=with_memory, mem_window=mem_window)
    return _compiled_flex_attention(query, key, value, block_mask=block_mask)


