import torch
import torch.nn as nn

def unroll_value_projection(v_proj: nn.Linear, n_rep: int) -> nn.Linear:
    """
    Takes a trained grouped-query value projection (e.g., 960 -> 320 for 10 heads)
    and unrolls it into a full-sized projection (e.g., 960 -> 960 for 30 heads)
    by repeating each head's weights n_rep times.
    
    Args:
        v_proj: nn.Linear of shape (in_features, n_kv_heads * head_dim)
        n_rep: int, number of repetitions per key/value head

    Returns:
        new_v_proj: nn.Linear of shape (in_features, n_kv_heads * head_dim * n_rep)
    """
    in_features = v_proj.in_features
    out_features = v_proj.out_features
    head_dim = 32  # or compute as out_features // num_kv_heads if dynamic

    num_kv_heads = out_features // head_dim
    new_out_features = out_features * n_rep

    # Create a new Linear layer with expanded output
    new_v_proj = nn.Linear(in_features, new_out_features, bias=v_proj.bias is not None)

    with torch.no_grad():
        for i in range(num_kv_heads):
            w_chunk = v_proj.weight[i * head_dim:(i + 1) * head_dim]  # (head_dim, in_features)
            if v_proj.bias is not None:
                b_chunk = v_proj.bias[i * head_dim:(i + 1) * head_dim]  # (head_dim)
            for j in range(n_rep):
                idx = (i * n_rep + j) * head_dim
                new_v_proj.weight[idx:idx + head_dim] = w_chunk
                if v_proj.bias is not None:
                    new_v_proj.bias[idx:idx + head_dim] = b_chunk

    return new_v_proj

