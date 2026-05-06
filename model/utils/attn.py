# Rotary Position Embedding (RoPE) helpers and grouped-query attention utilities.
#
# RoPE encodes token position by *rotating* the query/key vectors.
# Instead of adding a position vector to the embedding (like the original Transformer),
# it applies a rotation matrix whose angle depends on the position. This means
# the dot product q·k naturally captures *relative* distance between two tokens,
# which generalises better to sequence lengths not seen during training.

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Split the last dimension in half, then build [-x2, x1].
    # This is the 2D rotation trick: rotating vector (x1, x2) by 90° gives (-x2, x1).
    # Applied across all head dimensions, it achieves the full RoPE rotation.
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,       # query or key tensor, shape [..., seq_len, head_dim]
    cos: torch.Tensor,     # cosine part of the rotation, precomputed for each position
    sin: torch.Tensor,     # sine part of the rotation
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    # The rotation formula is:  x_rotated = x * cos + rotate_half(x) * sin
    # This is just Euler's rotation identity split into real and imaginary parts.
    # unsqueeze_dim adds a dimension so cos/sin broadcast across the heads axis.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Grouped Query Attention (GQA): Gemma uses fewer key/value heads than query heads.
    # For example, 8 KV heads shared across 32 query heads means n_rep = 4.
    # This function expands each KV head so there is one copy per query head,
    # making the shapes compatible for the standard matmul in attention.
    # Shape in:  [batch, num_kv_heads, seq_len, head_dim]
    # Shape out: [batch, num_kv_heads * n_rep, seq_len, head_dim]
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states  # Nothing to repeat — Q heads == KV heads
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
