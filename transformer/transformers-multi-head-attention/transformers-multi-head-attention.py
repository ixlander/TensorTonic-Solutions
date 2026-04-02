import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Q, K, V: (batch, seq_len, d_model)
    W_q, W_k, W_v, W_o: (d_model, d_model)
    Returns: (batch, seq_len, d_model)
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Linear projections
    Q_proj = Q @ W_q   # (B, T, D)
    K_proj = K @ W_k   # (B, T, D)
    V_proj = V @ W_v   # (B, T, D)

    # Split into heads: (B, H, T, d_k)
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scores = Q_heads @ K_heads.transpose(0, 1, 3, 2) / np.sqrt(d_k)   # (B, H, T, T)
    attn_weights = softmax(scores, axis=-1)
    head_outputs = attn_weights @ V_heads                              # (B, H, T, d_k)

    # Concatenate heads: (B, T, D)
    concat = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # Output projection
    output = concat @ W_o                                              # (B, T, D)

    return output