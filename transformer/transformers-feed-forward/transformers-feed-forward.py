import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    x: (batch, seq_len, d_model)
    W1: (d_model, d_ff)
    b1: (d_ff,)
    W2: (d_ff, d_model)
    b2: (d_model,)
    """
    # First linear
    hidden = x @ W1 + b1

    # ReLU
    hidden = np.maximum(0, hidden)

    # Second linear
    out = hidden @ W2 + b2

    return out