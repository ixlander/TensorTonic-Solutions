import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    h_t = np.tanh(x_t @ Wx + h_prev @ Wh + b)
    return h_t
    