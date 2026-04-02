import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    Input: (N, 224, 224, 3)
    Output: (N, 55, 55, 96)
    """
    N, H, W, C = image.shape

    kernel = 11
    stride = 4
    padding = 2   # <-- ключевой момент

    H_out = (H + 2 * padding - kernel) // stride + 1
    W_out = (W + 2 * padding - kernel) // stride + 1

    return np.zeros((N, H_out, W_out, 96), dtype=float)