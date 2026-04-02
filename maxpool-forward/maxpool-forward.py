def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    H = len(X)
    W = len(X[0])

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    out = []

    for i in range(H_out):
        row = []
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride

            window_max = max(
                X[h_start + a][w_start + b]
                for a in range(pool_size)
                for b in range(pool_size)
            )
            row.append(window_max)
        out.append(row)

    return out