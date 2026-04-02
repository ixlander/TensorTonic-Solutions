import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    indices = np.arange(N)

    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            np.random.shuffle(indices)

    # делим на k фолдов (размеры автоматически отличаются максимум на 1)
    folds = np.array_split(indices, k)

    result = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        result.append((train_idx.astype(int), val_idx.astype(int)))

    return result