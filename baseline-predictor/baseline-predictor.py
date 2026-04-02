import numpy as np

def baseline_predict(ratings_matrix, target_pairs):
    R = np.array(ratings_matrix, dtype=float)
    pairs = np.array(target_pairs, dtype=int)

    mask = R != 0

    # Global mean
    mu = R[mask].mean()

    # User bias
    user_counts = mask.sum(axis=1)
    user_sums = R.sum(axis=1)
    user_means = np.divide(
        user_sums,
        user_counts,
        out=np.full(R.shape[0], mu),
        where=user_counts > 0
    )
    user_bias = user_means - mu

    # Item bias
    item_counts = mask.sum(axis=0)
    item_sums = R.sum(axis=0)
    item_means = np.divide(
        item_sums,
        item_counts,
        out=np.full(R.shape[1], mu),
        where=item_counts > 0
    )
    item_bias = item_means - mu

    # Predictions
    users = pairs[:, 0]
    items = pairs[:, 1]

    preds = mu + user_bias[users] + item_bias[items]

    return preds.tolist()