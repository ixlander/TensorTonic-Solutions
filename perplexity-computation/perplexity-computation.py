import numpy as np

def perplexity(prob_distributions, actual_tokens):
    """
    prob_distributions: list/array of shape (N, V)
    actual_tokens: list/array of shape (N,)
    """
    probs = np.array(prob_distributions)
    targets = np.array(actual_tokens)

    # Берем вероятность правильного токена на каждом шаге
    p = probs[np.arange(len(targets)), targets]

    # Чтобы не словить log(0)
    p = np.clip(p, 1e-12, 1.0)

    # Cross-entropy
    H = -np.mean(np.log(p))

    # Perplexity
    return np.exp(H)