import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    E_grad_sq = np.asarray(E_grad_sq, dtype=float)
    E_update_sq = np.asarray(E_update_sq, dtype=float)

    new_E_grad_sq = rho * E_grad_sq + (1 - rho) * (grad ** 2)
    update = - (np.sqrt(E_update_sq + eps) / np.sqrt(new_E_grad_sq + eps)) * grad
    new_E_update_sq = rho * E_update_sq + (1 - rho) * (update ** 2)
    new_w = w + update

    return new_w, new_E_grad_sq, new_E_update_sq