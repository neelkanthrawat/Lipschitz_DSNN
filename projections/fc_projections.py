import torch
import numpy as np

def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-10)
    normalized_tensor = tensor / norm
    return normalized_tensor

def identity(weights, lipschitz_goal):
    """no projection"""
    return weights

def l2_normalization_fc(weights, lipschitz_goal):
    current_lipschitz = torch.linalg.matrix_norm(weights, 2)
    new_weights = lipschitz_goal * weights / current_lipschitz

    return new_weights

def bjorck_orthonormalize_fc(weights, lipschitz_goal, beta=0.5, iters=15):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    We only use the order 1.
    """
    # I am still not sure why are we dividing by sqrt(m*n)
    w = weights/ np.sqrt(weights.shape[0] * weights.shape[1]) ### what would happen if I dont do this
    for _ in range(iters):
        w_t_w = w.t().mm(w)
        w = (1 + beta) * w - beta * w.mm(w_t_w)
    new_weights = lipschitz_goal * w  

    return w # why are we not returning new_weights? it does not make any sense to me!


### layer-wise orthogonal training
def layerwise_orthogonal_fc(weights, lipschitz_goal=1, beta =0.5, iters = 10):
    """
    Layer-wise Orthogonal training (LOT)
    Reference: 1. (Main paper:) Xu et. al NeurIPS-2022-lot-layer-wise-orthogonal-training-on-improving-l2-certified-robustness-Paper-Conference
            2. Prach et. al. : 1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness, Pg. num 3
    """

    w = weights / np.sqrt(weights.shape[0] * weights.shape[1]) ### what would happen if I dont do this
    y = weights.t().mm(w)
    z = torch.eye(y.shape[0])
    for _ in range(iters):
        zi_yi = z.mm(y)
        diff_mat = 3 * torch.eye(zi_yi.shape[0]) - zi_yi
        y = beta * y.mm(diff_mat)
        z = beta * diff_mat.mm(z)

    new_weights = lipschitz_goal * w 
    return new_weights

