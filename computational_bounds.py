from copy import copy

import cvxpy as cp
import numpy as np
from numpy import linalg
from scipy import sparse

def generate_lower_bounds(z_hat, weights, A, b, t_max, idx_constrained=None,
                          suggested_design_params=True, eps=1e-8, verbose=True, solver="ECOS"):
    assert A.shape[0] == b.shape[0] and b.shape[0] == t_max.shape[0]
    assert np.all(t_max >= 0)

    n = A.shape[0]

    if idx_constrained is None:
        if verbose:
            print("No constrained indices passed. Assuming all "
                "indices are unconstrained.")
        idx_constrained = [[i] for i in range(n)]

    nu = cp.Variable(n)
    constraints = []
    dual_obj = - nu @ b + .5 * linalg.norm(weights * z_hat) ** 2

    # Could all be vectorized, but most time is spent solving the problem anyways.
    for S_k in idx_constrained:
        max_left_accumulate = 0
        max_right_accumulate = 0
        for j in S_k:
            if abs(weights[j]) <= eps:
                constraints.append(A[j,:] @ nu == 0)
                constraints.append(A[j,:] @ nu + nu[j] * t_max == 0)
            else:
                w2 = weights[j] ** 2
                max_left_accumulate += cp.square(A[:,j].T @ nu - w2 * z_hat[j]) / w2
                if t_max[j] > eps:
                    max_right_accumulate += cp.square(A[:,j].T @ nu + t_max[j] * nu[j] - w2 * z_hat[j]) / w2

        dual_obj += -.5 * cp.maximum(max_left_accumulate, max_right_accumulate)
        
    prob = cp.Problem(cp.Maximize(dual_obj), constraints)

    obj_value = prob.solve(solver=solver, verbose=verbose)

    if not suggested_design_params:
        return obj_value, nu.value

    max_left = (A @ nu.value - (weights ** 2) * z_hat) ** 2
    max_right = (A @ nu.value + nu.value * t_max - (weights ** 2) * z_hat) ** 2

    init_design = np.zeros(n)

    for S_k in idx_constrained:
        init_design_at_index = (np.sum(max_left[S_k]) <= np.sum(max_right[S_k]))
        init_design[S_k] = init_design_at_index * t_max[S_k]

    pinv_weights2 = np.copy(weights)
    eps_weight_indices = (weights < eps)
    pinv_weights2[~eps_weight_indices] = 1 / (weights[~eps_weight_indices] ** 2)
    pinv_weights2[eps_weight_indices] = 0

    init_field = z_hat - pinv_weights2 * (A @ nu.value + init_design * nu.value)

    return obj_value, nu.value, init_design, init_field


def complex_to_real(z_hat, weights, A, b, t_max, idx_constrained=None):
    assert A.shape[0] == b.shape[0] and b.shape[0] == t_max.shape[0]
    n = A.shape[0]

    if idx_constrained is None:
        idx_constrained = [[i] for i in range(n)]

    A_real, A_imag = np.real(A), np.imag(A)

    z_hat_t = np.r_[np.real(z_hat), np.imag(z_hat)]
    weights_t = np.r_[weights, weights]

    A_t = sparse.bmat(
        [[A_real, -A_imag],
        [A_imag, A_real]]
    ).tocsc()
    b_t = np.r_[np.real(b), np.imag(b)]

    new_idx_constrained = []

    for S_k in idx_constrained:
        curr_s_k = []

        for j in S_k:
            curr_s_k.append(j)
            curr_s_k.append(j + n)
        
        new_idx_constrained.append(curr_s_k)

    t_max_t = np.r_[t_max, t_max]

    return z_hat_t, weights_t, A_t, b_t, t_max_t, new_idx_constrained

def real_to_complex_design(init_design, init_field):
    assert init_design.shape[0] % 2 == 0

    n = init_design.shape[0] // 2

    return init_design[:n], init_field[:n] + 1.j * init_field[n:]