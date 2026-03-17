import torch
from .utils import *


def hsicagg(X, Y, alpha, n_perm, seed, hsic_collection_parameters=(2, -2, 2), R=200, B3=50):
    """
    HSICAggInc: Efficient Aggregated HSIC independence test using Incomplete U-statistics.
    Reference: https://github.com/antoninschrab/agginc-paper
    Paper: "Efficient Aggregated Kernel Tests using Incomplete U-statistics" (NeurIPS 2022)

    inputs:
        X : (2N, d_X) tensor — paired data, split in half internally
        Y : (2N, d_Y) tensor — same number of rows as X
        alpha : significance level
        n_perm : number of wild bootstrap samples (B1 = B2 = n_perm)
        seed : random seed
        hsic_collection_parameters : (power, l_minus, l_plus) for bandwidth collection
        R : number of superdiagonals for incomplete U-statistic
        B3 : bisection steps for level correction

    output: (p_value, hsic_value)
    """
    device = X.device
    B1, B2 = n_perm, n_perm
    N = X.shape[0] // 2

    bandwidths_X, bandwidths_Y = compute_hsic_bandwidths(X, Y, N, hsic_collection_parameters)
    index_i, index_j = create_superdiagonal_indices(N, min(R, N - 1), device)
    h_XY = compute_h_hsic(X, Y, N, index_i, index_j, bandwidths_X, bandwidths_Y)
    bootstrap_values, original_value = hsic_wild_bootstrap(h_XY, index_i, index_j, N, B1, B2, seed)

    weights = create_weights(h_XY.shape[0]).to(device=device, dtype=X.dtype)
    u, _ = compute_u_hsic(bootstrap_values, original_value, weights, B1, B3, alpha)

    return compute_hsic_result(bootstrap_values, original_value, weights, u, B1, alpha)


def rdc(X, Y, n_perm=200, seed=0, num_features=20, scale=None, reg=1e-6):
    """
    Reference: https://github.com/lopezpaz/randomized_dependence_coefficient
    Paper: "The Randomized Dependence Coefficient" (NeurIPS 2013)

    output: (p_value, rdc_value)
    """
    X = adjust_independence_input(X)
    Y = adjust_independence_input(Y)

    def stat_func(X_curr, Y_curr, current_seed):
        generator = create_independence_generator(X_curr, current_seed)
        return compute_rdc_statistic(
            X_curr,
            Y_curr,
            generator,
            num_features=num_features,
            scale=scale,
            reg=reg,
        )

    return compute_permutation_p_value(stat_func, X, Y, n_perm, seed)


def fsic(X, Y, n_perm=200, seed=0, n_locations=5, reg=1e-6):
    """
    Reference: https://github.com/wittawatj/fsic-test/tree/master
    Paper: "An Adaptive Test of Independence with Analytic Kernel Embeddings" (ICML 2017)

    output: (p_value, fsic_value)
    """
    X = adjust_independence_input(X)
    Y = adjust_independence_input(Y)
    n = X.shape[0]
    n_locations = min(max(1, n_locations), n)

    generator = create_independence_generator(X, seed)
    x_indices = torch.randperm(n, generator=generator, device=X.device)[:n_locations]
    y_indices = torch.randperm(n, generator=generator, device=Y.device)[:n_locations]

    def stat_func(X_curr, Y_curr, current_seed):
        return compute_fsic_statistic(X_curr, Y_curr, x_indices, y_indices, reg=reg)

    return compute_permutation_p_value(stat_func, X, Y, n_perm, seed)
