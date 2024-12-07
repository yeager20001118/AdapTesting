import torch
from .utils import *

def tst(
        X,
        Y,
        method='median',
        alpha=0.05,
        device="cpu",
        dtype=torch.float32,
        data_type='not_specified',
        n_perm=100,
        kernel="gaussian",
        n_bandwidth=10,
        seed=0):

    # print("test")

    # Check the X, Y are torch tensors
    if not isinstance(X, torch.Tensor):
        raise ValueError("X must be a torch tensor")
    if not isinstance(Y, torch.Tensor):
        raise ValueError("Y must be a torch tensor")

    # Ensure n_bandwidth is an integer > 0 for 'fuse' and 'agg' methods
    if method in ['fuse', 'agg']:
        if not isinstance(n_bandwidth, int):
            raise ValueError("n_bandwidth must be an integer")
        elif n_bandwidth < 1:
            raise ValueError("n_bandwidth must be greater than 0")

    # Ensure data_type is specified correctly for 'deep' and 'clf' methods
    if method in ['deep', 'clf'] and data_type not in ['tabular', 'image', 'text']:
        raise ValueError(
            "For 'deep' or 'clf' methods, data_type must be 'tabular', 'image', or 'text'")

    X = X.to(device=device, dtype=dtype)
    Y = Y.to(device=device, dtype=dtype)

    X, Y = check_shapes_and_adjust(X, Y)
    n_sample = len(X)  # represent the size of one sample

    # Unsupervised TST
    if method == 'median':
        # perform median heuristic mmd test
        p_value, mmd_value = median(X, Y, n_perm, kernel, seed)
    elif method == 'fuse':
        # perform a MMD-FUSE test
        p_value, mmd_value = fuse(X, Y)
    elif method == 'agg':
        # perform a MMD-Agg test
        p_value, mmd_value = agg(
            X, Y, alpha, n_perm, kernel, n_bandwidth, seed)
    # Supervised TST
    elif method == 'deep':
        # perform a MMD-Deep test
        p_value, mmd_value = deep(X, Y)
    elif method == 'clf':
        # perform a classifier two-sample test
        p_value, mmd_value = clf(X, Y)
    # Semi-supervised TST tbc...
    else:
        raise ValueError("Unsupported test type")

    if p_value < alpha:
        h = 1
        print(f"Reject the null hypothesis with p-value: {p_value}, "
              f"the MMD value is {mmd_value}.")
    else:
        h = 0
        print(f"Fail to reject the null hypothesis with p-value: {p_value}, "
              f"the MMD value is {mmd_value}.")

    return h, p_value


def median(X, Y, n_perm, kernel, seed):
    Z = torch.cat([X, Y], dim=0)
    norm = get_norms(kernel)[0]
    pairwise_matrix = torch_distance(Z, Z, norm)
    # print(Z)
    median = torch.median(pairwise_matrix)
    # print(median)
    p_value, mmd_value = mmd_permutation_test(
        X, Y, n_perm, kernel, [median], seed)

    return p_value, mmd_value


def fuse(X, Y):
    return 0, 0


def agg(X, Y, alpha, n_perm, kernel, n_bandwidth, seed):
    # print(X)
    B1 = 2000
    B2 = 2000
    B3 = 50
    m, n = len(X), len(Y)
    device = X.device
    kernel_bandwidths_l_list = []

    norms = get_norms(kernel)
    for norm in norms:
        dist = torch_distance(X, Y, norm, max_size=500, matrix=False)
        # print(torch.sort(dist)[0])
        bandwidths = compute_agg_bandwidths(dist, n_bandwidth)
        kernel_split = sorted(kernel.split("_"), reverse=True)
        if len(kernel_split) > 1:
            kernel_bandwidths_l_list.append(
                (kernel_split[norm-1], bandwidths, norm))
        else:
            kernel_bandwidths_l_list.append((kernel, bandwidths, norm))
    # print(kernel_bandwidths_l_list)
    weights = create_weights(n_bandwidth) / len(kernel_bandwidths_l_list)
    V11, V10, V01 = permute_data(len(X), len(Y), seed, B1, B2, device)

    # Step 1: compute all simulated MMD estimates (efficient as in Appendix C in MMD-Agg paper)
    N = n_bandwidth * len(kernel_bandwidths_l_list)
    M = torch.zeros((N, B1 + B2 + 1))
    last_norm_computed = 0
    for j in range(len(kernel_bandwidths_l_list)):
        kernel, bandwidths, norm = kernel_bandwidths_l_list[j]
        """ Since kernel_bandwidths_l_list is ordered "l1" first, "l2" second
            compute pairwise matrices the minimum amount of time
            store only one pairwise matrix at once """
        if norm != last_norm_computed:
            Z = torch.cat([X, Y], dim=0)
            pairwise_matrix = torch_distance(Z, Z, norm)
            last_norm_computed = norm
        for i in range(n_bandwidth):
            K = kernel_matrix(pairwise_matrix, kernel, bandwidths[i], True)

            # Set diagonal elements to zero
            K.fill_diagonal_(0)

            # Compute MMD permuted values
            M[n_bandwidth * j + i] = (
                torch.sum(V10 * (K @ V10), dim=0) * (n - m + 1) / (m * n * (m - 1)) +
                torch.sum(V01 * (K @ V01), dim=0) * (m - n + 1) / (m * n * (n - 1)) +
                torch.sum(V11 * (K @ V11), dim=0) / (m * n)
            )
    mmd_values = M[:, B1].clone()
    M1_sorted = torch.sort(M[:, :B1+1], dim=1)[0]
    M2 = M[:, B1+1:]

    # Step 2: compute u_alpha_hat using the bisection method
    quantiles = torch.zeros((N, 1))
    u_min = 0
    u_max = torch.min(1/weights)
    for _ in range(B3):
        u = (u_max + u_min) / 2
        for j in range(len(kernel_bandwidths_l_list)):
            for i in range(n_bandwidth):
                idx = (torch.ceil(
                    (B1 + 1) * (1 - u * weights[i])) - 1).to(torch.long)
                quantiles[n_bandwidth * j +
                          i] = M1_sorted[n_bandwidth * j + i, idx]

        P_u = torch.sum(torch.max(M2 - quantiles, dim=0)[0] > 0) / B2

        # Single line condition using torch.where
        u_min, u_max = torch.where(P_u <= alpha, torch.tensor(
            [u, u_max]), torch.tensor([u_min, u]))
    u = u_min
    for j in range(len(kernel_bandwidths_l_list)):
        for i in range(n_bandwidth):
            idx = (torch.ceil((B1 + 1) * (1 - u * weights[i])).long() - 1)
            quantiles[n_bandwidth * j +
                      i] = M1_sorted[n_bandwidth * j + i, idx]

    # Step 3: output test result
    p_vals = torch.mean(
        (M1_sorted - mmd_values.reshape(-1, 1) >= 0).float(), dim=1)
    all_weights = torch.zeros_like(p_vals)
    for j in range(len(kernel_bandwidths_l_list)):
        for i in range(n_bandwidth):
            all_weights[n_bandwidth * j + i] = weights[i]
    thresholds = u * all_weights

    # Calculate scaled p-values
    scaled_p_vals = p_vals * (alpha / thresholds)

    min_p_value, min_idx = torch.min(scaled_p_vals, dim=0)
    mmd_value = mmd_values[min_idx]
    return min_p_value, mmd_value


def deep(X, Y):
    return 0, 0


def clf(X, Y):
    return 0, 0
