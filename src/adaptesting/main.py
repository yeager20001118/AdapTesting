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
        model=None,
        n_perm=100,
        kernel="gaussian",
        n_bandwidth=10,
        patience=50,
        seed=0,
        train_ratio=0.5,
        is_jax=False,
        is_permuted=False,
        is_log = False,
        is_history = False,
        is_label = False):

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
            X, Y, alpha, n_perm, kernel, n_bandwidth, seed, is_jax, is_permuted)
    # Supervised TST
    elif method == 'deep':
        # perform a MMD-Deep test
        p_value, mmd_value = deep(X, Y)
    elif method == 'clf':
        # perform a classifier two-sample test
        p_value, mmd_value = clf(
            X, Y, n_perm, model, data_type, train_ratio, patience, is_log, is_history, is_label)
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


def agg(X, Y, alpha, n_perm, kernel, n_bandwidth, seed, is_jax, is_permuted):
    # print(X)
    B1 = n_perm
    B2 = n_perm
    B3 = 50
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
    if is_permuted:
        # Permutation test
        V11, V10, V01 = permute_data(len(X), len(Y), seed, B1, B2, device, is_jax)
    else:
        # Wild bootstrap
        R = wild_bootstrap(len(X), len(Y), seed, B1, B2, device, is_jax)

    # Step 1: compute all simulated MMD estimates (efficient as in Appendix C in MMD-Agg paper)
    if is_permuted:
        M = generate_mmd_matrix(X, Y, kernel_bandwidths_l_list,
                                n_bandwidth, B1, B2, [V11, V10, V01], is_permuted)
    else:
        M = generate_mmd_matrix(X, Y, kernel_bandwidths_l_list, n_bandwidth, B1, B2, [R], is_permuted)

    mmd_values = M[:, B1].clone()
    M1_sorted = torch.sort(M[:, :B1+1], dim=1)[0]

    # Step 2: compute u_alpha_hat using the bisection method
    u, _ = compute_u(M, kernel_bandwidths_l_list, n_bandwidth, B1, B2, B3, weights, alpha)

    # Step 3: output test result
    p_vals = torch.mean(
        (M1_sorted - mmd_values.reshape(-1, 1) >= 0).float(), dim=1)
    all_weights = torch.zeros_like(p_vals)
    for j in range(len(kernel_bandwidths_l_list)):
        for i in range(n_bandwidth):
            all_weights[n_bandwidth * j + i] = weights[i]
    thresholds = u * all_weights

    # print(p_vals, thresholds)
    # Calculate scaled p-values
    scaled_p_vals = p_vals * (alpha / thresholds)
    # print(scaled_p_vals)

    min_p_value, min_idx = torch.min(scaled_p_vals, dim=0)
    mmd_value = mmd_values[min_idx]
    return min_p_value, mmd_value


def deep(X, Y):
    
    return 0, 0


def clf(X, Y, n_perm, model, data_type, train_ratio, patience, is_log, is_history, is_label):
    X_tr, X_te, Y_tr, Y_te = split_datasets(X, Y, train_ratio=train_ratio)
    device = X.device
    if data_type == 'tabular' and model is None:
        input_dim = X.size(1)
        hidden_dim = min(max(16, input_dim // 2), input_dim * 4)
        model = TabNetNoEmbeddings(
            input_dim=input_dim,
            output_dim=2,  # binary classification
            n_d=hidden_dim,
            n_a=hidden_dim,
            n_steps=2,       
            gamma=1,       
            n_independent=1,  
            n_shared=1,      
            virtual_batch_size=128,
            momentum=0.02
        ).to(device)
        model.encoder.group_attention_matrix = model.encoder.group_attention_matrix.to(device)
        # print(model)

    model, history = train_clf(X_tr, Y_tr, val_ratio=0.1, batch_size=128,
                      max_epoch=2000, lr=1e-3, patience=patience, model=model, is_log=is_log)
    
    if is_history:
        fig = plot_training_history(history)
        plt.show()

    p_value, mmd_value = test_clf(X_te, Y_te, n_perm, model, is_label)
    return p_value, mmd_value
