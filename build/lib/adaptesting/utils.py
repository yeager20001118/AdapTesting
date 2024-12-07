import torch
import numpy as np
import jax.random as jrandom
import jax.numpy as jnp
from .constants import *


def check_shapes_and_adjust(X, Y):
    # Check if X and Y have the same length
    if len(X) != len(Y):
        # If not, use the minimum length
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]

    # Check if the non-batch dimensions of each tensor in X and Y are the same
    if X.size()[1:] != Y.size()[1:]:
        # If the non-batch shape of corresponding tensors is not the same, raise an error
        raise ValueError(
            f"Shape mismatch : X with dim shape {X.size()[1:]}, but Y has dim shape {Y.size()[1:]}")

    return X, Y


def get_norms(kernel):
    norms = []
    flag = False
    kernel = kernel.lower()  # Case-insensitive comparison

    for idx, kernel_group in enumerate(KERNEL_LIST):
        if kernel in kernel_group:
            flag = True
            norms.append(idx + 1)

    if flag:
        return norms
    else:
        raise ValueError(f"Kernel '{kernel}' currently not supported, we only support \n \
                     L1 norm kernel {KERNEL_LIST[0]} and \n \
                     L2 norm kernel {KERNEL_LIST[1]}")


def torch_distance(X, Y, norm=2, max_size=None, matrix=True):
    # Ensure X and Y are at least 2D (handles 1D cases)
    if X.dim() == 1:
        X = X.view(-1, 1)
    if Y.dim() == 1:
        Y = Y.view(-1, 1)

    # Broadcasting the subtraction operation across all pairs of vectors
    diff = X[None, :, :] - Y[:, None, :]

    if norm == 2:
        # Computing the L2 distance (Euclidean distance)
        dist = torch.sqrt(torch.sum(diff**2, dim=-1))
    elif norm == 1:
        # Computing the L1 distance (Manhattan distance)
        dist = torch.sum(torch.abs(diff), dim=-1)
    else:
        raise ValueError("Norm must be L1 or L2")

    if max_size:
        dist = dist[:max_size, :max_size]

    if matrix:
        return dist
    else:
        m = dist.shape[0]
        indices = torch.triu_indices(m, m, offset=0)
        return dist[indices[0], indices[1]]


def gaussian_kernel(pairwise_matrix, bandwidth, scale=False):
    d = pairwise_matrix / bandwidth

    if scale:
        return torch.exp(-(d**2))

    return torch.exp(-(d**2) / 2)


def laplace_kernel(pairwise_matrix, bandwidth):
    d = pairwise_matrix / bandwidth
    return torch.exp(-d)


def kernel_matrix(pairwise_matrix, kernel, bandwidth, scale=False):
    if kernel == "gaussian":
        return gaussian_kernel(pairwise_matrix, bandwidth, scale)
    elif kernel == "laplace":
        return laplace_kernel(pairwise_matrix, bandwidth)


def mmd_u(pairwise_matrix, n, m, kernel, bandwidth):
    # Compute the full Gaussian kernel matrix from the pairwise distance matrix
    K = kernel_matrix(pairwise_matrix, kernel, bandwidth)

    # Extract submatrices for XX, YY, and XY
    K_XX = K[:n, :n]
    K_YY = K[n:, n:]
    K_XY = K[:n, n:]

    # Ensure diagonal elements are zero (no self-comparison) for XX and YY
    K_XX.fill_diagonal_(0)
    K_YY.fill_diagonal_(0)

    # Calculate each term of the MMD_u^2
    mmd_u_squared = (K_XX.sum() / (n * (n - 1))) + \
        (K_YY.sum() / (m * (m - 1))) - (2 * K_XY.sum() / (n * m))
    return mmd_u_squared


def mmd_permutation_test(X, Y, num_permutations=100, kernel="gaussian", params=[1.0], kk=0):
    """Perform MMD permutation test and return the p-value."""

    if kernel == "gaussian":
        bandwidth = params[0]
    elif kernel == "laplace":
        bandwidth = params[0]

    norm = get_norms(kernel)[0]

    Z = torch.cat((X, Y))
    pairwise_matrix = torch_distance(Z, Z, norm)
    n = len(X)
    m = len(Y)

    observed_mmd = mmd_u(pairwise_matrix, n, m, kernel, bandwidth)
    count = 0
    # torch.manual_seed(kk+10)
    for _ in range(num_permutations):
        perm = torch.randperm(Z.size(0))
        perm_X = Z[perm[:n]]
        perm_Y = Z[perm[n:]]
        perm_Z = torch.cat((perm_X, perm_Y))
        pairwise_matrix = torch_distance(perm_Z, perm_Z, norm)
        perm_mmd = mmd_u(pairwise_matrix, n, m, kernel, bandwidth)
        if perm_mmd >= observed_mmd:
            count += 1

    p_value = count / num_permutations
    return p_value, observed_mmd

# Helper functions for MMD-Agg test


def compute_agg_bandwidths(dist, n_bandwidth):

    device = dist.device
    # Replace zero distances with median
    dist = dist + (dist == 0) * torch.median(dist)

    # Sort distances
    dd = torch.sort(dist)[0]  # torch.sort returns (values, indices)
    # print(len(dd))
    idx = torch.floor(torch.tensor(len(dd) * 0.05)).to(torch.int64)
    # print(idx, dd[idx])
    if torch.min(dist) < 10 ** (-1):
        lambda_min = torch.maximum(
            dd[idx],
            torch.tensor(10 ** (-1)).to(device)
        )
    else:
        lambda_min = torch.min(dist)

    # Adjust lambda_min and compute lambda_max
    lambda_min = lambda_min / 2
    lambda_max = torch.maximum(
        torch.max(dist),
        torch.tensor(3 * 10 ** (-1)).to(device)
    )
    lambda_max = lambda_max * 2

    # Compute power sequence
    power = (lambda_max / lambda_min) ** (1 / (n_bandwidth - 1))
    # print(lambda_min, lambda_max, power)
    # Generate geometric sequence of bandwidths
    bandwidths = torch.pow(power, torch.arange(
        n_bandwidth, device=device)) * lambda_min

    return bandwidths


def create_weights(n_bandwidth):
    return torch.full((n_bandwidth,), 1/n_bandwidth)


def permute_data(m, n, seed, B1, B2, device):
    # Set random seed
    key = jrandom.PRNGKey(seed)
    key, subkey = jrandom.split(key)
    base_array = jnp.arange(m + n)  # Create once
    repeated_array = jnp.tile(base_array, (B1 + B2 + 1, 1))
    idx = jrandom.permutation(
        subkey, repeated_array, axis=1, independent=True)

    # print(idx)

    idx = torch.from_numpy(np.array(idx)).long().to(device)

    v11 = torch.cat([torch.ones(m), -torch.ones(n)]).to(device)  # (m+n,)
    V11i = v11.repeat(B1 + B2 + 1, 1)  # (B1+B2+1, m+n)
    V11 = torch.gather(V11i, 1, idx)  # Permute each row according to idx
    V11[B1] = v11  # Set B1-th row back to original vector (no permutation)
    V11 = V11.T  # (m+n, B1+B2+1)

    # Create v10 vector and V10 matrix
    v10 = torch.cat([torch.ones(m), torch.zeros(n)]).to(device)
    V10i = v10.repeat(B1 + B2 + 1, 1)
    V10 = torch.gather(V10i, 1, idx)
    V10[B1] = v10
    V10 = V10.T

    # Create v01 vector and V01 matrix
    v01 = torch.cat([torch.zeros(m), -torch.ones(n)]).to(device)
    V01i = v01.repeat(B1 + B2 + 1, 1)
    V01 = torch.gather(V01i, 1, idx)
    V01[B1] = v01
    V01 = V01.T

    return V11, V10, V01
