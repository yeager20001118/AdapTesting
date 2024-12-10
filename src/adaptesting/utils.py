import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tabnet.tab_network import TabNetNoEmbeddings
import torchvision
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import jax.random as jrandom
import jax.numpy as jnp
import matplotlib.pyplot as plt
from .constants import *

#########################################################################################################

####################################### General helper functions ########################################


class DefaultImageModel(nn.Module):

    def __init__(self, n_channels=3, weights='DEFAULT', image_size=32):
        super().__init__()
        self.image_size = image_size

        # Load base ResNet with new weights parameter
        if weights == 'DEFAULT':
            weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.resnet = torchvision.models.resnet18(weights=weights)

        # Modify input layer if needed
        if n_channels != 3:
            self.resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7,
                                            stride=2, padding=3, bias=False)

        # For very small images (like MNIST 28x28 or CIFAR 32x32)
        if image_size < 64:
            # Modify first conv layer to have smaller stride
            self.resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3,
                                            stride=1, padding=1, bias=False)
            # Remove maxpool layer
            self.resnet.maxpool = nn.Identity()

        # Store the original fc layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 100)

    def forward(self, x):
        output = self.resnet(x)
        # Return both output and a zero sparse loss to match TabNet interface
        sparse_loss = torch.tensor(0.0, device=x.device)
        return output, sparse_loss

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
    # print(kernel)
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


def torch_distance(X, Y, norm=2, max_size=None, matrix=True, is_squared=False):
    if X.dim() == 4:
        # Flatten to (batch_size, channel*height*width)
        X = X.view(X.size(0), -1)
    if Y.dim() == 4:
        # Flatten to (batch_size, channel*height*width)
        Y = Y.view(Y.size(0), -1)

    # Ensure X and Y are at least 2D (handles 1D cases)
    if X.dim() == 1:
        X = X.view(-1, 1)
    if Y.dim() == 1:
        Y = Y.view(-1, 1)

    # Broadcasting the subtraction operation across all pairs of vectors
    diff = X[None, :, :] - Y[:, None, :]
    # print(diff.size())

    if norm == 2:
        # Computing the L2 distance (Euclidean distance)
        if is_squared:
            # If the pairwise matrix is squared
            dist = torch.sum(diff**2, dim=-1)
        else:
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


def gaussian_kernel(pairwise_matrix, bandwidth, scale=False, is_squared=False):
    d = pairwise_matrix / bandwidth

    if scale:
        return torch.exp(-(d**2))
    
    if is_squared:
        # If the pairwise matrix is squared
        return torch.exp(-(d/bandwidth) / 2)

    return torch.exp(-(d**2) / 2)


def laplace_kernel(pairwise_matrix, bandwidth, scale=False):
    d = pairwise_matrix / bandwidth

    if scale:
        return torch.exp(-d * torch.sqrt(torch.tensor(2.0)))
    
    return torch.exp(-d)


def kernel_matrix(pairwise_matrix, kernel, bandwidth, scale=False):
    if kernel == "gaussian":
        return gaussian_kernel(pairwise_matrix, bandwidth, scale)
    elif kernel == "laplace":
        return laplace_kernel(pairwise_matrix, bandwidth, scale)


def mmd_u(K, n, m, is_var=False):
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
    
    if is_var:
        h_matrix = K_XX + K_YY - K_XY - K_XY.t()
        row_means = h_matrix.sum(1) / m
        V1 = torch.dot(row_means, row_means) / m
        V2 = h_matrix.sum() / (n * n)
        variance = 4 * (V1 - V2**2)
        return mmd_u_squared, variance
    return mmd_u_squared


def mmd_permutation_test(X, Y, num_permutations=100, kernel="gaussian", params=[1.0], kk=0, data_type="tabular"):
    """Perform MMD permutation test and return the p-value."""

    if kernel == "gaussian":
        bandwidth = params[0]
    elif kernel == "laplace":
        bandwidth = params[0]
    elif kernel == "deep":
        c_epsilon, b_q, b_phi, model = params

    norm = get_norms(kernel)[0]

    Z = torch.cat((X, Y))
    pairwise_matrix = torch_distance(Z, Z, norm)
    n = len(X)
    m = len(Y)

    # Compute the Kernel matrix
    if kernel == "deep":
        if data_type == 'tabular':
            f = stack_representation(model)
        else:
            f = model
        fz = f(Z)[0]
        pairwise_matrix_f = torch_distance(fz, fz, norm)
        K_q = gaussian_kernel(pairwise_matrix, b_q)
        K_phi = gaussian_kernel(pairwise_matrix_f, b_phi)
        K = (1-c_epsilon) * K_phi * K_q + c_epsilon * K_q
    else:
        K = kernel_matrix(pairwise_matrix, kernel, bandwidth)

    observed_mmd = mmd_u(K, n, m)
    count = 0
    # torch.manual_seed(kk+10)
    for _ in range(num_permutations):
        perm = torch.randperm(Z.size(0), device=Z.device)
        perm_X = Z[perm[:n]]
        perm_Y = Z[perm[n:]]
        perm_Z = torch.cat((perm_X, perm_Y))
        pairwise_matrix = torch_distance(perm_Z, perm_Z, norm)
        if kernel == "deep":
            fz = f(perm_Z)[0]
            pairwise_matrix_f = torch_distance(fz, fz, norm)
            K_q = gaussian_kernel(pairwise_matrix, b_q)
            K_phi = gaussian_kernel(pairwise_matrix_f, b_phi)
            K_perm = (1-c_epsilon) * K_phi * K_q + c_epsilon * K_q
        else:
            K_perm = kernel_matrix(pairwise_matrix, kernel, bandwidth)
        perm_mmd = mmd_u(K_perm, n, m)
        if perm_mmd >= observed_mmd:
            count += 1

    p_value = count / num_permutations
    return p_value, observed_mmd

#########################################################################################################


def compute_bandwidths(dist, n_bandwidth):
    median = torch.median(dist)
    dist = dist + (dist == 0) * median

    dd = torch.sort(dist)[0]
    n = len(dd)
    idx_5 = int(torch.floor(torch.tensor(n * 0.05)).item())
    idx_95 = int(torch.floor(torch.tensor(n * 0.95)).item())
    lambda_min = dd[idx_5] / 2
    lambda_max = dd[idx_95] * 2
    bandwidths = torch.linspace(
        lambda_min, lambda_max, n_bandwidth).to(dist.device)
    return bandwidths


################################### Helper functions for MMD-Agg test ###################################

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

def permute_data(m, n, seed, B1, B2, device, is_jax=False):

    if is_jax:
        # Set random seed
        key = jrandom.PRNGKey(seed)
        key, subkey = jrandom.split(key)
        base_array = jnp.arange(m + n)  # Create once
        repeated_array = jnp.tile(base_array, (B1 + B2 + 1, 1))
        idx = jrandom.permutation(
            subkey, repeated_array, axis=1, independent=True)

        idx = torch.from_numpy(np.array(idx)).long().to(device)
        # print(idx.size())
    else:
        base_array = torch.arange(m + n, device=device)
        repeated_array = base_array.repeat(B1 + B2 + 1, 1)
        # Generate independent permutations for each row
        idx = torch.stack([torch.randperm(m + n, device=device)
                        for _ in range(B1 + B2 + 1)])
        # print(idx.size())

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


def wild_bootstrap(m, n, seed, B1, B2, device, is_jax=False):
    
    if is_jax:
        # Generate random keys
        key = jrandom.PRNGKey(seed)
        key, subkey = jrandom.split(key) # (B1+B2+1, n) Rademacher
        R = jrandom.choice(subkey, jnp.array(
            [-1.0, 1.0]), shape=(B1 + B2 + 1, n))
        R = R.at[B1].set(jnp.ones(n))
        R = R.transpose()
        R = jnp.concatenate((R, -R))  # (2n, B1+B2+1)
        R = torch.from_numpy(np.array(R)).long().to(device)
    else:
        R = torch.randint(0, 2, (B1 + B2 + 1, n), device=device) * 2.0 - 1.0
        # Set middle row (index B1) to ones
        R[B1] = torch.ones(n, device=device)

        # Transpose and concatenate with negation
        R = R.t()  # transpose
        R = torch.cat((R, -R), dim=0)  # concatenate with negation

    return R.float()


def generate_mmd_matrix(X, Y, kernel_bandwidths_l_list, n_bandwidth, B1, B2, params, is_permuted):
    N = n_bandwidth * len(kernel_bandwidths_l_list)
    M = torch.zeros((N, B1 + B2 + 1))
    m, n = len(X), len(Y)
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

            if is_permuted:
                [V11, V10, V01] = params
                # Compute MMD permuted values
                M[n_bandwidth * j + i] = (
                    torch.sum(V10 * (K @ V10), dim=0) * (n - m + 1) / (m * n * (m - 1)) +
                    torch.sum(V01 * (K @ V01), dim=0) * (m - n + 1) / (m * n * (n - 1)) +
                    torch.sum(V11 * (K @ V11), dim=0) / (m * n)
                )
            else:
                [R] = params
                # Get diagonal indices for n x n matrix
                diag_indices = torch.arange(n, device=K.device)

                # Set diagonal elements of all four submatrices to zero
                K[diag_indices, diag_indices] = 0
                K[diag_indices, diag_indices + n] = 0
                K[diag_indices + n, diag_indices] = 0
                K[diag_indices + n, diag_indices + n] = 0

                # Compute MMD bootstrapped values
                M[n_bandwidth * j + i] = torch.sum(R * (K @ R), dim=0) / (n * (n - 1))

    return M

def compute_u(M, kernel_bandwidths_l_list, n_bandwidth, B1, B2, B3, weights, alpha):
    
    M1_sorted = torch.sort(M[:, :B1+1], dim=1)[0]
    M2 = M[:, B1+1:]
    N = n_bandwidth * len(kernel_bandwidths_l_list)
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
    return u, quantiles
#########################################################################################################

################################### Helper functions for C2ST test ######################################

def split_datasets(X, Y, train_ratio=0.5, seed=None):
    """
    Split two groups of tensors into train and test sets while maintaining their structure.
    
    Args:
        X (torch.Tensor): First group tensor of shape (n, ...)
        Y (torch.Tensor): Second group tensor of shape (n, ...)
        train_ratio (float): Proportion of data to use for training (default: 0.5)
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, Y_train, Y_test)
    """
    # Input validation
    assert 0 < train_ratio < 1, "train_ratio must be between 0 and 1"

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Get number of samples
    n_samples = X.size(0)
    device = X.device

    n_train = int(n_samples * train_ratio)
    n_test = n_samples - n_train

    if n_train < 2 or n_test < 2:
        raise ValueError(
            f"Split would result in train_size={n_train}, test_size={n_test}. "
            f"Both must be >= 2. Either adjust train_ratio={train_ratio} or "
            f"provide more samples (current n_samples={n_samples})"
        )

    # Generate random permutation
    perm = torch.randperm(n_samples, device=device)

    # Split indices
    train_indices = perm[:n_train]
    test_indices = perm[n_train:]

    # Split the datasets
    X_train = X[train_indices]
    X_test = X[test_indices]
    Y_train = Y[train_indices]
    Y_test = Y[test_indices]

    return X_train, X_test, Y_train, Y_test


def train_clf(X, Y, val_ratio, batch_size, max_epoch, lr, patience, model, is_log, default_model):
    device = X.device
    n_samples = X.size(0)

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=patience, min_lr=1e-6
    )

    Z = torch.cat([X, Y], dim=0)
    y = torch.cat([torch.zeros(n_samples), torch.ones(n_samples)]).to(
        device, torch.long)
    
    # Z_dummy = Z.clone()

    Z_tr, Z_val, y_tr, y_val = split_datasets(Z, y, train_ratio=1-val_ratio)
    train_dataset = TensorDataset(
        Z_tr,
        y_tr
    )
    # print(Z_tr.size(), y_tr)
    # val_dataset = TensorDataset(
    #     Z_val.to(device),
    #     y_val.to(device)
    # )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    lambda_sparse = 1e-3

    for epoch in range(max_epoch):
        # Training phase
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # print(next(model.parameters()).is_cuda)
            # print(X_batch.is_cuda)
            if default_model:
                outputs, M_loss = model(X_batch)
            else:
                outputs = model(X_batch)
                M_loss = 0
                # loss = criterion(outputs, y_batch)
            loss = criterion(outputs, y_batch) + lambda_sparse * M_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if is_log and (epoch+1) % patience == 0:
            if default_model:
                print(f"Epoch {epoch+1}' training loss: ", criterion(model(Z_tr)[0], y_tr))
            else:
                print(f"Epoch {epoch+1}' training loss: ", criterion(model(Z_tr), y_tr))

        # Validation phase
        model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            if default_model:
                outputs = model(Z_val)[0]
            else:
                outputs = model(Z_val)
            loss = criterion(outputs, y_val)
            val_losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

        # Calculate metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_accuracy = correct / total
        
        scheduler.step(avg_val_loss)


        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if is_log:
                print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def plot_training_history(history, figsize=(12, 5)):
    """
    Plot training history showing loss curves and accuracy.
    
    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', and 'val_accuracy'
        figsize (tuple): Figure size (width, height)
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get number of epochs
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot training and validation loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation accuracy
    ax2.plot(epochs, history['val_accuracy'],
             'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Find best performance
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    best_accuracy = max(history['val_accuracy'])

    # Add text box with best performance
    textstr = f'Best Performance:\nEpoch: {best_epoch}\nVal Loss: {best_val_loss:.4f}\nVal Acc: {best_accuracy:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.05, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', bbox=props)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    return fig

def test_clf(X, Y, n_perm, model, is_label, default_model):
    device = X.device
    n_samples = X.size(0)

    Z = torch.cat([X, Y], dim=0)
    n_total = Z.size(0)

    if default_model:
        outputs = model(Z)[0]
    else:
        outputs = model(Z)
    outputs = torch.nn.Softmax(dim=1)(outputs)
    if is_label:
        outputs = outputs.max(1, keepdim=True)[1]

    outputs = outputs.float()
    mmd_value = torch.abs(torch.mean(
        outputs[:n_samples, 0]) - torch.mean(outputs[n_samples:, 0]))
    # print(torch.mean(outputs[:n_samples, 0]), torch.mean(outputs[n_samples:, 0]))
    mmd_values = torch.zeros(n_perm, device=device)
    for r in range(n_perm):
        ind = torch.randperm(n_total)

        ind_X = ind[:n_samples]
        ind_Y = ind[n_samples:]

        mmd_values[r] = torch.abs(torch.mean(
            outputs[ind_X, 0]) - torch.mean(outputs[ind_Y, 0]))

    p_value = (mmd_values >= mmd_value).float().mean().item()
    return p_value, mmd_value

#########################################################################################################

################################### Helper functions for MMD-Deep #######################################

def stack_representation(model):
    def feature_extractor(x):
        steps_output, M_Loss = model(x)
        return torch.stack(steps_output, dim=0).mean(dim=0), M_Loss
    return feature_extractor

def deep_objective(fz, Z, epsilon, b_q, b_phi, n_samples):

    # Compute pairwise distances
    pairwise_matrix = torch_distance(Z, Z, norm=2, is_squared=True)
    pairwise_matrix_f = torch_distance(fz, fz, norm=2, is_squared=True)

    # Compute kernel matrices
    K_q = gaussian_kernel(pairwise_matrix, b_q, is_squared=True)
    K_phi = gaussian_kernel(pairwise_matrix_f, b_phi, is_squared=True)

    # Compute deep kernel matrix
    K_deep = (1-epsilon) * K_phi * K_q + epsilon * K_q

    tmp = mmd_u(K_deep, n_samples, n_samples, is_var=True)
    mmd_value, mmd_std = tmp[0]+1e-8, torch.sqrt(tmp[1]+1e-8)

    if mmd_std.item() == 0:
        print("Warning: Zero variance in MMD estimate")

    stats = torch.div(-1*mmd_value, mmd_std)

    return stats, mmd_value

def get_median_bandwidth(X, Y, kernel):
    Z = torch.cat([X, Y], dim=0)
    norm = get_norms(kernel)[0]
    pairwise_matrix = torch_distance(Z, Z, norm)
    median = torch.median(pairwise_matrix)
    return median

def train_deep(X, Y, val_ratio, batch_size, max_epoch, lr, patience, model, is_log, default_model, data_type):
    
    device = X.device
    dtype = torch.float32
    n_samples = X.size(0)
    
    if default_model:
        if data_type == 'tabular':
            f = stack_representation(model)
        else:
            f = model
        fx, fy = f(X)[0], f(Y)[0]
    else:
        f = model
        fx, fy = f(X), f(Y)

    b_q = get_median_bandwidth(X, Y, 'gaussian').clone()
    b_phi = get_median_bandwidth(fx, fy, 'gaussian').clone()
    
    c_epsilon = torch.tensor(1.0).to(device)
    b_q = torch.nn.Parameter(b_q)
    b_phi = torch.nn.Parameter(b_phi)
    c_epsilon.requires_grad = True
    b_q.requires_grad = True
    b_phi.requires_grad = True
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(list(model.parameters()) + [c_epsilon, b_q, b_phi], lr=lr)
    optimizer = torch.optim.Adam(list(model.parameters()) + [c_epsilon, b_q, b_phi], lr=lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=patience, min_lr=1e-6
    )

    X_tr, X_val, Y_tr, Y_val = split_datasets(X, Y, train_ratio=1-val_ratio)
    train_dataset = TensorDataset(
        X_tr, 
        Y_tr
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    Z_tr = torch.cat([X_tr, Y_tr], dim=0)
    Z_val = torch.cat([X_val, Y_val], dim=0)
    n_samples_tr = len(X_tr)
    n_samples_val = len(X_val)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    lambda_sparse = 1e-3

    for epoch in range(max_epoch):
        
        # Training phase
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            Z_batch = torch.cat([X_batch, Y_batch], dim=0)
            optimizer.zero_grad()
            epsilon = torch.sigmoid(c_epsilon)
            if default_model:
                fz, M_loss = f(Z_batch)
            else:
                fz = f(Z_batch)
                M_loss = 0
            # print(fz)
            n_samples = X_batch.size(0)
            loss = deep_objective(fz, Z_batch, epsilon, b_q, b_phi, n_samples)[0] + lambda_sparse * M_loss
            # print("Loss:", loss)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        if is_log and (epoch+1) % patience == 0:
            if default_model:
                print(f"Epoch {epoch+1}' J stats: ",
                      -1 * deep_objective(f(Z_tr)[0], Z_tr, epsilon, b_q, b_phi, n_samples_tr)[0].item())
            else:
                print(f"Epoch {epoch+1}' J stats: ",
                      -1 * deep_objective(f(Z_tr), Z_tr, epsilon, b_q, b_phi, n_samples_tr)[0].item())
        
        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            epsilon = torch.sigmoid(c_epsilon)
            # f = stack_representation(model)
            if default_model:
                fz_val = f(Z_val)[0]
            else:
                fz_val = f(Z_val)
            # print(fz_val)
            loss, mmd_value = deep_objective(
                fz_val, Z_val, epsilon, b_q, b_phi, n_samples_val)
            val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_accuracy = mmd_value.item()

        scheduler.step(avg_val_loss)

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if is_log:
                print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, [c_epsilon, b_q, b_phi]
