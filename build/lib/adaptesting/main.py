import gc
import torch
from .utils import *
from .tst_method import median, fuse, agg, deep, clf
from .idt_method import hsicagg, rdc, fsic

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
        output_round=4,
        is_jax=False,
        is_permuted=False,
        is_log = False,
        is_history = False,
        is_label = False,
        is_report = True,
        is_balanced = True):
    
    if data_type == "text":
        if isinstance(X, list) or isinstance(Y, list):
            # Make sure the X and Y are both in the form of [str1, str2, ...]
            X = sentences_to_embeddings(X, device)
            Y = sentences_to_embeddings(Y, device)
            data_type = "tabular"
        else:
            print("Assuming the input text data are already in the form of embeddings.")
            data_type = "tabular"
    
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

    X, Y = check_shapes_and_adjust(X, Y, is_balanced, is_report)
    # n_sample = len(X)  # represent the size of one sample
    if model is None and method in ['deep', 'clf']:
        default_model = True
        if data_type == 'tabular':
            input_dim = X.size(1)
            model = MitraTabularModel(
                input_dim=input_dim,
                output_dim=2,  # binary classification
                device=device
            ).to(device)
        elif data_type == 'image':
            n_channels = X.size(1)
            image_size = X.size(2) # assuming square images where size(2) = size(3)
            model = DefaultImageModel(
                n_channels=n_channels, image_size=image_size, weights='DEFAULT')
            
            # Add classification layer if classifier method
            if method != "deep":
                model.resnet.fc = nn.Sequential(
                    model.resnet.fc,
                    nn.ReLU(),
                    nn.Linear(100, 2)
                )
            model = model.to(device)
    else:
        default_model = False

    # Unsupervised TST
    if method == 'median':
        # perform median heuristic mmd test
        p_value, mmd_value = median(X, Y, n_perm, kernel, seed)
    elif method == 'fuse':
        # perform a MMD-FUSE test
        p_value, mmd_value = fuse(X, Y, n_perm, kernel, n_bandwidth, seed, is_jax)
    elif method == 'agg':
        # perform a MMD-Agg test
        p_value, mmd_value = agg(
            X, Y, alpha, n_perm, kernel, n_bandwidth, seed, is_jax, is_permuted)
    # Supervised TST
    elif method == 'deep':
        # perform a MMD-Deep test
        p_value, mmd_value = deep(
            X, Y, n_perm, model, train_ratio, patience, is_log, is_history, default_model, data_type)
    elif method == 'clf':
        # perform a classifier two-sample test
        p_value, mmd_value = clf(
            X, Y, n_perm, model, train_ratio, patience, is_log, is_history, is_label, default_model)
    # Semi-supervised TST tbc...
    else:
        raise ValueError("Unsupported two-sample testing type")

    if p_value <= alpha:
        h = 1
        if is_report:
            print(f"Reject the null hypothesis with p-value: {p_value:.{output_round}f}, "
                f"the test statistics for {method} is {mmd_value:.{output_round}f}.")
    else:
        h = 0
        if is_report:
            print(f"Fail to reject the null hypothesis with p-value: {p_value:.{output_round}f}, "
                f"the test statistics for {method} is {mmd_value:.{output_round}f}.")

    # Clean up GPU memory
    if torch.cuda.is_available() and str(device) != "cpu":
        del X, Y
        if model is not None and default_model:
            del model
        gc.collect()
        torch.cuda.empty_cache()

    return h, mmd_value, p_value


def idt(
    X,
    Y,
    method='rdc',
    alpha=0.05,
    device="cpu",
    dtype=torch.float32,
    n_perm=200,
    seed=0,
    R=200,
    hsic_collection_parameters=(2, -2, 2),
    B3=50,
    output_round=4,
    is_report=True):

    # Check the X, Y are torch tensors
    if not isinstance(X, torch.Tensor):
        raise ValueError("X must be a torch tensor")
    if not isinstance(Y, torch.Tensor):
        raise ValueError("Y must be a torch tensor")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("For independence testing, X and Y must have the same number of samples (paired data)")

    X = X.to(device=device, dtype=dtype)
    Y = Y.to(device=device, dtype=dtype)

    if method == 'hsicagg':
        # HSICAggInc requires data with an even number of samples
        if X.shape[0] % 2 != 0:
            raise ValueError("For 'hsicagg', the number of samples must be even (required for the incomplete U-statistic split)")

        p_value, stat_value = hsicagg(X, Y, alpha, n_perm, seed, hsic_collection_parameters, R, B3)
    elif method == 'rdc':
        p_value, stat_value = rdc(X, Y, n_perm=n_perm, seed=seed)
    elif method == 'fsic':
        p_value, stat_value = fsic(X, Y, n_perm=n_perm, seed=seed)
    else:
        raise ValueError("Unsupported independence testing method")

    if p_value <= alpha:
        h = 1
        if is_report:
            print(f"Reject the null hypothesis with p-value: {p_value:.{output_round}f}, "
                f"the test statistics for {method} is {stat_value:.{output_round}f}.")
    else:
        h = 0
        if is_report:
            print(f"Fail to reject the null hypothesis with p-value: {p_value:.{output_round}f}, "
                f"the test statistics for {method} is {stat_value:.{output_round}f}.")

    # Clean up GPU memory
    if torch.cuda.is_available() and str(device) != "cpu":
        del X, Y
        gc.collect()
        torch.cuda.empty_cache()

    return h, stat_value, p_value

