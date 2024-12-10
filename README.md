# AdapTesting (Data Adaptive Hypothesis Testing Toolbox)

A Python package for state-of-the-art data adaptive hypothesis testing methods with GPU acceleration support. Currently only support for two-sample testing, independence testing will be released later.

## Methods for TST

- Median Heuristics
- MMD-FUSE
- MMD-Agg
- MMD-Deep
- C2ST-MMD

## Installation and Usage

### Installation

```bash
pip install adaptesting
```

### Example usage

```Python
from adaptesting import tst # Load the main library to conduct tst
# Generate data as example, make sure the input data should be Pytorch Tensor
import torch
import random
from torch.distributions import MultivariateNormal
torch.manual_seed(0)
random.seed(0)
mean = torch.tensor([0.5, 0.5])
cov1 = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
cov2 = torch.tensor([[1.0, 0], [0, 1.0]])
mvn1 = MultivariateNormal(mean, cov1)
mvn2 = MultivariateNormal(mean, cov2)
counter = 0
n_trial = 100
n_samples = 250
# Conduct Experiments for n_trial times, 
# remove the for loop if only want to get a result of reject or not
for _ in range(n_trial):

    # Uncomment Z2 with same distribution to test Type-I error
    Z1 = mvn1.sample((1000,))
    Z2 = mvn2.sample((1000,))  # Test power
    # Z2 = mvn1.sample((1000,))  # Type-I error

    # Create a list of indices from 0 to 1000
    indices = list(range(1000))

    # Shuffle the indices
    random.shuffle(indices)

    # Select the n_samples for X
    X_indices = indices[:n_samples]

    # Select the n_samples for Y
    # Y_indices = indices[:n_samples]

    # Sample X and Y from Z using the selected indices, 
    # X.size() = (n_samples, 2), Y.size() = (n_samples, 2)
    X = Z1[X_indices]
    Y = Z2[X_indices]

    # Five kinds of SOTA TST methods to choose
    h, _ = tst(X, Y, device = "cuda") # default method is median heuristic
    # h, _ = tst(X, Y, device="cuda", method="fuse", kernel="laplace_gaussian", n_perm=2000)
    # h, _ = tst(X, Y, device="cuda", method="agg", n_perm=3000)
    # h, _ = tst(X, Y, device="cuda", method="clf", data_type="tabular", patience=150, n_perm=200)
    # h, _ = tst(X, Y, device="cuda", method="deep", data_type="tabular", patience=150, n_perm=200)
    counter += h

print(f"Power: {counter}/{n_trial}")
```

## Performance Display

| Method       | Median | MMD-FUSE | MMD-Agg | MMD-Deep | C2ST-MMD |
| ------------ | ------ | -------- | ------- | -------- | -------- |
| Test Power   | 0.69   | 0.56     | 0.72    | 0.72     | 0.71     |
| Type-I Error | 0.01   | 0.03     | 0.04    | 0.05     | 0.06     |
| Runtime (s)  | 4.49   | 10.14    | 3.48    | 486.81   | 570.94   |

# Notes:

- Test Power: Ability to correctly reject H0 when false (higher is better)
- Type-I Error: Rate of falsely rejecting H0 when true (should be ≤ α)
- Running Time: Computational time in seconds

# Method Descriptions:

1. **Median-Heuristic**: Classic MMD test with median-based kernel bandwidth
2. **MMD-Fuse**: MMD with multiple kernel bandwidths
3. **MMD-Agg**: Aggregated MMD test across different kernels
4. **MMD-Deep**: Deep kernel MMD with neural network learned features
5. **C2ST-MMD**: Classifier two-sample test with MMD statistic
