# AdapTesting (Data Adaptive Hypothesis Testing Toolbox)

A Python package for state-of-the-art data adaptive hypothesis testing methods with GPU acceleration support. Currently only support for two-sample testing, independence testing will be released later.

## Methods for TST - referenced paper

- Median Heuristics - [Large Sample Analysis of the Median Heuristic](https://arxiv.org/pdf/1707.07269)
- MMD-FUSE - [MMD-FUSE: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting](https://arxiv.org/pdf/2306.08777)
- MMD-Agg - [MMD Aggregated Two-Sample Test](https://arxiv.org/pdf/2110.15073)
- MMD-Deep - [Learning Deep Kernels for Non-Parametric Two-Sample Tests](https://arxiv.org/pdf/2002.09116)
- C2ST-MMD - [Revisiting Classifier Two-Sample Tests](https://arxiv.org/pdf/1610.06545)

### Installation and Usage

#### Installation

```bash
pip install adaptesting
```

#### Example usage of Two-sample Testing

The detailed demo examples (for tabular, image and text data) can be found in the [examples](./examples) directory.

```Python
from adaptesting import tst # Import main function 'tst' from package 'adaptesting'

# Example synthetic input data
import torch
from torch.distributions import MultivariateNormal
mean = torch.tensor([0.5, 0.5])
cov1, cov2 = torch.tensor([[1.0, 0.5], [0.5, 1.0]]), torch.tensor([[1.0, 0], [0, 1.0]])
mvn1, mvn2 = MultivariateNormal(mean, cov1), MultivariateNormal(mean, cov2)

# Replace X, Y to your own data, make sure its type as torch.Tensor
X, Y = mvn1.sample((1000,)), mvn2.sample((1000,)) 

# Five kinds of SOTA TST methods to choose：
h, mmd_value, p_value = tst(X, Y, device="cuda") # Default method using median heuristic

# Other available methods and their default arguments setting (uncomment to use):
# h, mmd_value, p_value = tst(X, Y, device="cuda", method="fuse", kernel="laplace_gaussian", n_perm=2000)
# h, mmd_value, p_value = tst(X, Y, device="cuda", method="agg", n_perm=3000)
# h, mmd_value, p_value = tst(X, Y, device="cuda", method="clf", data_type="tabular", patience=150, n_perm=200)
# h, mmd_value, p_value = tst(X, Y, device="cuda", method="deep", data_type="tabular", patience=150, n_perm=200)

"""
Output of tst: 
    (result of testing: 0 or 1, 
    mmd value of two samples, 
    p-value of testing)

If testing the two samples are from different distribution, the console will output 
    'Reject the null hypothesis with p-value: ..., the MMD value is ...'.
Otherwise,
    'Fail to reject the null hypothesis with p-value: ..., the MMD value is ...'.
"""
```

### Performance of TST methods

Performance evaluations and benchmarks across tabular, image, and text data can be found in the [examples](./examples) directory.

- Test Power: Ability to correctly reject H0 when false (higher is better)
- Type-I Error: Rate of falsely rejecting H0 when true (should be ≤ α, the significance level)
- Running Time: Computational time in seconds

<!-- ### Method Descriptions:

1. **Median-Heuristic**: Classic MMD test with median-based kernel bandwidth
2. **MMD-Fuse**: MMD with multiple kernel bandwidths
3. **MMD-Agg**: Aggregated MMD test across different kernels
4. **MMD-Deep**: Deep kernel MMD with neural network learned features
5. **C2ST-MMD**: Classifier two-sample test with MMD statistic -->
