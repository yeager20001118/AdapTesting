# TST (Two-Sample Testing Toolbox)

A Python package for state-of-the-art two-sample testing methods with GPU acceleration support.

## Features

- Median Heuristics
- MMD-FUSE
- MMD-Agg
- MMD-Deep
- C2ST

## Installation

```bash
pip install tst
```

## Performance Analysis

| Method       | Median | MMD-FUSE | MMD-Agg | MMD-Deep | C2ST-S | C2ST-MMD |
|--------------|--------|----------|---------|----------|--------|----------|
| Test Power   | 0.69   |          | 0.72    |          |        |          |
| Type-I Error | 0.01   |          | 0.04    |          |        |          |
| Runtime (s)  | 4.49   |          | 3.48    |          |        |          |

# Notes:

- Test Power: Ability to correctly reject H0 when false (higher is better)
- Type-I Error: Rate of falsely rejecting H0 when true (should be ≤ α)
- Running Time: Computational time in seconds

# Method Descriptions:

1. **Median-Heuristic**: Classic MMD test with median-based kernel bandwidth
2. **MMD-Fuse**: MMD with multiple kernel bandwidths
3. **MMD-Agg**: Aggregated MMD test across different kernels
4. **MMD-Deep**: Deep kernel MMD with learned features
5. **C2ST-Sign**: Classifier two-sample test with sign test
6. **C2ST-MMD**: Classifier two-sample test with MMD statistic
