# Datasets Module

This module contains dataset abstractions and ready-to-use dataset classes for statistical testing tasks in `adaptesting`.

## Design

The dataset layer is organized by testing task first:

- `datasets.tst`: two-sample testing datasets
- `datasets.idt`: independence testing datasets

The public API is still exposed through `adaptesting.datasets`, so external users can keep using a flat import style.

## Public Usage

```python
from adaptesting import datasets
```

Available examples currently include:

### Two-sample testing datasets

- `datasets.HDGM`
- `datasets.BLOB`
- `datasets.HiggsBoson`
- `datasets.CIFAR10Adversarial`
- `datasets.CIFAR10_1`
- `datasets.HumanAIDetection`
- `datasets.HC3`

### Independence testing datasets

- `datasets.SyntheticJointSplit`
- `datasets.MNISTLabelCorruption`

## Examples

### Two-sample testing dataset

```python
from adaptesting import datasets

dataset = datasets.HDGM(N=500, M=500, level="hard")
X, Y = dataset()
```

### Independence testing dataset

```python
from adaptesting import datasets

dataset = datasets.SyntheticJointSplit(N=500, d=3, dx=1)
X, Y = dataset()
```

### MNIST dependence dataset

```python
from adaptesting import datasets

dataset = datasets.MNISTLabelCorruption(
    N=1000,
    corrupt_proportion=0.2,
)
X, Y = dataset()
```

In this dataset:

- `X` has shape `(N, 1, 28, 28)`
- `Y` has shape `(N, 1)`

## Reproducibility

All dataset classes follow the same seed handling style:

- `seed` is stored on the dataset object
- NumPy and PyTorch seeds are set during initialization
- synthetic and sampled datasets should be reproducible given the same constructor arguments

## Type-I Error Support

Many datasets support `t1_check=True`.

- For two-sample datasets, this means drawing both samples from the same distribution
- For independence datasets, this means generating `X` and `Y` under independence when the dataset naturally supports that option
