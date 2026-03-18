import sys
import types

import numpy as np
import torch

from adaptesting import datasets


def test_synthetic_joint_split_returns_tensors_with_aligned_counts():
    dataset = datasets.SyntheticJointSplit(N=32, d=4, dx=1, download=False, seed=3)
    X, Y = dataset()

    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)
    assert X.shape == (32, 1)
    assert Y.shape == (32, 3)
    assert X.dtype == torch.float32
    assert Y.dtype == torch.float32


def test_synthetic_joint_split_t1_check_keeps_sample_counts():
    dataset = datasets.SyntheticJointSplit(N=24, d=3, dx=1, download=False, seed=5, t1_check=True)
    X, Y = dataset()

    assert X.shape[0] == 24
    assert Y.shape[0] == 24
    assert Y.shape[1] == 2


class FakeMNIST:
    def __init__(self, root, train=True, download=False):
        self.samples = []
        for i in range(40):
            image = torch.full((1, 28, 28), float(i), dtype=torch.float32) / 255.0
            label = i % 10
            self.samples.append((image, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def install_fake_torchvision(monkeypatch):
    fake_torchvision = types.SimpleNamespace(
        datasets=types.SimpleNamespace(MNIST=FakeMNIST)
    )
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)


def test_mnist_label_corruption_idt_zero_corruption(monkeypatch):
    install_fake_torchvision(monkeypatch)
    seed = 0
    dataset = datasets.MNISTLabelCorruption(N=10, download=False, corrupt_proportion=0.0, seed=seed)
    X, Y = dataset()

    indices = np.random.RandomState(seed).choice(40, 10, replace=False)
    expected_labels = torch.tensor([[i % 10] for i in indices], dtype=torch.float32)
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)
    assert X.shape == (10, 1, 28, 28)
    assert Y.shape == (10, 1)
    assert torch.equal(Y, expected_labels)


def test_mnist_label_corruption_idt_full_corruption(monkeypatch):
    install_fake_torchvision(monkeypatch)
    seed = 0
    dataset = datasets.MNISTLabelCorruption(N=10, download=False, corrupt_proportion=1.0, seed=seed)
    _, Y = dataset()

    indices = np.random.RandomState(seed).choice(40, 10, replace=False)
    original_labels = torch.tensor([[i % 10] for i in indices], dtype=torch.float32)
    assert not torch.equal(Y, original_labels)
    assert Y.shape == (10, 1)
