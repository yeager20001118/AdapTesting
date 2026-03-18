import torch

from adaptesting import datasets
from adaptesting.datasets.tst.tabular import HDGM as TaskHDGM
from adaptesting.datasets.idt.tabular import SyntheticJointSplit as TaskSyntheticJointSplit


def test_legacy_and_task_first_tst_imports_match():
    legacy_dataset = datasets.HDGM(N=20, M=20, download=False, seed=1)
    task_dataset = TaskHDGM(N=20, M=20, download=False, seed=1)

    X_legacy, Y_legacy = legacy_dataset()
    X_task, Y_task = task_dataset()

    assert torch.equal(X_legacy, X_task)
    assert torch.equal(Y_legacy, Y_task)


def test_legacy_and_task_first_idt_imports_match():
    legacy_dataset = datasets.SyntheticJointSplit(N=20, d=3, dx=1, download=False, seed=2)
    task_dataset = TaskSyntheticJointSplit(N=20, d=3, dx=1, download=False, seed=2)

    X_legacy, Y_legacy = legacy_dataset()
    X_task, Y_task = task_dataset()

    assert torch.equal(X_legacy, X_task)
    assert torch.equal(Y_legacy, Y_task)
