import os
import importlib
import numpy as np
import torch
from typing import Tuple

from ..base import ImageIDTDataset


class MNISTLabelCorruption(ImageIDTDataset):
    """
    MNIST image-label dependence dataset with controllable label corruption.

    X is returned in standard image format (N, 1, 28, 28).
    Y is returned as a float tensor of shape (N, 1).
    """

    def __init__(
        self,
        root: str = './data',
        N: int = 1000,
        download: bool = True,
        train: bool = True,
        corrupt_proportion: float = 0.0,
        **kwargs
    ):
        self.train = train
        self.corrupt_proportion = corrupt_proportion
        super().__init__(root, N, download, **kwargs)

    def _get_torchvision(self):
        return importlib.import_module('torchvision')

    def _download(self):
        torchvision = self._get_torchvision()
        torchvision.datasets.MNIST(
            root=self.root,
            train=self.train,
            download=True
        )

    def _check_exists(self) -> bool:
        mnist_path = os.path.join(self.root, 'MNIST')
        return os.path.exists(mnist_path) or not self.download

    def _to_image_tensor(self, image):
        if isinstance(image, torch.Tensor):
            tensor = image.clone().detach()
        else:
            tensor = torch.tensor(np.array(image), dtype=torch.float32)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] != 1:
            tensor = tensor.permute(2, 0, 1)

        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        if tensor.max() > 1:
            tensor = tensor / 255.0
        return tensor

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= self.corrupt_proportion <= 1.0:
            raise ValueError("corrupt_proportion must be between 0 and 1")

        torchvision = self._get_torchvision()
        dataset = torchvision.datasets.MNIST(
            root=self.root,
            train=self.train,
            download=self.download
        )

        rs = np.random.RandomState(self.seed)
        replace = len(dataset) < self.N
        indices = rs.choice(len(dataset), self.N, replace=replace)

        images = []
        labels = []
        for idx in indices:
            image, label = dataset[idx]
            images.append(self._to_image_tensor(image))
            labels.append(int(label))

        X = torch.stack(images)
        Y = np.array(labels, dtype=np.int64)

        if self.t1_check:
            Y = rs.randint(0, 10, size=self.N)
        else:
            n_corrupt = int(round(self.corrupt_proportion * self.N))
            if n_corrupt > 0:
                corrupt_idx = rs.choice(self.N, n_corrupt, replace=False)
                new_labels = rs.randint(0, 9, size=n_corrupt)
                Y[corrupt_idx] = (Y[corrupt_idx] + 1 + new_labels) % 10

        Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None:
            Y = self.target_transform(Y)

        return X, Y

__all__ = [
    'MNISTLabelCorruption',
]
