"""
Image datasets for two-sample testing.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Optional

from .base import ImageTSTDataset

try:
    import torchattacks
    TORCHATTACKS_AVAILABLE = True
except ImportError:
    TORCHATTACKS_AVAILABLE = False
    print("Warning: torchattacks not available. Install with: pip install torchattacks")


class CIFAR10Adversarial(ImageTSTDataset):
    """
    CIFAR-10 original vs adversarial examples using torchattacks.
    """

    def __init__(
        self,
        root: str = './data',
        N: int = 1000,
        M: int = 1000,
        download: bool = True,
        attack_method: str = 'FGSM',
        epsilon: float = 8 / 255,
        class_subset: Optional[list] = None,
        pretrained_model: str = 'resnet18',
        **kwargs
    ):
        if not TORCHATTACKS_AVAILABLE:
            print("Warning: torchattacks not available. Using simple noise perturbation as fallback.")
        self.attack_method = attack_method
        self.epsilon = epsilon
        self.class_subset = class_subset
        self.pretrained_model = pretrained_model
        super().__init__(root, N, M, download, **kwargs)

    def _download(self):
        torchvision.datasets.CIFAR10(root=self.root, train=True, download=True)

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root, 'cifar-10-batches-py'))

    def _get_attack_model(self):
        if self.pretrained_model == 'resnet18':
            model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
        else:
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(32, 10)
            )
        model.eval()
        return model

    def _generate_adversarial(self, images, labels, model):
        if not TORCHATTACKS_AVAILABLE:
            noise = torch.randn_like(images) * self.epsilon
            return torch.clamp(images + noise, 0, 1)

        device = next(model.parameters()).device
        images, labels = images.to(device), labels.to(device)

        if self.attack_method == 'FGSM':
            attack = torchattacks.FGSM(model, eps=self.epsilon)
        elif self.attack_method == 'PGD':
            attack = torchattacks.PGD(model, eps=self.epsilon, alpha=2 / 255, steps=4)
        elif self.attack_method == 'CW':
            attack = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
        elif self.attack_method == 'DeepFool':
            attack = torchattacks.DeepFool(model, steps=10, overshoot=0.02)
        else:
            attack = torchattacks.FGSM(model, eps=self.epsilon)

        adv_images = attack(images, labels)
        return adv_images.cpu()

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(root=self.root, train=True, transform=transform)

        if self.class_subset:
            indices = [i for i, (_, label) in enumerate(dataset) if label in self.class_subset]
            dataset = torch.utils.data.Subset(dataset, indices)

        model = self._get_attack_model()

        np.random.seed(self.seed)
        total_samples = min(self.N + self.M, len(dataset))
        indices = np.random.choice(len(dataset), total_samples, replace=False)

        original_images = []
        labels = []
        for i in indices[:self.N]:
            img, label = dataset[i]
            original_images.append(img)
            labels.append(label)

        adversarial_images = []
        batch_size = 32
        for i in range(0, len(original_images), batch_size):
            batch_imgs = torch.stack(original_images[i:i + batch_size])
            batch_labels = torch.tensor(labels[i:i + batch_size])
            adv_batch = self._generate_adversarial(batch_imgs, batch_labels, model)
            adversarial_images.extend([adv_batch[j] for j in range(len(adv_batch))])

        X = torch.stack(original_images[:self.N])
        Y = torch.stack(adversarial_images[:self.M])

        if self.t1_check:
            print("Type-I error check: Drawing both X and Y from adversarial distribution")
            total_needed = self.N + self.M
            if len(adversarial_images) < total_needed:
                additional_needed = total_needed - len(adversarial_images)
                for i in range(additional_needed):
                    idx = i % len(original_images)
                    batch_imgs = torch.stack([original_images[idx]])
                    batch_labels = torch.tensor([labels[idx]])
                    adv_batch = self._generate_adversarial(batch_imgs, batch_labels, model)
                    adversarial_images.append(adv_batch[0])
            X = torch.stack(adversarial_images[:self.N])
            Y = torch.stack(adversarial_images[self.N:self.N + self.M])

        return X, Y


class CIFAR10_1(ImageTSTDataset):
    """
    CIFAR-10 original test set vs CIFAR-10.1 v4 dataset.
    """

    _PACKAGE_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

    def __init__(
        self,
        root: str = None,
        N: int = 1000,
        M: int = 1000,
        download: bool = True,
        cifar101_file: str = 'cifar10.1_v4_data.npy',
        **kwargs
    ):
        if root is None:
            root = os.path.normpath(self._PACKAGE_DATA_DIR)
        self.cifar101_file = cifar101_file
        super().__init__(root, N, M, download, **kwargs)

    def _download(self):
        torchvision.datasets.CIFAR10(root=self.root, train=False, download=True)

    def _check_exists(self) -> bool:
        cifar_path = os.path.join(self.root, 'cifar-10-batches-py')
        cifar101_path = os.path.join(self.root, self.cifar101_file)
        return os.path.exists(cifar_path) and os.path.exists(cifar101_path)

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        transform = transforms.Compose([transforms.ToTensor()])
        cifar10_test = torchvision.datasets.CIFAR10(root=self.root, train=False, transform=transform)

        cifar101_path = os.path.join(self.root, self.cifar101_file)
        cifar101_data = np.load(cifar101_path)
        cifar101_tensor = torch.from_numpy(cifar101_data).float()
        cifar101_tensor = (cifar101_tensor.permute(0, 3, 1, 2) / 255.0).contiguous()

        np.random.seed(self.seed)
        cifar10_indices = np.random.choice(len(cifar10_test), min(self.N, len(cifar10_test)), replace=False)

        original_images = []
        for i in cifar10_indices:
            img, _ = cifar10_test[i]
            original_images.append(img)
        X = torch.stack(original_images[:self.N])

        cifar101_indices = np.random.choice(len(cifar101_tensor), min(self.M, len(cifar101_tensor)), replace=False)
        Y = cifar101_tensor[cifar101_indices[:self.M]]

        if self.t1_check:
            print("Type-I error check: Drawing both X and Y from CIFAR-10.1 distribution")
            total_needed = self.N + self.M
            if len(cifar101_tensor) < total_needed:
                indices = np.random.choice(len(cifar101_tensor), total_needed, replace=True)
            else:
                indices = np.random.choice(len(cifar101_tensor), total_needed, replace=False)
            X = cifar101_tensor[indices[:self.N]]
            Y = cifar101_tensor[indices[self.N:self.N + self.M]]

        return X, Y


__all__ = [
    'CIFAR10Adversarial',
    'CIFAR10_1',
]
