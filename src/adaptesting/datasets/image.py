"""
Image datasets for two-sample testing using real datasets and established attacks.

These datasets use torchattacks, torchvision, and other established libraries
for authentic image data representing different conditions.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Optional, Any, Callable
from PIL import Image
import requests
from .base import ImageTSTDataset

try:
    import torchattacks
    TORCHATTACKS_AVAILABLE = True
except ImportError:
    TORCHATTACKS_AVAILABLE = False
    print("Warning: torchattacks not available. Install with: pip install torchattacks")

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: HuggingFace datasets not available. Install with: pip install datasets")


class CIFAR10Adversarial(ImageTSTDataset):
    """
    CIFAR-10 original vs adversarial examples using torchattacks.
    
    Uses CIFAR-10 dataset and generates adversarial examples using
    established attacks from the torchattacks library.
    """
    
    def __init__(
        self,
        root: str = './data',
        N: int = 1000,
        M: int = 1000,
        download: bool = True,
        attack_method: str = 'FGSM',
        epsilon: float = 8/255,
        class_subset: Optional[list] = None,
        pretrained_model: str = 'resnet18',
        **kwargs
    ):
        """
        Args:
            attack_method (str): Attack method from torchattacks ('FGSM', 'PGD', 'C&W', 'DeepFool')
            epsilon (float): Perturbation strength (for FGSM/PGD)
            class_subset (list): Subset of CIFAR-10 classes to use (0-9)
            pretrained_model (str): Model to use for generating adversarial examples
        """
        if not TORCHATTACKS_AVAILABLE:
            print("Warning: torchattacks not available. Using simple noise perturbation as fallback.")
        
        self.attack_method = attack_method
        self.epsilon = epsilon
        self.class_subset = class_subset
        self.pretrained_model = pretrained_model
        super().__init__(root, N, M, download, **kwargs)
    
    def _download(self):
        # Download CIFAR-10 dataset
        torchvision.datasets.CIFAR10(
            root=self.root, 
            train=True, 
            download=True
        )
    
    def _check_exists(self) -> bool:
        cifar_path = os.path.join(self.root, 'cifar-10-batches-py')
        return os.path.exists(cifar_path)
    
    def _get_attack_model(self):
        """Load a pretrained model for generating adversarial examples."""
        if self.pretrained_model == 'resnet18':
            model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            # Modify for CIFAR-10 (10 classes)
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
        else:
            # Default to simple CNN
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
        """Generate adversarial examples using torchattacks."""
        if not TORCHATTACKS_AVAILABLE:
            # Fallback: simple Gaussian noise
            noise = torch.randn_like(images) * self.epsilon
            return torch.clamp(images + noise, 0, 1)
        
        device = next(model.parameters()).device
        images, labels = images.to(device), labels.to(device)
        
        if self.attack_method == 'FGSM':
            attack = torchattacks.FGSM(model, eps=self.epsilon)
        elif self.attack_method == 'PGD':
            attack = torchattacks.PGD(model, eps=self.epsilon, alpha=2/255, steps=4)
        elif self.attack_method == 'CW':
            attack = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
        elif self.attack_method == 'DeepFool':
            attack = torchattacks.DeepFool(model, steps=10, overshoot=0.02)
        else:
            # Default to FGSM
            attack = torchattacks.FGSM(model, eps=self.epsilon)
        
        adv_images = attack(images, labels)
        return adv_images.cpu()
    
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root=self.root, 
            train=True, 
            transform=transform
        )
        
        # Filter by class subset if specified
        if self.class_subset:
            indices = [i for i, (_, label) in enumerate(dataset) 
                      if label in self.class_subset]
            dataset = torch.utils.data.Subset(dataset, indices)
        
        # Load model for adversarial generation
        model = self._get_attack_model()
        
        # Sample random indices
        np.random.seed(self.seed)
        total_samples = min(self.N + self.M, len(dataset))
        indices = np.random.choice(len(dataset), total_samples, replace=False)
        
        # Collect original images and labels
        original_images = []
        labels = []
        
        for i in indices[:self.N]:
            img, label = dataset[i]
            original_images.append(img)
            labels.append(label)
        
        # Generate adversarial images
        batch_size = 32  # Process in batches to avoid memory issues
        adversarial_images = []
        
        for i in range(0, len(original_images), batch_size):
            batch_imgs = torch.stack(original_images[i:i+batch_size])
            batch_labels = torch.tensor(labels[i:i+batch_size])
            
            adv_batch = self._generate_adversarial(batch_imgs, batch_labels, model)
            adversarial_images.extend([adv_batch[j] for j in range(len(adv_batch))])
        
        # Take required number of samples
        X = torch.stack(original_images[:self.N])
        Y = torch.stack(adversarial_images[:self.M])
        
        # Type-I error check: draw both samples from the same distribution
        if self.t1_check:
            print("Type-I error check: Drawing both X and Y from adversarial distribution")
            # Make sure we have enough adversarial samples for both X and Y
            total_needed = self.N + self.M
            if len(adversarial_images) < total_needed:
                # Generate more adversarial samples if needed
                additional_needed = total_needed - len(adversarial_images)
                for i in range(additional_needed):
                    idx = i % len(original_images)
                    batch_imgs = torch.stack([original_images[idx]])
                    batch_labels = torch.tensor([labels[idx]])
                    adv_batch = self._generate_adversarial(batch_imgs, batch_labels, model)
                    adversarial_images.append(adv_batch[0])
            
            X = torch.stack(adversarial_images[:self.N])
            Y = torch.stack(adversarial_images[self.N:self.N+self.M])
        
        return X, Y


class CIFAR10_1(ImageTSTDataset):
    """
    CIFAR-10 original test set vs CIFAR-10.1 v4 dataset.

    CIFAR-10.1 is a new test set for CIFAR-10 collected by Recht et al. (2018)
    to measure classifier accuracy under distribution shift.

    Reference: https://github.com/modestyachts/CIFAR-10.1
    """

    # Default data directory bundled with the package
    _PACKAGE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

    def __init__(
        self,
        root: str = None,
        N: int = 1000,
        M: int = 1000,
        download: bool = True,
        cifar101_file: str = 'cifar10.1_v4_data.npy',
        **kwargs
    ):
        """
        Args:
            root (str): Root directory for dataset storage. If None, uses the package's bundled data directory.
            N (int): Number of samples from CIFAR-10 original test set
            M (int): Number of samples from CIFAR-10.1 v4 dataset
            download (bool): If True, download CIFAR-10 if not found
            cifar101_file (str): Filename of the CIFAR-10.1 v4 data file
        """
        if root is None:
            root = self._PACKAGE_DATA_DIR
        self.cifar101_file = cifar101_file
        super().__init__(root, N, M, download, **kwargs)

    def _download(self):
        # Download CIFAR-10 dataset (test set)
        torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True
        )

    def _check_exists(self) -> bool:
        cifar_path = os.path.join(self.root, 'cifar-10-batches-py')
        cifar101_path = os.path.join(self.root, self.cifar101_file)
        return os.path.exists(cifar_path) and os.path.exists(cifar101_path)

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load CIFAR-10 original test set
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        cifar10_test = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            transform=transform
        )

        # Load CIFAR-10.1 v4 data
        cifar101_path = os.path.join(self.root, self.cifar101_file)
        cifar101_data = np.load(cifar101_path)

        # Convert CIFAR-10.1 data to torch tensor
        # Shape: (N, 32, 32, 3) -> (N, 3, 32, 32), normalized to [0, 1]
        cifar101_tensor = torch.from_numpy(cifar101_data).float()
        cifar101_tensor = (cifar101_tensor.permute(0, 3, 1, 2) / 255.0).contiguous()

        # Sample from CIFAR-10 original test set
        np.random.seed(self.seed)
        cifar10_indices = np.random.choice(
            len(cifar10_test),
            min(self.N, len(cifar10_test)),
            replace=False
        )

        original_images = []
        for i in cifar10_indices:
            img, _ = cifar10_test[i]
            original_images.append(img)

        X = torch.stack(original_images[:self.N])

        # Sample from CIFAR-10.1 v4 dataset
        cifar101_indices = np.random.choice(
            len(cifar101_tensor),
            min(self.M, len(cifar101_tensor)),
            replace=False
        )
        Y = cifar101_tensor[cifar101_indices[:self.M]]

        # Type-I error check: draw both samples from the same distribution
        if self.t1_check:
            print("Type-I error check: Drawing both X and Y from CIFAR-10.1 distribution")
            total_needed = self.N + self.M
            if len(cifar101_tensor) < total_needed:
                indices = np.random.choice(len(cifar101_tensor), total_needed, replace=True)
            else:
                indices = np.random.choice(len(cifar101_tensor), total_needed, replace=False)

            X = cifar101_tensor[indices[:self.N]]
            Y = cifar101_tensor[indices[self.N:self.N+self.M]]

        return X, Y


# class MNISTCorrupted(ImageTSTDataset):
#     """
#     MNIST original vs corrupted images using MNIST-C dataset.
    
#     Uses the MNIST-C (MNIST Corrupted) dataset which contains various
#     types of corruption applied to MNIST images.
#     """
    
#     def __init__(
#         self,
#         root: str = './data',
#         N: int = 1000,
#         M: int = 1000,
#         download: bool = True,
#         corruption_type: str = 'gaussian_noise',
#         severity: int = 3,
#         **kwargs
#     ):
#         """
#         Args:
#             corruption_type (str): Type of corruption ('gaussian_noise', 'shot_noise', 'impulse_noise',
#                                  'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 
#                                  'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression')
#             severity (int): Corruption severity level (1-5)
#         """
#         self.corruption_type = corruption_type
#         self.severity = severity
#         super().__init__(root, N, M, download, **kwargs)
    
#     def _download(self):
#         # Download original MNIST
#         torchvision.datasets.MNIST(
#             root=self.root, 
#             train=True, 
#             download=True
#         )
        
#         # For MNIST-C, we'll implement basic corruptions
#         # In practice, you might download the actual MNIST-C dataset
#         print(f"Using MNIST with {self.corruption_type} corruption (severity {self.severity})")
    
#     def _check_exists(self) -> bool:
#         mnist_path = os.path.join(self.root, 'MNIST')
#         return os.path.exists(mnist_path)
    
#     def _apply_corruption(self, img: torch.Tensor) -> torch.Tensor:
#         """Apply corruption to image based on corruption type and severity."""
#         severity_factor = self.severity / 5.0  # Normalize to 0-1
        
#         if self.corruption_type == 'gaussian_noise':
#             noise = torch.randn_like(img) * severity_factor * 0.5
#             return torch.clamp(img + noise, 0, 1)
        
#         elif self.corruption_type == 'shot_noise':
#             # Poisson noise
#             img_scaled = img * 255 * severity_factor
#             noisy = torch.poisson(img_scaled) / (255 * severity_factor)
#             return torch.clamp(noisy, 0, 1)
        
#         elif self.corruption_type == 'impulse_noise':
#             # Salt and pepper noise
#             mask = torch.rand_like(img) < severity_factor * 0.1
#             noise = torch.rand_like(img)
#             corrupted = torch.where(mask, noise, img)
#             return torch.clamp(corrupted, 0, 1)
        
#         elif self.corruption_type == 'defocus_blur':
#             # Simple blur using average pooling
#             kernel_size = 2 + int(severity_factor * 3)
#             if kernel_size % 2 == 0:
#                 kernel_size += 1
#             blurred = torch.nn.functional.avg_pool2d(
#                 img.unsqueeze(0), 
#                 kernel_size=kernel_size, 
#                 stride=1, 
#                 padding=kernel_size//2
#             ).squeeze(0)
#             return blurred
        
#         elif self.corruption_type == 'brightness':
#             brightness_factor = 1.0 + (severity_factor - 0.5) * 2.0
#             return torch.clamp(img * brightness_factor, 0, 1)
        
#         elif self.corruption_type == 'contrast':
#             mean = img.mean()
#             contrast_factor = 1.0 + severity_factor
#             return torch.clamp((img - mean) * contrast_factor + mean, 0, 1)
        
#         else:
#             # Default: Gaussian noise
#             noise = torch.randn_like(img) * severity_factor * 0.3
#             return torch.clamp(img + noise, 0, 1)
    
#     def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Load MNIST
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])
        
#         dataset = torchvision.datasets.MNIST(
#             root=self.root, 
#             train=True, 
#             transform=transform
#         )
        
#         # Sample random indices
#         np.random.seed(self.seed)
#         total_samples = min(self.N + self.M, len(dataset))
#         indices = np.random.choice(len(dataset), total_samples, replace=False)
        
#         # Get original images
#         original_images = []
#         for i in indices[:self.N]:
#             img, _ = dataset[i]
#             original_images.append(img)
        
#         # Generate corrupted images
#         corrupted_images = []
#         for i in indices[self.N:self.N+self.M]:
#             img, _ = dataset[i]
#             corrupted_img = self._apply_corruption(img)
#             corrupted_images.append(corrupted_img)
        
#         X = torch.stack(original_images)
#         Y = torch.stack(corrupted_images)
        
#         return X, Y


