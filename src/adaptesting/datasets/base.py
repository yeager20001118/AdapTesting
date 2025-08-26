"""
Base dataset class for two-sample testing datasets.
"""

import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import urllib.request
import zipfile
import tarfile
import gzip
import shutil


class TSTDataset(ABC):
    """
    Base class for all two-sample testing datasets.
    
    All datasets should return two torch tensors X and Y representing
    samples from two different distributions P and Q.
    """
    
    def __init__(
        self, 
        root: str = './data',
        N: int = 1000,
        M: int = 1000,
        download: bool = True,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        seed: int = 42,
        t1_check: bool = False
    ):
        """
        Args:
            root (str): Root directory where dataset will be stored
            N (int): Number of samples from distribution P (X)
            M (int): Number of samples from distribution Q (Y) 
            download (bool): If True, download dataset if not found
            transform: Optional transform to apply to the data
            target_transform: Optional transform to apply to targets
            seed (int): Random seed for reproducibility
            t1_check (bool): If True, draw both X and Y from the same distribution for Type-I error testing
        """
        self.root = os.path.expanduser(root)
        self.N = N
        self.M = M
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed
        self.t1_check = t1_check
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create data directory
        os.makedirs(self.root, exist_ok=True)
        
        if download:
            self._download()
            
        if not self._check_exists():
            raise RuntimeError(
                f'Dataset not found at {self.root}. '
                f'Set download=True to download it.'
            )
    
    @abstractmethod
    def _download(self):
        """Download the raw dataset."""
        pass
    
    @abstractmethod
    def _check_exists(self) -> bool:
        """Check if the processed dataset exists."""
        pass
    
    @abstractmethod
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return the two samples.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X, Y) where X ~ P and Y ~ Q
        """
        pass
    
    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the dataset as two torch tensors.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X, Y) representing samples from P and Q
        """
        return self._load_data()
    
    def __len__(self) -> int:
        """Return total number of samples (N + M)."""
        return self.N + self.M
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"N={self.N}, M={self.M}, "
                f"root='{self.root}')")
    
    @staticmethod
    def _download_url(url: str, filepath: str, chunk_size: int = 8192):
        """Download file from URL."""
        print(f"Downloading {url} to {filepath}")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with urllib.request.urlopen(url) as response:
            with open(filepath, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
    
    @staticmethod
    def _extract_archive(filepath: str, extract_root: str):
        """Extract archive file."""
        print(f"Extracting {filepath} to {extract_root}")
        
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_file:
                zip_file.extractall(extract_root)
        elif filepath.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(filepath, 'r:gz') as tar_file:
                tar_file.extractall(extract_root)
        elif filepath.endswith('.tar'):
            with tarfile.open(filepath, 'r') as tar_file:
                tar_file.extractall(extract_root)
        elif filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as gz_file:
                with open(filepath[:-3], 'wb') as out_file:
                    shutil.copyfileobj(gz_file, out_file)
        else:
            raise ValueError(f"Unsupported archive format: {filepath}")


class TabularTSTDataset(TSTDataset):
    """Base class for tabular datasets."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_type = "tabular"


class ImageTSTDataset(TSTDataset):
    """Base class for image datasets."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_type = "image"


class TextTSTDataset(TSTDataset):
    """Base class for text datasets."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_type = "text"