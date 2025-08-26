"""
Datasets module for adaptesting package.

This module provides ready-to-use datasets for two-sample testing across different data types:
- Tabular: Medical, financial, and scientific datasets
- Image: Original vs adversarial, clean vs corrupted images
- Text: Human vs AI generated, different domains
"""

from .base import TSTDataset
from .tabular import *
from .image import *
from .text import *

__all__ = [
    'TSTDataset',
    # Challenging tabular datasets
    'HiggsBoson',
    # 'AdultIncome', 
    # 'ChallengingSynthetic',
    'HDGM',
    # Easy tabular datasets (for comparison)
    # 'BreastCancer',
    # 'Wine',
    # Image datasets with real attacks
    'CIFAR10Adversarial',
    # 'MNISTCorrupted',
    # 'ImageNetAdversarial',
    # 'NaturalImageShifts',
    # Text datasets using real sources
    'HumanAIDetection',
    'HC3',
    # 'FakeNewsDetection',
    # 'DomainShift',
    # 'SentimentShift'
]