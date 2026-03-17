"""
Datasets module for adaptesting package.

This module provides ready-to-use datasets for two-sample testing across different data types:
- Tabular: Medical, financial, and scientific datasets
- Image: Original vs adversarial, clean vs corrupted images
- Text: Human vs AI generated, different domains
"""

from importlib import import_module


_EXPORT_TO_MODULE = {
    'TSTDataset': '.base',
    'HiggsBoson': '.tabular',
    'HDGM': '.tabular',
    'CIFAR10Adversarial': '.image',
    'CIFAR10_1': '.image',
    'HumanAIDetection': '.text',
    'HC3': '.text',
}

__all__ = list(_EXPORT_TO_MODULE.keys())


def __getattr__(name):
    if name in _EXPORT_TO_MODULE:
        module = import_module(_EXPORT_TO_MODULE[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
