# This exports the tst function from main.py
from .main import tst

# Import datasets module
from . import datasets

# Specify exactly what should be available when someone imports the package
__all__ = ['tst', 'datasets']