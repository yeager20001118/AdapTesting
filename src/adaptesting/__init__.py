# This exports the tst function from main.py
from .main import tst, median, fuse

# Specify exactly what should be available when someone imports the package
__all__ = ['tst', "median", "fuse"]