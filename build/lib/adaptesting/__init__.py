# This exports the tst function from main.py
from importlib import import_module

from .main import tst, idt

# Specify exactly what should be available when someone imports the package
__all__ = ['tst', 'idt', 'datasets']


def __getattr__(name):
    if name == 'datasets':
        datasets_module = import_module('.datasets', __name__)
        return datasets_module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
