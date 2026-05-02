"""
AltaStata Python Package
A powerful Python package for data processing and machine learning integration with Altastata.
"""

# Derive __version__ from the installed package metadata (set by setup.py at
# build time), so we have a single source of truth and never need to bump it
# in two places. Previously hardcoded as a literal here and silently drifted
# from the real PyPI version across many releases. importlib.metadata is
# stdlib since Python 3.8.
from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFound

try:
    __version__ = _pkg_version("altastata")
except _PkgNotFound:
    # Importing from a source checkout without `pip install -e .` (or the
    # package was renamed at install time) — fall back rather than crash.
    __version__ = "0.0.0+unknown"

from .altastata_functions import AltaStataFunctions

# Lazy import for PyTorch
def _import_pytorch_dataset():
    from .altastata_pytorch_dataset import AltaStataPyTorchDataset
    return AltaStataPyTorchDataset

# Lazy import for TensorFlow
def _import_tensorflow_dataset():
    from .altastata_tensorflow_dataset import AltaStataTensorFlowDataset
    return AltaStataTensorFlowDataset

# Create lazy loaders for optional datasets
class _LazyTensorFlowDataset:
    def __init__(self):
        self._dataset_class = None
    
    def __call__(self, *args, **kwargs):
        if self._dataset_class is None:
            self._dataset_class = _import_tensorflow_dataset()
        return self._dataset_class(*args, **kwargs)

class _LazyPyTorchDataset:
    def __init__(self):
        self._dataset_class = None
    
    def __call__(self, *args, **kwargs):
        if self._dataset_class is None:
            self._dataset_class = _import_pytorch_dataset()
        return self._dataset_class(*args, **kwargs)

AltaStataPyTorchDataset = _LazyPyTorchDataset()
AltaStataTensorFlowDataset = _LazyTensorFlowDataset()

# fsspec support - import the actual availability flag from the fsspec module
try:
    from .fsspec import create_filesystem, register_filesystem, FSSPEC_AVAILABLE
except ImportError:
    FSSPEC_AVAILABLE = False

__all__ = [
    'AltaStataFunctions',
    'AltaStataPyTorchDataset',
    'AltaStataTensorFlowDataset'
]

if FSSPEC_AVAILABLE:
    __all__.extend(['create_filesystem', 'register_filesystem'])
