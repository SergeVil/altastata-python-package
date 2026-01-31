"""
AltaStata Python Package
A powerful Python package for data processing and machine learning integration with Altastata.
"""

__version__ = "0.1.21"

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

# fsspec support
try:
    from .fsspec import create_filesystem, register_filesystem
    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False

__all__ = [
    'AltaStataFunctions',
    'AltaStataPyTorchDataset',
    'AltaStataTensorFlowDataset'
]

if FSSPEC_AVAILABLE:
    __all__.extend(['create_filesystem', 'register_filesystem'])
