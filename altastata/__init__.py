"""
AltaStata Python Package
A powerful Python package for data processing and machine learning integration with Altastata.
"""

__version__ = "0.1.16"

from .altastata_functions import AltaStataFunctions
from .altastata_pytorch_dataset import AltaStataPyTorchDataset

# Lazy import for TensorFlow
def _import_tensorflow_dataset():
    from .altastata_tensorflow_dataset import AltaStataTensorFlowDataset
    return AltaStataTensorFlowDataset

# Create a lazy loader for TensorFlow dataset
class _LazyTensorFlowDataset:
    def __init__(self):
        self._dataset_class = None
    
    def __call__(self, *args, **kwargs):
        if self._dataset_class is None:
            self._dataset_class = _import_tensorflow_dataset()
        return self._dataset_class(*args, **kwargs)

AltaStataTensorFlowDataset = _LazyTensorFlowDataset()

# fsspec support
def _import_fsspec_filesystem():
    from .fsspec_filesystem import AltaStataFileSystem, register_altastata_filesystem
    return AltaStataFileSystem, register_altastata_filesystem

# Create lazy loader for fsspec filesystem
class _LazyFSSpecFilesystem:
    def __init__(self):
        self._filesystem_class = None
        self._register_func = None
    
    def __call__(self, *args, **kwargs):
        if self._filesystem_class is None:
            self._filesystem_class, self._register_func = _import_fsspec_filesystem()
        return self._filesystem_class(*args, **kwargs)
    
    def register(self):
        """Register the filesystem with fsspec."""
        if self._register_func is None:
            _, self._register_func = _import_fsspec_filesystem()
        return self._register_func()

AltaStataFileSystem = _LazyFSSpecFilesystem()

__all__ = [
    'AltaStataFunctions',
    'AltaStataPyTorchDataset',
    'AltaStataTensorFlowDataset',
    'AltaStataFileSystem'
]
