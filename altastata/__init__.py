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

__all__ = [
    'AltaStataFunctions',
    'AltaStataPyTorchDataset',
    'AltaStataTensorFlowDataset'
]
