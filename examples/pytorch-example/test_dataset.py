import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from altastata import AltaStataPyTorchDataset
import altastata_config


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    array = np.asarray(img, dtype=np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img


class PILToTensor:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        return _pil_to_tensor(img)


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.uint8 and self.dtype.is_floating_point:
            return tensor.to(self.dtype) / 255.0
        return tensor.to(self.dtype)


class Resize:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Simple resize using interpolate
        import torch.nn.functional as F
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False
        result = F.interpolate(tensor, size=self.size, mode='bilinear', align_corners=False)
        if squeezed:
            result = result.squeeze(0)
        return result


def test_dataset_with_transforms(root_dir, pattern, expected_shape):
    """Test dataset with transforms and print results."""
    print(f"\nTesting dataset with pattern: {pattern}")
    print(f"Root directory: {root_dir}")
    
    transform = Compose([
        PILToTensor(),
        ConvertImageDtype(torch.float32),
        Resize((224, 224))
    ])
    
    dataset = AltaStataPyTorchDataset(
        "bob123_rsa",
        root_dir=root_dir,
        file_pattern=pattern,
        transform=transform
    )
    
    print(f"Number of files found: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=True)
    batch_data, batch_labels = next(iter(dataloader))
    
    print(f"Data shape: {batch_data.shape}")
    print(f"Data type: {batch_data.dtype}")
    print(f"Data range: [{batch_data.min().item():.3f}, {batch_data.max().item():.3f}]")
    print(f"Labels: {batch_labels.tolist()}")
    
    assert batch_data.shape == expected_shape, f"Expected shape {expected_shape}, got {batch_data.shape}"
    print("Test passed successfully!")
    print("-" * 50)

def main():
    print("Starting dataset tests...")
    print("=" * 50)
    
    # Test with numpy files
    test_dataset_with_transforms(
        "pytorch_test/data/numpy",
        "*.npy",
        torch.Size([2, 10, 5])
    )
    
    # Test with images
    test_dataset_with_transforms(
        "pytorch_test/data/images",
        "*.png",
        torch.Size([2, 3, 224, 224])
    )
    
    # Test with CSV files
    test_dataset_with_transforms(
        "pytorch_test/data/csv",
        "*.csv",
        torch.Size([2, 11, 5])
    )
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 