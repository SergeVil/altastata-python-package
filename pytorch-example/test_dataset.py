import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from altastata import AltaStataPyTorchDataset
import altastata_config

def test_dataset_with_transforms(root_dir, pattern, expected_shape):
    """Test dataset with transforms and print results."""
    print(f"\nTesting dataset with pattern: {pattern}")
    print(f"Root directory: {root_dir}")
    
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((224, 224))
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