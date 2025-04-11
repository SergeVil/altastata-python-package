import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from altastata_pytorch_dataset.altastata_pytorch_dataset import AltaStataPyTorchDataset
import torch
from torchvision import transforms

def test_dataset():
    # Test with image files
    image_dataset = AltaStataPyTorchDataset(
        data_dir="data/images",
        pattern="*.jpg",
        transform=transforms.ToTensor()
    )
    print(f"Image dataset size: {len(image_dataset)}")
    if len(image_dataset) > 0:
        img, _ = image_dataset[0]
        print(f"Image shape: {img.shape}")

    # Test with CSV files
    csv_dataset = AltaStataPyTorchDataset(
        data_dir="data/csv",
        pattern="*.csv"
    )
    print(f"CSV dataset size: {len(csv_dataset)}")
    if len(csv_dataset) > 0:
        data, _ = csv_dataset[0]
        print(f"CSV data shape: {data.shape}")

    # Test with numpy files
    numpy_dataset = AltaStataPyTorchDataset(
        data_dir="data/numpy",
        pattern="*.npy"
    )
    print(f"Numpy dataset size: {len(numpy_dataset)}")
    if len(numpy_dataset) > 0:
        data, _ = numpy_dataset[0]
        print(f"Numpy data shape: {data.shape}")

if __name__ == "__main__":
    test_dataset() 