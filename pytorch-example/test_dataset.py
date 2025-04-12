import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from altastata import AltaStataPyTorchDataset

class NumpyDataset(Dataset):
    def __init__(self, root_dir, file_pattern="**/*.npy"):
        self.file_dataset = AltaStataPyTorchDataset(root_dir, file_pattern)

    def __len__(self):
        return len(self.file_dataset)

    def __getitem__(self, idx):
        file_path = self.file_dataset[idx]
        data = np.load(file_path)
        return torch.tensor(data, dtype=torch.float32)

class ImageDataset(Dataset):
    def __init__(self, root_dir, file_pattern="**/*.jpg", transform=None):
        self.file_dataset = AltaStataPyTorchDataset(root_dir, file_pattern)
        if transform is None:
            transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.transform = transform

    def __len__(self):
        return len(self.file_dataset)

    def __getitem__(self, idx):
        file_path = self.file_dataset[idx]
        image = Image.open(file_path).convert('RGB')
        return self.transform(image)

class CSVDataset(Dataset):
    def __init__(self, root_dir, file_pattern="**/*.csv"):
        self.file_dataset = AltaStataPyTorchDataset(root_dir, file_pattern)

    def __len__(self):
        return len(self.file_dataset)

    def __getitem__(self, idx):
        file_path = self.file_dataset[idx]
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        return torch.tensor(data, dtype=torch.float32)

def main():
    # Example 1: Working with numpy files
    print("Example 1: Working with numpy files")
    numpy_dataset = AltaStataPyTorchDataset(
        root_dir="data/numpy",
        pattern="*.npy"
    )
    numpy_loader = DataLoader(numpy_dataset, batch_size=2, shuffle=True)
    
    for batch_idx, (data, _) in enumerate(numpy_loader):
        print(f"Batch {batch_idx}:")
        print(f"Data shape: {data.shape}")
        break

    # Example 2: Working with images
    print("\nExample 2: Working with images")
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((224, 224))
    ])
    
    image_dataset = AltaStataPyTorchDataset(
        root_dir="data/images",
        pattern="*.jpg",
        transform=transform
    )
    image_loader = DataLoader(image_dataset, batch_size=2, shuffle=True)
    
    for batch_idx, (data, _) in enumerate(image_loader):
        print(f"Batch {batch_idx}:")
        print(f"Data shape: {data.shape}")
        break

    # Example 3: Working with CSV files
    print("\nExample 3: Working with CSV files")
    csv_dataset = AltaStataPyTorchDataset(
        root_dir="data/csv",
        pattern="*.csv"
    )
    csv_loader = DataLoader(csv_dataset, batch_size=2, shuffle=True)
    
    for batch_idx, (data, _) in enumerate(csv_loader):
        print(f"Batch {batch_idx}:")
        print(f"Data shape: {data.shape}")
        break

if __name__ == "__main__":
    main() 