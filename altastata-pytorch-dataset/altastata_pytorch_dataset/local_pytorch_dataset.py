import torch
from torch.utils.data import Dataset
from typing import List, Optional, Union
import os
import numpy as np
from PIL import Image
import pandas as pd

class LocalPyTorchDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        target_transform=None
    ):
        """
        Custom PyTorch Dataset for reading files from local directory.
        
        Args:
            root_dir: Root directory containing the data files
            transform: Optional transform to be applied to the data
            target_transform: Optional transform to be applied to the target
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Get list of all files
        self.file_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.csv', '.npy')):
                    self.file_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Load data based on file extension
        if file_path.endswith(('.png', '.jpg', '.jpeg')):
            data = self._load_image(file_path)
        elif file_path.endswith('.csv'):
            data = self._load_csv(file_path)
        elif file_path.endswith('.npy'):
            data = self._load_numpy(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Apply transforms if specified
        if self.transform:
            data = self.transform(data)
            
        # For this example, we'll return the data as both input and target
        # You can modify this based on your specific use case
        if self.target_transform:
            target = self.target_transform(data)
        else:
            target = data
            
        return data, target

    def _load_image(self, file_path):
        """Load and preprocess image file"""
        image = Image.open(file_path).convert('RGB')
        return image

    def _load_csv(self, file_path):
        """Load and preprocess CSV file"""
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        return torch.FloatTensor(data)

    def _load_numpy(self, file_path):
        """Load and preprocess numpy file"""
        data = np.load(file_path)
        return torch.FloatTensor(data) 