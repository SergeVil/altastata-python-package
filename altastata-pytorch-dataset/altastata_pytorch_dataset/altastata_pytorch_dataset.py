import torch
from torch.utils.data import Dataset
from typing import List, Optional, Union
import os
import numpy as np
from PIL import Image
import io
import glob
import torchvision.transforms.functional as F

class AltaStataPyTorchDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        pattern: str = "*",
        transform=None,
        target_transform=None
    ):
        """
        Custom PyTorch Dataset for reading local files.
        
        Args:
            data_dir: Directory containing the data files
            pattern: Pattern to match files (e.g., "*.jpg" for images)
            transform: Optional transform to be applied to the data
            target_transform: Optional transform to be applied to the target
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Get list of files
        self.file_paths = glob.glob(os.path.join(data_dir, pattern))
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Process data based on file extension
        if file_path.endswith(('.png', '.jpg', '.jpeg')):
            data = self._process_image(file_path)
        elif file_path.endswith('.csv'):
            data = self._process_csv(file_path)
        elif file_path.endswith('.npy'):
            data = self._process_numpy(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Apply transforms if specified
        if self.transform:
            if isinstance(data, Image.Image):
                data = F.pil_to_tensor(data).float() / 255.0
            else:
                data = self.transform(data)
            
        # For this example, we'll return the data as both input and target
        # You can modify this based on your specific use case
        if self.target_transform:
            target = self.target_transform(data)
        else:
            target = data
            
        return data, target

    def _process_image(self, file_path):
        """Process image file"""
        image = Image.open(file_path).convert('RGB')
        return image

    def _process_csv(self, file_path):
        """Process CSV file"""
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        return torch.FloatTensor(data)

    def _process_numpy(self, file_path):
        """Process numpy file"""
        data = np.load(file_path)
        return torch.FloatTensor(data) 