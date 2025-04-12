import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F

class AltaStataPyTorchDataset(Dataset):
    def __init__(self, root_dir, file_pattern="**/*", transform=None):
        """
        A PyTorch Dataset for loading various file types (images, CSV, NumPy) from a directory.
        
        Args:
            root_dir (str): Root directory containing the data
            file_pattern (str): Glob pattern to match files (default: "**/*")
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.file_pattern = file_pattern
        self.transform = transform
        
        # Get list of files matching the pattern
        self.file_paths = sorted(list(self.root_dir.glob(self.file_pattern)))
        
        if not self.file_paths:
            raise ValueError(f"No files found in {root_dir} matching pattern {file_pattern}")
            
        # Create labels based on filenames
        self.labels = [1 if 'circle' in str(path) else 0 for path in self.file_paths]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load different file types
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            data = Image.open(file_path).convert('RGB')
            if self.transform:
                data = self.transform(data)
            else:
                data = F.pil_to_tensor(data).float() / 255.0
        elif file_path.suffix.lower() == '.csv':
            data = np.genfromtxt(file_path, delimiter=',')
            data = torch.FloatTensor(data)
        elif file_path.suffix.lower() == '.npy':
            data = np.load(file_path)
            data = torch.FloatTensor(data)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        return data, label 