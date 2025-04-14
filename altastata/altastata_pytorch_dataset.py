import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import io
import os
import json
from typing import Dict, Any
from altastata.altastata_functions import AltaStataFunctions
import tempfile
import fnmatch

class AltaStataPyTorch:
    """
    High-level class that combines AltaStataPyTorchDataset with AltaStataFunctions.
    Provides a unified interface for both PyTorch dataset operations and AltaStata functionality.
    """
    def __init__(self, altastata_functions: AltaStataFunctions):
        """
        Initialize AltaStataPyTorch with an instance of AltaStataFunctions.
        
        Args:
            altastata_functions (AltaStataFunctions): An instance of AltaStataFunctions
        """
        self.altastata_functions = altastata_functions

    def create_dataset(self, root_dir, file_pattern=None, transform=None, require_files=True):
        """
        Create and return an instance of AltaStataPyTorchDataset.
        
        Args:
            root_dir (str): Root directory containing the data
            file_pattern (str, optional): Pattern to filter files (default: None)
            transform (callable, optional): Transform to be applied on image samples
            require_files (bool): Whether to require files matching the pattern (default: True)
            
        Returns:
            AltaStataPyTorchDataset: An instance of the dataset
        """
        return self.__AltaStataPyTorchDataset(self, root_dir, file_pattern, transform, require_files)

    class __AltaStataPyTorchDataset(Dataset):
        def __init__(self, outer_instance, root_dir, file_pattern=None, transform=None, require_files=True):
            """
            A PyTorch Dataset for loading various file types (images, CSV, NumPy) from a directory.
            
            Args:
                outer_instance (AltaStataPyTorch): The outer class instance
                root_dir (str): Root directory containing the data
                file_pattern (str, optional): Pattern to filter files (default: None)
                transform (callable, optional): Transform to be applied on image samples. 
                    For non-image files, basic tensor conversion is applied.
                require_files (bool): Whether to require files matching the pattern (default: True)
            """
            self.outer = outer_instance
            if self.outer.altastata_functions is not None:
                self.root_dir = root_dir

                versions_iterator = self.outer.altastata_functions.list_cloud_files_versions(root_dir, False, None, None)

                # Convert iterator to list of versions
                all_files = []
                for java_array in versions_iterator:
                    for element in java_array:
                        all_files.append(str(element))

                if not all_files:
                    raise FileNotFoundError(f"No versions found for file: {root_dir}")
                else:
                    print('all_files:')
                    for file in all_files:
                        print(f'  {file}')
            else:
                self.root_dir = Path(root_dir).expanduser().resolve()

                # Get all files first
                all_files = sorted(list(self.root_dir.iterdir()))
                print('all_files:')
                for file in all_files:
                    print(f'  {file}')

            self.transform = transform

            # Filter by pattern if provided
            if file_pattern is not None:
                if self.outer.altastata_functions is not None:
                    # For cloud storage, use fnmatch for pattern matching
                    # Remove version suffix before matching
                    self.file_paths = [f for f in all_files if fnmatch.fnmatch(f.split('✹')[0], file_pattern)]
                else:
                    # For local storage, all_files are Path objects
                    self.file_paths = [f for f in all_files if f.match(file_pattern)]
            else:
                self.file_paths = all_files
            
            if require_files and not self.file_paths:
                raise ValueError(f"No files found in {root_dir}" + 
                               (f" matching pattern {file_pattern}" if file_pattern else ""))
            
            # Create labels based on filenames
            self.labels = [1 if 'circle' in str(path) else 0 for path in self.file_paths]

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            
            file_path = self.file_paths[idx]
            label = self.labels[idx]
            
            # Read file content using _read_file method
            file_content = self._read_file(file_path)
            
            # Get file extension for type checking
            if isinstance(file_path, Path):
                file_ext = file_path.suffix.lower()
            else:
                # For cloud storage, file_path is a string
                file_ext = os.path.splitext(file_path.split('✹')[0])[1].lower()
            
            # Load different file types based on extension
            if file_ext in ['.jpg', '.jpeg', '.png']:
                # Convert bytes to PIL Image
                data = Image.open(io.BytesIO(file_content)).convert('RGB')
                # Apply transform if provided, otherwise just convert to tensor
                if self.transform:
                    data = self.transform(data)
                else:
                    data = F.pil_to_tensor(data).float() / 255.0
            elif file_ext == '.csv':
                # Convert bytes to numpy array and then to tensor
                data = np.genfromtxt(io.BytesIO(file_content), delimiter=',')
                data = torch.FloatTensor(data)
            elif file_ext == '.npy':
                # Convert bytes to numpy array and then to tensor
                data = np.load(io.BytesIO(file_content))
                data = torch.FloatTensor(data)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            return data, label

        def _write_file(self, path: str, data: bytes) -> None:
            """Write bytes to a file using either AltaStataFunctions or local file operations."""
            if self.outer.altastata_functions is not None:
                # Use AltaStataFunctions to create file in the cloud
                print(f"Writing to cloud file: {path}")
                self.outer.altastata_functions.create_file(path, data)
            else:
                # Fall back to local file operations
                local_path = str(self.root_dir / path)
                print(f"Writing to local file: {local_path}")
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'wb') as f:
                    f.write(data)

        def _read_file(self, path: str) -> bytes:
            """Read bytes from a file using either AltaStataFunctions or local file operations."""
            if self.outer.altastata_functions is not None:
                # Use AltaStataFunctions to read file from the cloud
                print(f"Reading from cloud file: {path}")

                latest_version_timestamp = self.get_latest_version_timestamp(path)

                try:
                    # Get the file size using the latest version
                    try:
                        file_size_json = self.outer.altastata_functions.get_file_attribute(path, latest_version_timestamp, "size")

                        # Access the file size
                        result = json.loads(file_size_json)
                        file_size = int(result['DataSizeAttribute']['fileSize'])

                        print(f"File size: {file_size} bytes")
                    except Exception as e:
                        print(f"Error getting file size: {e}. Using default size.")
                        file_size = 50 * 1024 * 1024  # 50MB default

                    # Create a temporary file for memory mapping
                    temp_file = os.path.join(tempfile.gettempdir(), f"altastata_temp_{os.urandom(8).hex()}")

                    # Read the file using memory mapping with the latest version
                    data = self.outer.altastata_functions.get_buffer_via_mapped_file(
                        temp_file,
                        path,
                        latest_version_timestamp,  # Use latest version timestamp
                        0,     # start_position
                        4,     # how_many_chunks_in_parallel
                        file_size
                    )
                    
                    print(f"Read {len(data)} bytes from cloud")
                    return data
                finally:
                    # Clean up the temporary file if it exists
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            else:
                # Fall back to local file operations
                local_path = str(self.root_dir / path)
                print(f"Reading from local file: {local_path}")
                with open(local_path, 'rb') as f:
                    return f.read()

        def get_latest_version_timestamp(self, path):
            # Get the latest version time using list_cloud_files_versions
            try:
                if '✹' in path:
                    latest_version = path
                else:
                    # For file paths, we need to use just the path
                    # Pass False for includingSubdirectories since we're looking for a specific file
                    versions_iterator = self.outer.altastata_functions.list_cloud_files_versions(path, True, None, None)

                    # Convert iterator to list of versions
                    versions = []
                    for java_array in versions_iterator:
                        for element in java_array:
                            versions.append(str(element))

                    if not versions:
                        raise FileNotFoundError(f"No versions found for file: {path}")

                    # Get the latest version (last in the list)
                    latest_version = versions[-1]

                # Split on '✹' first to get the version part
                _, version_part = latest_version.split('✹', 1)

                # Then split on '_' and take the last part
                latest_version_timestamp = int(version_part.split('_')[-1])

                print(f"Using latest version time: {latest_version_timestamp}")
            except Exception as e:
                print(f"Error getting file versions: {e}. Using current time.")
                # Fallback to current time if we can't get versions
                latest_version_timestamp = self.outer.altastata_functions.gateway.jvm.java.lang.System.currentTimeMillis()
                print(f"Using current time: {latest_version_timestamp}")
            return latest_version_timestamp

        def save_model(self, state_dict: Dict[str, torch.Tensor], filename: str) -> None:
            """Save a model's state dictionary to a file."""
            # Serialize using PyTorch
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            data = buffer.getvalue()
            print(f"Saving model data length: {len(data)} bytes")
            
            # Write using our own file I/O
            self._write_file(filename, data)

        def load_model(self, filename: str) -> Dict[str, torch.Tensor]:
            """Load a model's state dictionary from a file."""
            # Read using our own file I/O
            serialized_data = self._read_file(filename)
            print(f"Loaded model data length: {len(serialized_data)} bytes")
            
            # Deserialize using PyTorch
            return torch.load(io.BytesIO(serialized_data))