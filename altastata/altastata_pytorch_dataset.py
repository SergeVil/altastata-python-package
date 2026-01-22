import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import io
import os
import json
from typing import Dict, Any
from altastata.altastata_functions import AltaStataFunctions
import tempfile
import fnmatch

# Global hashmap to store altastata function instances by ID for PyTorch
_altastata_pytorch_account_registry = {}

# Optional torchvision for image tensor conversion
try:
    import torchvision.transforms.functional as _tv_f
except ImportError:  # pragma: no cover - depends on environment
    _tv_f = None


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to float tensor without torchvision."""
    if _tv_f is not None:
        return _tv_f.pil_to_tensor(image)
    array = np.asarray(image, dtype=np.uint8).copy()
    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor

def register_altastata_functions_for_pytorch(altastata_functions, account_id):
    _altastata_pytorch_account_registry[account_id] = altastata_functions

def _get_altastata_functions(account_id: str) -> AltaStataFunctions:
    """Get AltaStataFunctions instance for the given account ID."""
    functions = _altastata_pytorch_account_registry.get(account_id)
    if functions is None:
        print(f"WARNING: No AltaStataFunctions found for account {account_id}")
    return functions

class AltaStataPyTorchDataset(Dataset):
    def __init__(self, account_id, root_dir, file_pattern=None, transform=None, require_files=True):
        """
        A PyTorch Dataset for loading various file types (images, CSV, NumPy) from a directory.

        Args:
            account_id (str): account_id
            root_dir (str): Root directory containing the data
            file_pattern (str, optional): Pattern to filter files (default: None)
            transform (callable, optional): Transform to be applied on image samples.
                For non-image files, basic tensor conversion is applied.
            require_files (bool): Whether to require files matching the pattern (default: True)
        """
        
        print(f"account_id: {account_id}")
        
        self.account_id = account_id
        self.file_content_cache = {}  # Cache for small files
        self.cache_size_limit = 1024 * 1024 * 1024  # 1GB limit
        self.current_cache_size = 0
        self.max_file_size_for_cache = 16 * 1024 * 1024  # 16MB limit per file

        altastata_functions = _get_altastata_functions(account_id)

        if altastata_functions is not None:
            self.root_dir = root_dir

            versions_iterator = altastata_functions.list_cloud_files_versions(root_dir, False, None, None)

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
            if altastata_functions is not None:
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
                data = _pil_to_tensor(data).float() / 255.0
        elif file_ext == '.csv':
            # Convert bytes to numpy array and then to tensor
            data = np.genfromtxt(io.BytesIO(file_content), delimiter=',')
            data = torch.FloatTensor(data)
        elif file_ext == '.npy':
            # Convert bytes to numpy array and then to tensor
            data = np.load(io.BytesIO(file_content))
            if data.dtype.byteorder not in ("=", "|"):
                data = data.byteswap().view(data.dtype.newbyteorder("="))
            data = torch.FloatTensor(data)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        return data, label

    def _write_file(self, path: str, data: bytes) -> None:
        """Write bytes to a file using either AltaStataFunctions or local file operations."""

        # Remove from cache if present
        if path in self.file_content_cache:
            self.current_cache_size -= len(self.file_content_cache[path])
            del self.file_content_cache[path]
            print(f"Worker {os.getpid()} - Removed {path} from cache")

        altastata_functions = _get_altastata_functions(self.account_id)

        if altastata_functions is not None:
            # Use AltaStataFunctions to create file in the cloud
            print(f"Writing to cloud file: {path}")

            altastata_functions.create_file(path, data)
        else:
            # Fall back to local file operations
            local_path = str(self.root_dir / path)
            print(f"Writing to local file: {local_path}")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(data)

    def _read_file(self, path: str) -> bytes:
        """Read bytes from a file using either AltaStataFunctions or local file operations."""
        
        # Check if file is in cache
        if path in self.file_content_cache:
            #print(f"Worker {os.getpid()} - Reading from cache: {path}")
            return self.file_content_cache[path]
        
        altastata_functions = _get_altastata_functions(self.account_id)
        worker_pid = os.getpid()
        
        if altastata_functions is not None:
            # Use AltaStataFunctions to read file from the cloud
            #print(f"Worker {worker_pid} - Reading from cloud file: {path}")

            # Check if path has version suffix pattern
            if '✹' in path:
                # Split on '✹' first to get the version part
                _, version_part = path.split('✹', 1)

                # Then split on '_' and take the last part
                version_timestamp = int(version_part.split('_')[-1])

                #print(f"Worker {worker_pid} - Using latest version time: {version_timestamp}")
            else:
                version_timestamp = None

            #print(f"Worker {worker_pid} - Getting file size for path: {path}, version: {version_timestamp}")

            # Get specific size attribute and convert to integer
            file_size_str = altastata_functions.get_file_attribute(path, version_timestamp, "size")
            
            # Convert file size to integer with basic error handling
            try:
                file_size = int(file_size_str) if file_size_str else 0
            except (ValueError, TypeError):
                file_size = 0
            #print(f"Worker {worker_pid} - File size: {file_size}"o 
            # Read the file using memory mapping with the latest version
            temp_file = None  # Initialize temp_file
            try:
                
                if file_size <= self.max_file_size_for_cache:
                    # For small files, use get_buffer directly
                    #print(f"Worker {worker_pid} - Reading small file {path} directly")
                    data = altastata_functions.get_buffer(
                        path,
                        version_timestamp,  # Use appropriate version time
                        0,     # start_position
                        4,     # how_many_chunks_in_parallel
                        file_size
                    )
                    # Update file_size with actual size
                    file_size = len(data)
                    #print(f"Worker {worker_pid} - Determined file size from actual read: {file_size} bytes")
                    
                    # Cache the file if it's small enough and we have space
                    if self.current_cache_size + file_size <= self.cache_size_limit:
                        self.file_content_cache[path] = data
                        self.current_cache_size += file_size
                        #print(f"Worker {worker_pid} - Cached file {path} ({file_size} bytes)")
                    else:
                        #print(f"Worker {worker_pid} - Not enough space in cache for {path}")
                        pass
                else:
                    # For larger files, use memory mapping
                    # Create a temporary file for memory mapping
                    temp_file = os.path.join(tempfile.gettempdir(), f"altastata_temp_{os.urandom(8).hex()}")

                    #print(f"Worker {worker_pid} - Reading large file {path} via memory mapping")

                    data = altastata_functions.get_buffer_via_mapped_file(
                        temp_file,
                        path,
                        version_timestamp,  # Use appropriate version time
                        0,     # start_position
                        4,     # how_many_chunks_in_parallel
                        file_size
                    )

                #print(f"Worker {worker_pid} - Successfully read {len(data)} bytes from cloud")
                return data
            finally:
                # Clean up the temporary file if it exists
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
        else:
            # Fall back to local file operations
            local_path = str(self.root_dir / path)
            #print(f"Worker {worker_pid} - Reading from local file: {local_path}")
            with open(local_path, 'rb') as f:
                return f.read()

    def save_model(self, state_dict: Dict[str, torch.Tensor], filename: str) -> None:
        """Save a model's state dictionary to a file."""
        # Serialize using PyTorch
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        data = buffer.getvalue()
        print(f"Saving model data length: {len(data)} bytes")

        # Write using our own file I/O
        self._write_file(filename, data)
        
        # Create provenance file with list of all file paths
        provenance_filename = filename + ".provenance.txt"
        provenance_text = "\n".join(str(file_path) for file_path in self.file_paths)
        provenance_data = provenance_text.encode('utf-8')
        print(f"Saving provenance file: {provenance_filename} with {len(self.file_paths)} file paths")
        
        # Write provenance file using our own file I/O
        self._write_file(provenance_filename, provenance_data)

    def load_model(self, filename: str) -> Dict[str, torch.Tensor]:
        """Load a model's state dictionary from a file."""
        # Read using our own file I/O
        serialized_data = self._read_file(filename)
        print(f"Loaded model data length: {len(serialized_data)} bytes")

        # Deserialize using PyTorch
        return torch.load(io.BytesIO(serialized_data))
