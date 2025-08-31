import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import io
import os
import json
from typing import Dict, Any, Optional, Callable, Tuple
from altastata.altastata_functions import AltaStataFunctions
import tempfile
import fnmatch

# Global hashmap to store altastata function instances by ID for TensorFlow
_altastata_tensorflow_account_registry = {}
_warned_accounts = set()  # Track which accounts we've warned about

def register_altastata_functions_for_tensorflow(altastata_functions, account_id):
    _altastata_tensorflow_account_registry[account_id] = altastata_functions

def _get_altastata_functions(account_id: str) -> AltaStataFunctions:
    """Get AltaStataFunctions instance for the given account ID."""
    functions = _altastata_tensorflow_account_registry.get(account_id)
    if functions is None and account_id not in _warned_accounts:
        print(f"WARNING: No AltaStataFunctions found for account {account_id}")
        _warned_accounts.add(account_id)
    return functions

class AltaStataTensorFlowDataset(tf.data.Dataset):
    def __init__(self, account_id: str, root_dir: str, file_pattern: Optional[str] = None,
                 preprocess_fn: Optional[Callable] = None, require_files: bool = True):
        """
        A TensorFlow Dataset for loading various file types (images, CSV, NumPy) from a directory.

        Args:
            account_id (str): account_id
            root_dir (str): Root directory containing the data
            file_pattern (str, optional): Pattern to filter files (default: None)
            preprocess_fn (callable, optional): Function to preprocess samples.
                For non-image files, basic tensor conversion is applied.
            require_files (bool): Whether to require files matching the pattern (default: True)
        """
        print(f"account_id: {account_id}")
        
        self.account_id = account_id
        self.file_content_cache = {}  # Cache for small files
        self.cache_size_limit = 1024 * 1024 * 1024  # 1GB limit
        self.current_cache_size = 0
        self.max_file_size_for_cache = 16 * 1024 * 1024  # 16MB limit per file
        self.preprocess_fn = preprocess_fn

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
            # Convert root_dir to string and resolve it
            self.root_dir = str(Path(root_dir).expanduser().resolve())

            # Get all files first and convert to strings
            all_files = [str(f) for f in sorted(Path(self.root_dir).iterdir())]

        # Filter by pattern if provided
        if file_pattern is not None:
            if altastata_functions is not None:
                # For cloud storage, use fnmatch for pattern matching
                # Remove version suffix before matching
                self.file_paths = [f for f in all_files if fnmatch.fnmatch(f.split('✹')[0], file_pattern)]
            else:
                # For local storage, use fnmatch directly on strings
                self.file_paths = [f for f in all_files if fnmatch.fnmatch(f, file_pattern)]
        else:
            # All paths are already strings
            self.file_paths = all_files

        if require_files and not self.file_paths:
            raise ValueError(f"No files found in {root_dir}" +
                           (f" matching pattern {file_pattern}" if file_pattern else ""))

        # Create labels based on filenames
        self.labels = [1 if 'circle' in str(path) else 0 for path in self.file_paths]

        # Create TensorFlow dataset
        self.dataset = tf.data.Dataset.from_tensor_slices((self.file_paths, self.labels))
        self.dataset = self.dataset.map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        # Initialize the base class with the variant tensor
        variant_tensor = self.dataset._variant_tensor
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        """Returns the type specification of elements of this dataset."""
        return (
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),  # image
            tf.TensorSpec(shape=(), dtype=tf.int32)  # label
        )

    def _inputs(self):
        """Returns the list of input datasets."""
        return []

    def _read_from_altastata(self, altastata_functions, path, version_timestamp, max_file_size_for_cache, worker_pid=None, should_cache=False):
        """Read file content from AltaStata.
        
        Args:
            altastata_functions: AltaStataFunctions instance
            path: File path
            version_timestamp: Version timestamp
            max_file_size_for_cache: Maximum file size for direct reading
            worker_pid: Optional worker process ID for logging
            should_cache: Whether to cache the file content
            
        Returns:
            bytes: File content
        """
        # Get file size directly
        file_size_str = altastata_functions.get_file_attribute(path, version_timestamp, "size")
        file_size = int(file_size_str) if file_size_str else 0
        file_path = path

        if version_timestamp is None:
            _, version_part = file_path.split('✹', 1)
            version_timestamp = int(version_part.split('_')[-1])

        # Read the file using memory mapping with the latest version
        temp_file = None
        try:
            if file_size <= max_file_size_for_cache:
                #if worker_pid:
                #    print(f"Worker {worker_pid} - Reading small file {path} directly")
                data = altastata_functions.get_buffer(
                    path,
                    version_timestamp,
                    0,
                    4,
                    file_size
                )
                if should_cache and self.current_cache_size + file_size <= self.cache_size_limit:
                    self.file_content_cache[path] = data
                    self.current_cache_size += file_size
                    if worker_pid:
                        print(f"Worker {worker_pid} - Cached file {path} ({file_size} bytes)")
                elif should_cache and worker_pid:
                    print(f"Worker {worker_pid} - Not enough space in cache for {path}")
            else:
                #if worker_pid:
                #    print(f"Worker {worker_pid} - Reading large file {path} via memory mapping")
                temp_file = os.path.join(tempfile.gettempdir(), f"altastata_temp_{os.urandom(8).hex()}")
                data = altastata_functions.get_buffer_via_mapped_file(
                    temp_file,
                    path,
                    version_timestamp,
                    0,
                    4,
                    file_size
                )

            return data
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

    def _load_and_preprocess(self, file_path, label):
        """Load and preprocess a single sample."""
        def _read_file_py(path):
            # Convert tensor to string if needed
            if tf.is_tensor(path):
                path = path.numpy().decode('utf-8')
            
            altastata_functions = _get_altastata_functions(self.account_id)
            worker_pid = os.getpid()
            
            if altastata_functions is not None:
                # Use AltaStataFunctions to read file from the cloud
                #print(f"Worker {worker_pid} - Reading from cloud file: {path}")

                # Check if path has version suffix pattern
                if '✹' in path:
                    _, version_part = path.split('✹', 1)
                    version_timestamp = int(version_part.split('_')[-1])
                else:
                    version_timestamp = None

                return self._read_from_altastata(
                    altastata_functions, 
                    path, 
                    version_timestamp, 
                    self.max_file_size_for_cache, 
                    worker_pid,
                    should_cache=False
                )
            else:
                # Fall back to local file operations
                local_path = os.path.join(self.root_dir, path)
                #print(f"Worker {worker_pid} - Reading from local file: {local_path}")
                with open(local_path, 'rb') as f:
                    return f.read()

        def _process_file_py(path, content):
            """Process file and return both the processed data and file type."""
            # Convert tensor to string if needed
            if tf.is_tensor(path):
                path = path.numpy().decode('utf-8')
            
            # Get file extension from string path, handling AltaStata path format
            if '✹' in str(path):
                # Extract the file part before the version marker
                file_part = str(path).split('✹')[0]
                # Get the extension
                _, ext = os.path.splitext(file_part)
                ext = ext.lower()
            else:
                # Standard path handling
                ext = path.lower()[-4:]
            
            # Determine label based on filename
            label = 1 if 'circle' in str(path) else 0
            
            if ext in ['.jpg', '.jpeg', '.png']:
                def process_image_py(content, label):
                    image = tf.image.decode_image(content, channels=3)
                    image = tf.image.convert_image_dtype(image, tf.float32)
                    image = tf.image.resize(image, [96, 96])
                    # Set shape explicitly
                    image.set_shape([96, 96, 3])
                    return image, label
                
                image, label = tf.py_function(
                    process_image_py,
                    [content, label],
                    [tf.float32, tf.int32]
                )
                # Set the shape explicitly after py_function
                image.set_shape([96, 96, 3])
                label.set_shape([])
                
                return image, label
            elif ext == '.csv':
                data = np.genfromtxt(io.StringIO(content.decode('utf-8')), delimiter=',')
                image = tf.convert_to_tensor(data.astype(np.float32))
                image = tf.reshape(image, [-1])  # Flatten the data
                return image, label
            elif ext == '.npy':
                data = np.load(io.BytesIO(content))
                image = tf.convert_to_tensor(data.astype(np.float32))
                return image, label
            else:
                raise ValueError(f"Unsupported file type: {ext} for path {path}")

        # Read file content using py_function
        file_content = tf.py_function(
            _read_file_py,
            [file_path],
            tf.string
        )

        # Process file and get image and label in one go
        image, label = tf.py_function(
            _process_file_py,
            [file_path, file_content],
            [tf.float32, tf.int32]
        )
        
        # Set shapes explicitly to ensure they're preserved
        image.set_shape([96, 96, 3])
        label.set_shape([])

        # Apply preprocessing if needed
        if self.preprocess_fn:
            # Create a wrapper to apply preprocessing in a safe way
            def safe_preprocess(image, label):
                # Ensure image has shape
                image.set_shape([96, 96, 3])
                # Apply user-provided preprocessing
                processed_image, processed_label = self.preprocess_fn(image, label)
                # Set shape after preprocessing
                processed_image.set_shape([96, 96, 3])
                return processed_image, processed_label
            
            # Apply preprocessing with proper shape handling
            image, label = safe_preprocess(image, label)
        
        return image, label

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

    def _read_file(self, path):
        """Read bytes from a file using either AltaStataFunctions or local file operations."""
        # Convert tensor to string if needed
        if tf.is_tensor(path):
            path = path.numpy().decode('utf-8')
        
        # Check if file is in cache
        if path in self.file_content_cache:
            print(f"Worker {os.getpid()} - Reading from cache: {path}")
            return self.file_content_cache[path]
        
        altastata_functions = _get_altastata_functions(self.account_id)
        worker_pid = os.getpid()
        
        if altastata_functions is not None:
            print(f"Worker {worker_pid} - Reading from cloud file: {path}")

            # Check if path has version suffix pattern
            if '✹' in path:
                _, version_part = path.split('✹', 1)
                version_timestamp = int(version_part.split('_')[-1])
            else:
                version_timestamp = None

            return self._read_from_altastata(
                altastata_functions, 
                path, 
                version_timestamp, 
                self.max_file_size_for_cache, 
                worker_pid,
                should_cache=True
            )
        else:
            # Fall back to local file operations
            local_path = os.path.join(self.root_dir, path)
            print(f"Worker {worker_pid} - Reading from local file: {local_path}")
            with open(local_path, 'rb') as f:
                return f.read()

    def save_model(self, model: tf.keras.Model, filename: str) -> None:
        """Save a TensorFlow model to AltaStata storage.
        
        Args:
            model: The TensorFlow model to save
            filename: The path where to save the model
        """
        # Get model architecture and weights
        config = model.get_config()
        weights = model.get_weights()
        
        # Serialize to bytes using numpy
        buffer = io.BytesIO()
        np.savez(buffer, 
                 config=json.dumps(config).encode('utf-8'),
                 **{f'weight_{i}': w for i, w in enumerate(weights)})
        
        # Save to AltaStata using the working _write_file method
        buffer.seek(0)
        model_data = buffer.read()
        print(f"Saving model data length: {len(model_data)} bytes")
        self._write_file(filename, model_data)
        
        # Create provenance file with list of all file paths
        provenance_filename = filename + ".provenance.txt"
        provenance_text = "\n".join(str(file_path) for file_path in self.file_paths)
        provenance_data = provenance_text.encode('utf-8')
        print(f"Saving provenance file: {provenance_filename} with {len(self.file_paths)} file paths")
        
        # Write provenance file using our own file I/O
        self._write_file(provenance_filename, provenance_data)

    def load_model(self, filename: str) -> tf.keras.Model:
        """Load a TensorFlow model from AltaStata storage.
        
        Args:
            filename: The path to the saved model
            
        Returns:
            The loaded TensorFlow model
        """
        # Use the working _read_file method directly (we know this works)
        model_data = self._read_file(filename)
        print(f"Model data loaded: {len(model_data)} bytes")
        
        # Load from bytes
        buffer = io.BytesIO(model_data)
        data = np.load(buffer, allow_pickle=True)
        
        # Get config and weights
        config = json.loads(data['config'].tobytes().decode('utf-8'))
        weights = [data[f'weight_{i}'] for i in range(len(data.files) - 1)]  # -1 for config
        
        # Fix BatchNormalization axis compatibility issue
        def fix_batch_norm_config(layer_config):
            """Fix BatchNormalization axis parameter from list to int."""
            if layer_config.get('class_name') == 'BatchNormalization':
                axis = layer_config.get('config', {}).get('axis')
                if isinstance(axis, list) and len(axis) == 1:
                    layer_config['config']['axis'] = axis[0]
            return layer_config
        
        # Recursively fix config
        def fix_config_recursive(obj):
            if isinstance(obj, dict):
                # Fix this layer if it's BatchNormalization
                obj = fix_batch_norm_config(obj)
                # Recursively fix nested objects
                for key, value in obj.items():
                    obj[key] = fix_config_recursive(value)
            elif isinstance(obj, list):
                # Fix items in list
                obj = [fix_config_recursive(item) for item in obj]
            return obj
        
        # Apply fixes to the config
        config = fix_config_recursive(config)
        
        # Reconstruct model - handle Sequential models correctly
        try:
            if config.get('name') == 'sequential':
                # For Sequential models, use Sequential.from_config
                model = tf.keras.Sequential.from_config(config)
            else:
                # For other models, use Model.from_config
                model = tf.keras.Model.from_config(config)
        except Exception as e:
            print(f"Error loading with from_config: {e}")
            # Fallback: Try to save to temporary file and load with tf.keras.models.load_model
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                tmp_file.write(model_data)
                tmp_file_path = tmp_file.name
            
            try:
                # Try loading as a standard Keras file
                model = tf.keras.models.load_model(tmp_file_path)
                os.unlink(tmp_file_path)  # Clean up
                return model
            except Exception as e2:
                os.unlink(tmp_file_path)  # Clean up
                raise Exception(f"Could not load model. Original error: {e}, Fallback error: {e2}")
        
        model.set_weights(weights)
        
        return model 

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.file_paths) 