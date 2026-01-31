"""
Generic fsspec integration for AltaStata

This module provides a clean, generic fsspec filesystem interface
that can be used with any AltaStataFunctions instance.
"""

import io
from typing import Any, Dict, List, Optional, Union

try:
    import fsspec
    from fsspec.spec import AbstractFileSystem
    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False
    AbstractFileSystem = object  # Dummy base class when fsspec not available


class AltaStataFileSystem(AbstractFileSystem):
    """Generic fsspec filesystem for AltaStata using py4j."""
    
    protocol = "altastata"
    
    def __init__(self, altastata_functions, account_id: str = "default", **kwargs):
        """
        Initialize fsspec filesystem with AltaStata connection.
        
        Args:
            altastata_functions: AltaStataFunctions instance
            account_id: Account identifier
        """
        if not FSSPEC_AVAILABLE:
            raise ImportError("fsspec is required. Install with: pip install fsspec")
        
        super().__init__(**kwargs)
        self.altastata_functions = altastata_functions
        self.account_id = account_id
    
    def _strip_protocol(self, path: str) -> str:
        """Remove protocol from path."""
        return path[12:] if path.startswith("altastata://") else path
    
    def ls(self, path: str, detail: bool = False, **kwargs) -> Union[List[str], List[Dict[str, Any]]]:
        """List files (latest versions only)."""
        path = self._strip_protocol(path)
        
        # Get all file versions and keep only the latest
        versions_iterator = self.altastata_functions.list_cloud_files_versions(path, True, None, None)
        
        file_versions = {}  # base_path -> latest_version_path
        
        for java_array in versions_iterator:
            for element in java_array:
                file_path = str(element)
                # Extract base path (remove version info)
                base_path = file_path.split('✹')[0] if '✹' in file_path else file_path
                
                # Keep the latest version (first occurrence is sufficient)
                if base_path not in file_versions:
                    file_versions[base_path] = file_path
        
        files = []
        for base_path in sorted(file_versions.keys()):
            if detail:
                try:
                    size_str = self.altastata_functions.get_file_attribute(base_path, None, "size")
                    size = int(size_str) if size_str else 0
                    files.append({
                        "name": base_path,
                        "size": size,
                        "type": "file",
                        "created": None,
                        "modified": None,
                    })
                except Exception:
                    files.append({
                        "name": base_path,
                        "size": 0,
                        "type": "file",
                        "created": None,
                        "modified": None,
                    })
            else:
                files.append(base_path)
        
        return files
    
    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        """Get file information (latest version)."""
        path = self._strip_protocol(path)
        
        size_str = self.altastata_functions.get_file_attribute(path, None, "size")
        size = int(size_str) if size_str else 0
        
        return {
            "name": path,
            "size": size,
            "type": "file",
            "created": None,
            "modified": None,
        }
    
    def exists(self, path: str, **kwargs) -> bool:
        """Check if file exists (latest version)."""
        try:
            path = self._strip_protocol(path)
            self.altastata_functions.get_file_attribute(path, None, "size")
            return True
        except Exception:
            return False
    
    def open(self, path: str, mode: str = "rb", **kwargs) -> io.IOBase:
        """Open file for reading (latest version)."""
        if "w" in mode or "a" in mode:
            raise NotImplementedError("Writing not implemented")
        
        path = self._strip_protocol(path)
        return AltaStataFile(self, path, mode)


class AltaStataFile(io.IOBase):
    """Simple file-like object for reading from AltaStata."""
    
    def __init__(self, filesystem: AltaStataFileSystem, path: str, mode: str):
        self.filesystem = filesystem
        self.path = path
        self.mode = mode
        
        # Use current system time
        import time
        current_time = int(time.time() * 1000)
        self._java_stream = self.filesystem.altastata_functions.get_java_input_stream(self.path, current_time, 0, 4)
        self._position = 0  # Track position based on last operation
    
    def read(self, size: int = -1) -> Union[bytes, str]:
        """Read data from file."""
        # Use the buffer method instead of direct read
        if size == -1:
            # Read all available data
            data = self.filesystem.altastata_functions.get_buffer_from_input_stream(self._java_stream, 1024)
        else:
            # Read specified amount
            data = self.filesystem.altastata_functions.get_buffer_from_input_stream(self._java_stream, size)
        
        # Update position based on bytes read
        if data:
            self._position += len(data)
        
        if self.mode != "rb" and data:
            return data.decode('utf-8')
        
        return data if data else (b"" if self.mode == "rb" else "")
    
    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position."""
        # Mark position for reset capability
        try:
            self.filesystem.altastata_functions.mark_input_stream_position(self._java_stream, offset)
        except Exception:
            pass
        
        # Update position based on seek
        self._position = offset
        return offset
    
    def tell(self) -> int:
        """Get current position based on last operation (read or seek)."""
        return self._position
    
    def close(self):
        """Close file."""
        self._java_stream = None


def create_filesystem(altastata_functions, account_id: str = "default"):
    """
    Create a generic fsspec filesystem for AltaStata.
    
    Args:
        altastata_functions: AltaStataFunctions instance
        account_id: Account identifier
        
    Returns:
        AltaStataFileSystem: fsspec filesystem instance
    """
    return AltaStataFileSystem(altastata_functions, account_id)


def register_filesystem():
    """Register AltaStata filesystem with fsspec."""
    if FSSPEC_AVAILABLE:
        fsspec.register_implementation("altastata", AltaStataFileSystem)
