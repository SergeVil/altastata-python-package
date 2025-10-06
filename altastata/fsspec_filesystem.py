"""
fsspec filesystem interface for AltaStata

Simple implementation that automatically uses the latest version of files.
"""

import io
from typing import Any, Dict, List, Optional, Union

import fsspec
from fsspec.spec import AbstractFileSystem

from .altastata_functions import AltaStataFunctions


class AltaStataFileSystem(AbstractFileSystem):
    """fsspec-compatible filesystem for AltaStata (latest version only)."""
    
    protocol = "altastata"
    
    def __init__(self, account_id: str, account_dir: Optional[str] = None, 
                 user_properties: Optional[str] = None, 
                 private_key_encrypted: Optional[str] = None,
                 password: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        
        self.account_id = account_id
        
        # Initialize AltaStata connection
        if account_dir:
            self._altastata_functions = AltaStataFunctions.from_account_dir(account_dir)
        elif user_properties and private_key_encrypted:
            self._altastata_functions = AltaStataFunctions.from_credentials(
                user_properties, private_key_encrypted
            )
        else:
            raise ValueError("Either account_dir or user_properties/private_key_encrypted required")
        
        if password:
            self._altastata_functions.set_password(password)
    
    def _strip_protocol(self, path: str) -> str:
        """Remove protocol from path."""
        return path[12:] if path.startswith("altastata://") else path
    
    def ls(self, path: str, detail: bool = False, **kwargs) -> Union[List[str], List[Dict[str, Any]]]:
        """List files (latest versions only)."""
        path = self._strip_protocol(path)
        
        # Get all file versions and keep only the latest
        versions_iterator = self._altastata_functions.list_cloud_files_versions(path, True, None, None)
        
        file_versions = {}  # base_path -> latest_version_path
        
        for java_array in versions_iterator:
            for element in java_array:
                file_path = str(element)
                # Extract base path (remove version info)
                base_path = file_path.split('✹')[0] if '✹' in file_path else file_path
                
                # Keep the latest version (first occurrence is sufficient for demo)
                if base_path not in file_versions:
                    file_versions[base_path] = file_path
        
        files = []
        for base_path in sorted(file_versions.keys()):
            if detail:
                try:
                    size_str = self._altastata_functions.get_file_attribute(base_path, None, "size")
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
        
        size_str = self._altastata_functions.get_file_attribute(path, None, "size")
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
            self._altastata_functions.get_file_attribute(path, None, "size")
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
        self._content = None
        self._position = 0
    
    def _load_content(self):
        """Load file content (latest version)."""
        if self._content is not None:
            return
        
        altastata_functions = self.filesystem._altastata_functions
        size_str = altastata_functions.get_file_attribute(self.path, None, "size")
        file_size = int(size_str) if size_str else 0
        
        if file_size == 0:
            self._content = b""
        else:
            self._content = altastata_functions.get_buffer(self.path, None, 0, 4, file_size)
    
    def read(self, size: int = -1) -> Union[bytes, str]:
        """Read data from file."""
        self._load_content()
        
        if self.mode == "rb":
            # Binary mode
            if size == -1:
                data = self._content[self._position:]
                self._position = len(self._content)
            else:
                end_pos = min(self._position + size, len(self._content))
                data = self._content[self._position:end_pos]
                self._position = end_pos
            return data
        else:
            # Text mode
            if not hasattr(self, '_text_content'):
                self._text_content = self._content.decode('utf-8')
            
            if size == -1:
                data = self._text_content[self._position:]
                self._position = len(self._text_content)
            else:
                end_pos = min(self._position + size, len(self._text_content))
                data = self._text_content[self._position:end_pos]
                self._position = end_pos
            return data
    
    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position in file."""
        self._load_content()
        
        if whence == 0:  # SEEK_SET
            self._position = offset
        elif whence == 1:  # SEEK_CUR
            self._position += offset
        elif whence == 2:  # SEEK_END
            content_len = len(self._content) if self.mode == "rb" else len(getattr(self, '_text_content', self._content.decode('utf-8')))
            self._position = content_len + offset
        else:
            raise ValueError("Invalid whence value")
        
        self._position = max(0, self._position)
        return self._position
    
    def tell(self) -> int:
        """Get current position."""
        return self._position
    
    def close(self):
        """Close file."""
        self._content = None
        if hasattr(self, '_text_content'):
            delattr(self, '_text_content')


def register_altastata_filesystem():
    """Register AltaStata filesystem with fsspec."""
    fsspec.register_implementation("altastata", AltaStataFileSystem)