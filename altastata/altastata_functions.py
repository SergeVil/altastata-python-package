from .base_gateway import BaseGateway

from typing import List, Any, Dict, Optional, Union, Callable
from py4j.java_gateway import JavaGateway, JavaObject, GatewayParameters, CallbackServerParameters, java_import
from py4j.java_collections import JavaList

import base64
import io
import os
import tempfile
import time
import threading
import queue

PLAIN_CHUNK_MAX_SIZE = 8 * 1024 * 1024  # 8MB - matches Java Constants.PLAIN_CHUNK_MAX_SIZE
TEMP_FILE_THRESHOLD = 64 * 1024 * 1024  # 64MB — above this, use temp file for performance



class AltaStataEventListener:
    """
    Python implementation of the Java AltaStataEventListener interface.
    This class receives events from the Java side and forwards them to a Python callback.
    """
    
    def __init__(self, callback: Callable[[str, Any], None]):
        """
        Initialize the event listener with a Python callback function.
        
        Args:
            callback: A function that takes (event_name: str, data: Any) as parameters
        """
        self.callback = callback
        self._event_queue = queue.Queue()  # Queue for events
        self._processing = False
        self._lock = threading.Lock()  # Thread synchronization
    
    def notify(self, altastata_event):
        """
        Called by Java when an event occurs.
        This method is thread-safe to handle concurrent events.
        
        Args:
            altastata_event: Java AltaStataEvent object
        """
        # Serialize event processing to prevent race conditions
        with self._lock:
            try:
                event_name = altastata_event.getEventName()
                data = altastata_event.getData()
                
                # Convert data to Python-friendly format if possible
                if data is not None:
                    data = str(data)
                
                # Call the Python callback
                self.callback(event_name, data)
            except Exception as e:
                print(f"Error in event listener callback: {e}")
    
    class Java:
        implements = ["com.altastata.api.AltaStataEventListener"]

class AltaStataFunctions(BaseGateway):
    def __init__(self, port=25333, enable_callback_server=True, callback_server_port=None):
        """
        Base initialization. This should not be called directly.
        Use from_account_dir or from_credentials instead.
        
        Args:
            port (int): Port number for the gateway
            enable_callback_server (bool): Enable callback server for event listeners
            callback_server_port (int, optional): Custom port for callback server. None = auto-select
        """
        super().__init__(port, enable_callback_server, callback_server_port)
        self._event_listeners = []  # Track registered listeners

    @classmethod
    def from_account_dir(cls, account_dir_path, port=25333, enable_callback_server=True, callback_server_port=None):
        """
        Create an instance using account directory path.
        
        Args:
            account_dir_path (str): Path to the account directory
            port (int, optional): Port number for the gateway. Defaults to 25333.
            enable_callback_server (bool, optional): Enable callback server for event listeners. Defaults to True.
            callback_server_port (int, optional): Custom port for callback server. None = auto-select. Defaults to None.
            
        Returns:
            AltaStataFunctions: New instance initialized with account directory
        """
        instance = cls(port, enable_callback_server, callback_server_port)
        instance.altastata_file_system = instance.gateway.jvm.com.altastata.api.AltaStataFileSystem(account_dir_path)
        return instance

    @classmethod
    def from_credentials(cls, user_properties, private_key_encrypted, port=25333, enable_callback_server=True, callback_server_port=None):
        """
        Create an instance using user properties and private key.
        
        Args:
            user_properties (str): User properties string
            private_key_encrypted (str): Encrypted private key
            port (int, optional): Port number for the gateway. Defaults to 25333.
            enable_callback_server (bool, optional): Enable callback server for event listeners. Defaults to True.
            callback_server_port (int, optional): Custom port for callback server. None = auto-select. Defaults to None.
            
        Returns:
            AltaStataFunctions: New instance initialized with credentials
        """
        instance = cls(port, enable_callback_server, callback_server_port)
        instance.altastata_file_system = instance.gateway.jvm.com.altastata.api.AltaStataFileSystem(user_properties, private_key_encrypted)
        return instance

    def convert_java_list_to_python(self, java_list):
        # Ensure the input is a JavaList
        if not isinstance(java_list, JavaList):
            raise TypeError("Expected a JavaList but got something else.")

        # Convert JavaList to Python list
        python_list = [item for item in java_list]

        return python_list

    def python_list_to_java_arraylist(self, python_list: list) -> 'JavaObject':
        # Create a Java ArrayList instance
        java_arraylist = self.gateway.jvm.java.util.ArrayList()

        # Add each element from the Python list to the Java ArrayList
        for item in python_list:
            java_arraylist.add(item)

        return java_arraylist

    def python_list_to_java_array(self, python_list: list) -> JavaObject:
        string_class = self.gateway.jvm.java.lang.String

        java_array = self.gateway.new_array(string_class, len(python_list))
        for i in range(len(python_list)):
            java_array[i] = python_list[i]

        return java_array

    def set_password(self, account_password: str):
        result = self.altastata_file_system.setPassword(account_password)

        # Process the result
        return result

    def create_file(self, cloud_file_path, buffer=None):
        """
        Create a new file version on cloud and add the buffer (may be empty).
        This operation is fast but does not guarantee streaming order.
        
        Args:
            cloud_file_path (str): The file path on the cloud
            buffer (bytes, optional): Initial buffer to store in the file. Defaults to None (empty buffer).
            
        Returns:
            CloudFileOperationStatus: Status of the file creation operation
        """
        if buffer is None:
            buffer = bytes()
            
        return self.altastata_file_system.createFile(cloud_file_path, buffer)

    def append_buffer_to_file(self, cloud_file_path, buffer, snapshot_time=None):
        """
        Append the buffer as an output stream to the File version.
        
        Args:
            cloud_file_path (str): The file path on the cloud
            buffer (bytes): The buffer to append
            snapshot_time (Long, optional): File version creation time. Defaults to None (current time).
            
        Raises:
            IOException: If there is an error during the append operation
        """
        self.altastata_file_system.appendBufferToFile(cloud_file_path, snapshot_time, buffer)

    def store(self, localFilesOrDirectories: List[str], localFSPrefix: str, cloudPathPrefix: str, waitUntilDone: bool):
        # Call the Java method
        java_list = self.altastata_file_system.store(self.python_list_to_java_arraylist(localFilesOrDirectories), localFSPrefix, cloudPathPrefix, waitUntilDone)

        # Convert the result to a Python list
        return self.convert_java_list_to_python(java_list)

    def retrieve_files(self, output_dir, cloud_path_prefix, including_subdirectories, snapshot_time, is_streaming, wait_until_done):
        # Call the Java method
        java_list = self.altastata_file_system.retrieve(output_dir, cloud_path_prefix, including_subdirectories, snapshot_time, is_streaming, wait_until_done)

        # Convert the Java List to Python List
        return self.convert_java_list_to_python(java_list)

    def delete_files(self, cloud_path_prefix, including_subdirectories, time_interval_start, time_interval_end):
        # Call the Java method
        java_list = self.altastata_file_system.delete(cloud_path_prefix, including_subdirectories, time_interval_start, time_interval_end)

        # Convert the Java List to Python List
        return self.convert_java_list_to_python(java_list)

    def share_files(self, cloud_path_prefix: str, including_subdirectories: bool, time_interval_start: str, time_interval_end: str, users: list) -> list:
        # Call the Java method
        # Note: time_interval_start/end can be None (null in Java) or string
        java_list = self.altastata_file_system.share(cloud_path_prefix, including_subdirectories, time_interval_start, time_interval_end, self.python_list_to_java_array(users))

        # Convert the Java List to Python List
        return self.convert_java_list_to_python(java_list)

    def revoke_reader_access(self, cloud_path_prefix: str, including_subdirectories: bool, time_interval_start: str, time_interval_end: str, readers_to_revoke: list) -> list:
        """
        Revoke reader access for the given users from files that match the cloud path and time range.
        Callable by the data owner or the custodian. The file is kept; only the listed readers lose access.

        Args:
            cloud_path_prefix: Prefix that matches the cloud files (e.g. "MyDir/file.txt" or "MyDir/").
            including_subdirectories: If True, include files in subdirectories.
            time_interval_start: Filter file versions with creation time >= this value. Use None to ignore.
            time_interval_end: Filter file versions with creation time <= this value. Use None to ignore.
            readers_to_revoke: List of user names to revoke access from.

        Returns:
            List of CloudFileOperationStatus for each revoked file version.
        """
        java_list = self.altastata_file_system.revokeReaderAccess(
            cloud_path_prefix, including_subdirectories, time_interval_start, time_interval_end,
            self.python_list_to_java_array(readers_to_revoke)
        )
        return self.convert_java_list_to_python(java_list)

    def list_cloud_files_versions(self, cloudPathPrefix, includingSubdirectories, timeIntervalStart, timeIntervalEnd):
        # Call the Java method and return the iterator
        return self.altastata_file_system.listCloudFilesVersions(cloudPathPrefix, includingSubdirectories, timeIntervalStart, timeIntervalEnd)

    def get_buffer(self, cloudFilePath, snapshotTime, startPosition, howManyChunksInParallel, size):
        """Read file content from cloud storage as ``bytes``.

        Args:
            cloudFilePath: Cloud file path (may include ✹ version suffix).
            snapshotTime: Version timestamp, or None for latest.
            startPosition: Byte offset to start reading from.
            howManyChunksInParallel: Number of chunks to download concurrently.
            size: Expected file size in bytes.

        Returns:
            bytes: File content.

        Strategy:
            ≤ 8 MB  — single ``getBufferAsBase64`` call (no temp file, no streaming).
            ≤ 64 MB — streaming 8 MB Base64 chunks over Py4J.
            > 64 MB — Java writes to temp file, Python reads and deletes it.
        """
        if snapshotTime is None:
            snapshotTime = int(time.time() * 1000)
        if size <= PLAIN_CHUNK_MAX_SIZE:
            return base64.b64decode(
                self.altastata_file_system.getBufferAsBase64(
                    cloudFilePath, snapshotTime, startPosition,
                    howManyChunksInParallel, size,
                )
            )
        if size > TEMP_FILE_THRESHOLD:
            return self._get_buffer_via_temp_file(
                cloudFilePath, snapshotTime, startPosition, howManyChunksInParallel,
            )
        java_stream = self.altastata_file_system.getFileInputStream(
            cloudFilePath, snapshotTime, startPosition, howManyChunksInParallel,
        )
        try:
            buf = bytearray(size)
            view = memoryview(buf)
            offset = 0
            while offset < size:
                b64_str = self.altastata_file_system.readBufferFromInputStreamAsBase64(
                    java_stream, min(PLAIN_CHUNK_MAX_SIZE, size - offset),
                )
                if b64_str is None:
                    break
                chunk = base64.b64decode(b64_str)
                n = len(chunk)
                view[offset:offset + n] = chunk
                offset += n
            return bytes(buf[:offset])
        finally:
            java_stream.close()

    def _get_buffer_via_temp_file(self, cloudFilePath, snapshotTime, startPosition, howManyChunksInParallel):
        """Download large file via temp file for performance.

        Java streams chunks directly to a temp file (no heap buffering),
        Python reads the result and immediately deletes it.
        """
        fd, tmp_path = tempfile.mkstemp(prefix="altastata_")
        os.close(fd)
        try:
            self.altastata_file_system.streamToFile(
                tmp_path, cloudFilePath, snapshotTime, startPosition,
                howManyChunksInParallel,
            )
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def get_java_input_stream(self, cloud_file_path, snapshot_time, start_position, how_many_chunks_in_parallel):
        """Open a Java InputStream for the given cloud file.

        Returns a Py4J reference to a Java InputStream. The caller is
        responsible for reading from and closing the stream.

        For most use cases, prefer ``get_buffer`` which handles streaming
        and cleanup internally. Use this only when true chunk-by-chunk
        streaming is needed without loading the full file into memory.
        """
        if snapshot_time is None:
            snapshot_time = int(time.time() * 1000)
        return self.altastata_file_system.getFileInputStream(
            cloud_file_path, snapshot_time, start_position, how_many_chunks_in_parallel,
        )

    def get_file_attribute(self, cloud_file_path, snapshot_time, name):
        """
        Get file attribute from Altastata file system.
        """
        try:
            if snapshot_time is None:
                snapshot_time = int(time.time() * 1000)
            result = self.altastata_file_system.getFileAttribute(cloud_file_path, snapshot_time, name)
            return str(result) if result is not None else None
        except Exception as e:
            print(f"Warning: Failed to get file attribute '{name}' for '{cloud_file_path}': {e}")
            return None

    def copy_file(self, from_cloud_file_path: str, to_cloud_file_path: str):
        """
        Copy a file from one cloud path to another.
        
        Args:
            from_cloud_file_path (str): The source file path on the cloud
            to_cloud_file_path (str): The destination file path on the cloud
            
        Returns:
            CloudFileOperationStatus: Status of the copy operation
        """
        return self.altastata_file_system.copyFile(from_cloud_file_path, to_cloud_file_path)

    def add_event_listener(self, callback: Callable[[str, Any], None]) -> AltaStataEventListener:
        """
        Add an event listener to receive file share/delete/etc events.
        
        Args:
            callback: A function that takes (event_name: str, data: Any) as parameters.
                      Will be called when events occur (e.g., SHARE, DELETE)
        
        Returns:
            AltaStataEventListener: The listener object (keep reference to remove later)
        
        Example:
            def my_event_handler(event_name, data):
                print(f"Event: {event_name}, Data: {data}")
                if event_name == "SHARE":
                    # Handle file sharing event
                    pass
                elif event_name == "DELETE":
                    # Handle file deletion event
                    pass
            
            listener = altastata.add_event_listener(my_event_handler)
            # ... do work ...
            altastata.remove_event_listener(listener)
        """
        listener = AltaStataEventListener(callback)
        self.altastata_file_system.addAltaStataEventListener(listener)
        self._event_listeners.append(listener)
        return listener
    
    def remove_event_listener(self, listener: AltaStataEventListener):
        """
        Remove a previously registered event listener.
        
        Args:
            listener: The listener object returned by add_event_listener()
        """
        try:
            self.altastata_file_system.removeAltaStataEventListener(listener)
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)
        except Exception as e:
            print(f"Warning: Failed to remove event listener: {e}")
    
    def remove_all_event_listeners(self):
        """
        Remove all registered event listeners.
        """
        for listener in self._event_listeners[:]:  # Copy list to avoid modification during iteration
            self.remove_event_listener(listener)



