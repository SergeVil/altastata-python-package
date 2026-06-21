from .grpc_client import AltaStataGrpcClient, GrpcEndpoint

from typing import List, Any, Dict, Optional, Callable, Tuple

import io
import json
import os
import threading
import queue
import warnings
import urllib.error
import urllib.parse
import urllib.request

def _parse_user_name_from_properties(text: str) -> str:
    """Extract the ``myuser`` value from a user.properties text blob.

    Mirrors what {@code com.altastata.utils.Account} does on the Java side
    so the boto3 helper can derive the same user name that
    ``AccountRegistry.getOrCreate`` / ``getOrCreateFromDir`` would have
    chosen.

    Raises:
        ValueError: if no ``myuser=...`` line is present.
    """
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("myuser="):
            value = line.split("=", 1)[1].strip()
            if value:
                return value
    raise ValueError("user_properties does not contain a non-empty 'myuser=' line")


def _http_put_text(url: str, body: str, timeout_s: float = 30.0) -> Tuple[int, bytes]:
    """Issue a ``PUT`` with a plain-text body using stdlib urllib.

    Used by the S3 boto3 helper to drive the three admin bootstrap PUTs
    (setUserProperties / setPrivateKey / setPassword). Kept on stdlib so we
    don't have to pull ``requests`` into install_requires just for this one
    code path.

    Returns:
        (status_code, body_bytes). Caller decides how to handle non-2xx.
    """
    req = urllib.request.Request(
        url=url,
        data=body.encode("utf-8"),
        method="PUT",
        headers={"Content-Type": "text/plain"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b""



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

class AltaStataFunctions:
    def __init__(
        self,
        port=25333,
        enable_callback_server=True,
        callback_server_port=None,
        *,
        transport: str = "grpc",
        grpc_client: Optional[AltaStataGrpcClient] = None,
    ):
        """
        Base initialization. This should not be called directly.
        Use from_account_dir or from_credentials instead.
        
        Args:
            port (int): Port number for the gateway
            enable_callback_server (bool): Enable callback server for event listeners
            callback_server_port (int, optional): Custom port for callback server. None = auto-select
        """
        self.transport = "grpc"
        if transport.lower() != "grpc":
            warnings.warn(
                "transport parameter is kept for compatibility and ignored; "
                "AltaStataFunctions now always uses gRPC.",
                DeprecationWarning,
                stacklevel=2,
            )
        del port
        del enable_callback_server
        del callback_server_port
        self.grpc_client = grpc_client
        self._event_listeners = []  # Track registered listeners

        # Material the S3 boto3 helper needs to drive the admin bootstrap
        # PUTs (setUserProperties / setPrivateKey / setPassword) without
        # asking the caller to repeat what they already gave us.
        # Populated by from_account_dir / from_credentials below.
        self._account_dir_path: Optional[str] = None
        self._user_properties: Optional[str] = None
        self._private_key_encrypted: Optional[str] = None
        self._cached_password: Optional[str] = None
        # Cache of bootstrapped S3 credentials keyed by endpoint URL. The
        # access/secret pair never changes for a given (endpoint, user) so a
        # second `s3_credentials()` / `boto3_s3()` call is a dict lookup.
        self._s3_credentials_cache: Dict[str, Dict[str, str]] = {}

    @classmethod
    def from_account_dir(
        cls,
        account_dir_path,
        port=25333,
        enable_callback_server=True,
        callback_server_port=None,
        *,
        transport: str = "grpc",
        password: Optional[str] = None,
        user_name: Optional[str] = None,
        grpc_endpoint: Optional[GrpcEndpoint] = None,
        grpc_setup_port: int = 9880,
        grpc_auto_start_server: bool = True,
    ):
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
        # grpc_setup_port is accepted for backwards compatibility with
        # callers from before the AuthService.Login migration but is no
        # longer used internally; the gateway is reached via grpc_endpoint.
        del grpc_setup_port
        endpoint = grpc_endpoint or GrpcEndpoint()
        client = AltaStataGrpcClient.from_account_dir(
            account_dir_path=account_dir_path,
            password=password,
            user_name=user_name,
            endpoint=endpoint,
            auto_start_server=grpc_auto_start_server,
        )
        instance = cls(
            port=port,
            enable_callback_server=enable_callback_server,
            callback_server_port=callback_server_port,
            transport=transport,
            grpc_client=client,
        )
        instance._account_dir_path = account_dir_path
        instance._cached_password = password
        return instance

    @classmethod
    def from_credentials(
        cls,
        user_properties,
        private_key_encrypted,
        port=25333,
        enable_callback_server=True,
        callback_server_port=None,
        *,
        transport: str = "grpc",
        password: Optional[str] = None,
        user_name: Optional[str] = None,
        grpc_endpoint: Optional[GrpcEndpoint] = None,
        grpc_setup_port: int = 9880,
        grpc_auto_start_server: bool = True,
    ):
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
        # See note in from_account_dir about grpc_setup_port deprecation.
        del grpc_setup_port
        endpoint = grpc_endpoint or GrpcEndpoint()
        client = AltaStataGrpcClient.from_credentials(
            user_properties=user_properties,
            private_key_encrypted=private_key_encrypted,
            password=password,
            user_name=user_name,
            endpoint=endpoint,
            auto_start_server=grpc_auto_start_server,
        )
        instance = cls(
            port=port,
            enable_callback_server=enable_callback_server,
            callback_server_port=callback_server_port,
            transport=transport,
            grpc_client=client,
        )
        instance._user_properties = user_properties
        instance._private_key_encrypted = private_key_encrypted
        instance._cached_password = password
        return instance

    def convert_java_list_to_python(self, java_list):
        return list(java_list) if java_list is not None else []

    def set_password(self, account_password: str):
        # Remember the plaintext so s3_credentials() / boto3_s3() / install_aws_env()
        # can drive the S3 admin PUTs without forcing the caller to retype it.
        self._cached_password = account_password
        return self.grpc_client.set_password(account_password)

    # ------------------------------------------------------------------ #
    # S3 / boto3 bridge                                                  #
    # ------------------------------------------------------------------ #
    def s3_credentials(
        self,
        *,
        password: Optional[str] = None,
        endpoint: Optional[str] = None,
        region: str = "us-east-1",
    ) -> Dict[str, str]:
        """Bootstrap and return boto3-ready S3 credentials.

        Drives the three admin PUTs against the S3 gateway running inside the
        same ``altastata-services`` JVM that backs this Python session
        (setUserProperties → setPrivateKey → setPassword), then returns the
        access/secret pair the gateway generated.

        Args:
            password: Plaintext account password used to unlock the encrypted
                private key on the gateway side. Falls back to the value
                cached by :meth:`set_password` (or the ``password`` kwarg
                passed to :meth:`from_account_dir` / :meth:`from_credentials`).
                Required for non-HSM users; HSM/HPCS users can pass ``""``.
            endpoint: Base URL of the S3 gateway. Defaults to
                ``http://<grpc-host>:9876`` for remote gRPC endpoints, or
                ``http://127.0.0.1:9876`` for a local co-hosted gateway.
            region: AWS region for SigV4. The gateway is region-agnostic but
                boto3 still demands a value; ``us-east-1`` is the safe default.

        Returns:
            Dict with keys ``endpoint_url``, ``aws_access_key_id``,
            ``aws_secret_access_key``, ``region_name`` — directly usable as
            ``boto3.client('s3', **result)``.

        Caveat:
            The third PUT (``setPassword``) calls
            ``AltaStataFileSystem.setPassword(...)`` on the shared instance
            stored in ``AccountRegistry``. Passing the same password you
            already used for this session is a no-op; passing a different
            one mutates the shared instance. The planned unified
            ``UserAdminRegistry`` (see
            ``mycloud/ALTASTATA_SERVICES_UBER_DESIGN.md`` §3.1) will close
            this gap.
        """
        endpoint = (endpoint or self._resolve_s3_endpoint()).rstrip("/")

        cached = self._s3_credentials_cache.get(endpoint)
        if cached is not None and cached.get("region_name") == region:
            return dict(cached)

        pw = password if password is not None else self._cached_password
        if pw is None:
            raise ValueError(
                "s3_credentials() requires a password. Either pass "
                "password=... explicitly, or call set_password() (or pass "
                "password=... to from_account_dir / from_credentials) first."
            )

        user_name, user_properties, private_key_encrypted = (
            self._read_bootstrap_material()
        )

        def _put(path: str, body: str) -> Dict[str, Any]:
            status, raw = _http_put_text(f"{endpoint}{path}", body)
            if status < 200 or status >= 300:
                raise RuntimeError(
                    f"S3 admin PUT {path} failed with HTTP {status}: "
                    f"{raw[:500].decode('utf-8', errors='replace')}"
                )
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except ValueError:
                return {"_raw": raw.decode("utf-8", errors="replace")}

        # setUserProperties / setPrivateKey are idempotent on the S3 side
        # ONLY when no UserData exists yet. Once the user has been bootstrapped
        # in this JVM (this same wheel call, an earlier wheel call, the
        # standalone S3 daemon, or external admin tooling), the gateway
        # refuses both PUTs with HTTP 400 unless `?password=<current>` is
        # supplied so it can validate the caller against the existing private
        # key (S3Controller.setUserProperties / setPrivateKey). Always send
        # the password as query — for fresh users it is ignored, for known
        # users it unblocks re-bootstrap with the same credentials.
        pw_q = "?password=" + urllib.parse.quote(pw, safe="")
        _put(f"/setUserProperties/{user_name}{pw_q}", user_properties)
        _put(f"/setPrivateKey/{user_name}{pw_q}", private_key_encrypted)
        body = _put(f"/setPassword/{user_name}", pw)

        access_key = body.get("accessKey")
        secret_key = body.get("secretKey")
        if not access_key or not secret_key:
            raise RuntimeError(
                "S3 gateway did not return accessKey/secretKey from "
                f"setPassword; response was: {body}"
            )

        creds = {
            "endpoint_url": endpoint,
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": region,
        }
        self._s3_credentials_cache[endpoint] = dict(creds)
        return creds

    def boto3_s3(self, **overrides):
        """Return a ready-to-use boto3 S3 client.

        Equivalent to::

            boto3.client('s3', **self.s3_credentials(), **overrides)

        Any keyword in ``overrides`` wins over the helper's defaults — use
        this to pass ``config=botocore.config.Config(...)``, override
        ``endpoint_url`` for a remote deployment, etc.

        Requires ``boto3`` to be installed in the environment; raises
        ``ImportError`` with a clear hint otherwise. ``boto3`` is not in
        ``install_requires`` because not every wheel consumer wants the AWS
        SDK on the import path.
        """
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "boto3 is required for AltaStataFunctions.boto3_s3(). "
                "Install it with `pip install boto3`."
            ) from e
        creds = self.s3_credentials()
        return boto3.client("s3", **{**creds, **overrides})

    def install_aws_env(
        self,
        *,
        password: Optional[str] = None,
        endpoint: Optional[str] = None,
        region: str = "us-east-1",
    ) -> Dict[str, str]:
        """Bootstrap S3 credentials and export them as ``AWS_*`` env vars.

        Sets four variables in ``os.environ`` so that subprocess shells
        (``!aws s3 ls``, ``!s3cmd``, etc.) and any AWS SDK that reads the
        ambient environment can see them without further configuration:

        - ``AWS_ACCESS_KEY_ID``
        - ``AWS_SECRET_ACCESS_KEY``
        - ``AWS_DEFAULT_REGION``
        - ``AWS_ENDPOINT_URL_S3`` (picked up by boto3 ≥ 1.30 and the
          ``aws`` CLI v2 via ``--endpoint-url`` shorthand)

        Returns:
            The dict that was applied to ``os.environ`` — handy for
            eval-exporting into a parent shell.
        """
        creds = self.s3_credentials(password=password, endpoint=endpoint, region=region)
        aws_env = {
            "AWS_ACCESS_KEY_ID": creds["aws_access_key_id"],
            "AWS_SECRET_ACCESS_KEY": creds["aws_secret_access_key"],
            "AWS_DEFAULT_REGION": creds["region_name"],
            "AWS_ENDPOINT_URL_S3": creds["endpoint_url"],
        }
        os.environ.update(aws_env)
        return aws_env

    def _resolve_s3_endpoint(self) -> str:
        """Best-effort default URL for the S3 gateway.

        - gRPC mode: same host as the gRPC target, port 9876.
        - Local fallback: ``http://127.0.0.1:9876``.
        """
        if self.grpc_client is not None:
            host = self.grpc_client.endpoint.host
            return f"http://{host}:9876"
        return "http://127.0.0.1:9876"

    def _read_bootstrap_material(self) -> Tuple[str, str, str]:
        """Resolve ``(user_name, user_properties, private_key_encrypted)``.

        For instances built via :meth:`from_account_dir`, reads the
        ``*user.properties`` file and ``private.key`` from disk on each call
        so updates to those files take effect on the next bootstrap.
        For :meth:`from_credentials` instances, returns the strings supplied
        at construction.
        """
        if self._account_dir_path is not None:
            props_path = None
            for fname in sorted(os.listdir(self._account_dir_path)):
                if fname.endswith("user.properties"):
                    props_path = os.path.join(self._account_dir_path, fname)
                    break
            if props_path is None:
                raise FileNotFoundError(
                    f"No *user.properties file in account dir {self._account_dir_path}"
                )
            with open(props_path, "r", encoding="utf-8") as f:
                user_properties = f.read()
            key_path = os.path.join(self._account_dir_path, "private.key")
            with open(key_path, "r", encoding="utf-8") as f:
                private_key_encrypted = f.read()
        elif self._user_properties is not None and self._private_key_encrypted is not None:
            user_properties = self._user_properties
            private_key_encrypted = self._private_key_encrypted
        else:
            raise RuntimeError(
                "S3 bootstrap material unavailable. Construct this "
                "AltaStataFunctions via from_account_dir(path) or "
                "from_credentials(user_properties, private_key)."
            )

        user_name = _parse_user_name_from_properties(user_properties)
        return user_name, user_properties, private_key_encrypted

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
        return self.grpc_client.create_file(cloud_file_path, buffer)

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
        return self.grpc_client.append_buffer_to_file(
            cloud_file_path, buffer, snapshot_time=snapshot_time
        )

    def store(self, localFilesOrDirectories: List[str], localFSPrefix: str, cloudPathPrefix: str, waitUntilDone: bool):
        return self.grpc_client.store(
            localFilesOrDirectories, localFSPrefix, cloudPathPrefix, waitUntilDone
        )

    def retrieve_files(self, output_dir, cloud_path_prefix, including_subdirectories, snapshot_time, is_streaming, wait_until_done):
        return self.grpc_client.retrieve_files(
            output_dir,
            cloud_path_prefix,
            including_subdirectories,
            snapshot_time,
            is_streaming,
            wait_until_done,
        )

    def delete_files(self, cloud_path_prefix, including_subdirectories, time_interval_start, time_interval_end):
        return self.grpc_client.delete_files(
            cloud_path_prefix,
            including_subdirectories=including_subdirectories,
            time_interval_start=time_interval_start,
            time_interval_end=time_interval_end,
        )

    def share_files(self, cloud_path_prefix: str, including_subdirectories: bool, time_interval_start: str, time_interval_end: str, users: list) -> list:
        return self.grpc_client.share_files(
            cloud_path_prefix, including_subdirectories, time_interval_start, time_interval_end, users
        )

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
        return self.grpc_client.revoke_reader_access(
            cloud_path_prefix,
            including_subdirectories,
            time_interval_start,
            time_interval_end,
            readers_to_revoke,
        )

    def list_cloud_files_versions(self, cloudPathPrefix, includingSubdirectories, timeIntervalStart, timeIntervalEnd):
        return self.grpc_client.list_cloud_files_versions(
            cloudPathPrefix,
            including_subdirectories=includingSubdirectories,
            time_interval_start=timeIntervalStart,
            time_interval_end=timeIntervalEnd,
        )

    def get_buffer(self, cloudFilePath, snapshotTime, startPosition, howManyChunksInParallel, size, trust_cached_size=False):
        """Read file content from cloud storage as ``bytes``.

        Args:
            cloudFilePath: Cloud file path (may include ✹ version suffix).
            snapshotTime: Version timestamp, or None for latest.
            startPosition: Byte offset to start reading from.
            howManyChunksInParallel: Number of chunks to download concurrently.
            size: Expected file size in bytes.
            trust_cached_size: When True, declare this file's content immutable
                (write-once) so the per-open fresh cloud GET of the ``size``
                attribute is skipped and the cached value is trusted. Big win
                for read-many workloads (ML dataset epochs); leave False (the
                default) for mutable files.

        Returns:
            bytes: File content.

        Strategy:
            Uses gRPC unary for small payloads and gRPC stream assembly for
            large/unknown payloads (implemented in AltaStataGrpcClient).
        """
        return self.grpc_client.get_buffer(
            cloudFilePath,
            size=size,
            snapshot_time=0 if snapshotTime is None else snapshotTime,
            start_position=startPosition,
            parallel_chunks=howManyChunksInParallel,
            trust_cached_size=trust_cached_size,
        )

    def get_java_input_stream(self, cloud_file_path, snapshot_time, start_position, how_many_chunks_in_parallel):
        """Open a stream iterator for the given cloud file."""
        return self.grpc_client.get_java_input_stream(
            cloud_file_path, snapshot_time, start_position, how_many_chunks_in_parallel,
        )

    def get_file_attribute(self, cloud_file_path, snapshot_time, name):
        """
        Get file attribute from Altastata file system.
        """
        return self.grpc_client.get_file_attribute(cloud_file_path, snapshot_time, name)

    def copy_file(self, from_cloud_file_path: str, to_cloud_file_path: str):
        """
        Copy a file from one cloud path to another.
        
        Args:
            from_cloud_file_path (str): The source file path on the cloud
            to_cloud_file_path (str): The destination file path on the cloud
            
        Returns:
            CloudFileOperationStatus: Status of the copy operation
        """
        return self.grpc_client.copy_file(from_cloud_file_path, to_cloud_file_path)

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
        return self.grpc_client.add_event_listener(callback)
    
    def remove_event_listener(self, listener: AltaStataEventListener):
        """
        Remove a previously registered event listener.
        
        Args:
            listener: The listener object returned by add_event_listener()
        """
        return self.grpc_client.remove_event_listener(listener)
    
    def remove_all_event_listeners(self):
        """
        Remove all registered event listeners.
        """
        return self.grpc_client.remove_all_event_listeners()

    def shutdown(self):
        if self.grpc_client is not None:
            self.grpc_client.close()



