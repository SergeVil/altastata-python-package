from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import grpc
import threading
import os
import re
import socket
import subprocess
import time
import pkg_resources


def _token_from_params(
    bearer_token: Optional[str],
    local_user: Optional[str],
    access_key: Optional[str],
) -> str:
    if bearer_token:
        return bearer_token
    if local_user:
        return f"local-{local_user}"
    if access_key:
        return f"access-{access_key}"
    raise ValueError("Provide one of: bearer_token, local_user, access_key")


@dataclass
class GrpcEndpoint:
    host: str = "127.0.0.1"
    port: int = 9877
    secure: bool = False

    @property
    def target(self) -> str:
        return f"{self.host}:{self.port}"


class AltaStataGrpcClient:
    """
    gRPC client for AltaStata APIs.
    """

    def __init__(
        self,
        endpoint: GrpcEndpoint = GrpcEndpoint(),
        *,
        bearer_token: Optional[str] = None,
        local_user: Optional[str] = None,
        access_key: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self._token = _token_from_params(bearer_token, local_user, access_key)
        self._metadata: List[Tuple[str, str]] = [("authorization", f"Bearer {self._token}")]
        self._channel = self._create_channel(endpoint)

        # Lazy import after channel creation for clearer error messaging.
        try:
            from .v1 import attributes_pb2, attributes_pb2_grpc
            from .v1 import sharing_pb2, sharing_pb2_grpc
            from .v1 import users_pb2, users_pb2_grpc
            from .v1 import fileops_pb2, fileops_pb2_grpc
            from .v1 import events_pb2, events_pb2_grpc
        except Exception as exc:
            raise ImportError(
                "gRPC stubs are missing. Run: python scripts/generate_grpc_stubs.py"
            ) from exc

        self._users_pb2 = users_pb2
        self._sharing_pb2 = sharing_pb2
        self._attributes_pb2 = attributes_pb2
        self._fileops_pb2 = fileops_pb2
        self._events_pb2 = events_pb2

        self._users_stub = users_pb2_grpc.UsersServiceStub(self._channel)
        self._sharing_stub = sharing_pb2_grpc.SharingServiceStub(self._channel)
        self._attributes_stub = attributes_pb2_grpc.AttributesServiceStub(self._channel)
        self._fileops_stub = fileops_pb2_grpc.FileOpsServiceStub(self._channel)
        self._events_stub = events_pb2_grpc.EventsServiceStub(self._channel)
        self._listener_threads: List[threading.Thread] = []
        self._server_process = None

    @classmethod
    def from_account_dir(
        cls,
        account_dir_path: str,
        *,
        password: Optional[str] = None,
        user_name: Optional[str] = None,
        endpoint: GrpcEndpoint = GrpcEndpoint(),
        setup_port: int = 9880,
        auto_start_server: bool = True,
        grpc_server_command: Optional[Sequence[str]] = None,
        grpc_server_working_dir: Optional[str] = None,
        start_timeout_s: int = 45,
    ) -> "AltaStataGrpcClient":
        """
        Create a gRPC client from an AltaStata account directory.

        This method mirrors the convenience of AltaStataFunctions.from_account_dir:
        it reads account files and initializes the local gRPC service user context.
        """
        account_dir = os.path.abspath(account_dir_path)
        if not os.path.isdir(account_dir):
            raise FileNotFoundError(f"Account directory not found: {account_dir}")

        user_properties_path = _find_user_properties_file(account_dir)
        private_key_path = os.path.join(account_dir, "private.key")
        if not os.path.exists(private_key_path):
            raise FileNotFoundError(f"private.key not found in account directory: {account_dir}")

        with open(user_properties_path, "r", encoding="utf-8") as f:
            user_properties = f.read()
        with open(private_key_path, "r", encoding="utf-8") as f:
            private_key = f.read()

        resolved_user_name = user_name or _infer_user_name(account_dir, user_properties_path)

        started_process = None
        if not _is_port_open(endpoint.host, endpoint.port) and auto_start_server:
            started_process = _start_local_grpc_service(
                grpc_server_command=grpc_server_command,
                working_dir=grpc_server_working_dir,
            )
            _wait_for_port(endpoint.host, endpoint.port, timeout_s=start_timeout_s)
        _bootstrap_via_grpc(
            endpoint=endpoint,
            user_name=resolved_user_name,
            user_properties=user_properties,
            private_key_encrypted=private_key,
            password=password,
        )

        client = cls(endpoint=endpoint, local_user=resolved_user_name)
        client._server_process = started_process
        return client

    @classmethod
    def from_credentials(
        cls,
        user_properties: str,
        private_key_encrypted: str,
        *,
        password: Optional[str] = None,
        user_name: Optional[str] = None,
        endpoint: GrpcEndpoint = GrpcEndpoint(),
        setup_port: int = 9880,
        auto_start_server: bool = True,
        grpc_server_command: Optional[Sequence[str]] = None,
        grpc_server_working_dir: Optional[str] = None,
        start_timeout_s: int = 45,
    ) -> "AltaStataGrpcClient":
        """
        Create a gRPC client directly from credentials payloads.

        Mirrors AltaStataFunctions.from_credentials(...) convenience.
        """
        resolved_user_name = user_name or _infer_user_name_from_properties_text(user_properties)

        started_process = None
        if not _is_port_open(endpoint.host, endpoint.port) and auto_start_server:
            started_process = _start_local_grpc_service(
                grpc_server_command=grpc_server_command,
                working_dir=grpc_server_working_dir,
            )
            _wait_for_port(endpoint.host, endpoint.port, timeout_s=start_timeout_s)
        _bootstrap_via_grpc(
            endpoint=endpoint,
            user_name=resolved_user_name,
            user_properties=user_properties,
            private_key_encrypted=private_key_encrypted,
            password=password,
        )

        client = cls(endpoint=endpoint, local_user=resolved_user_name)
        client._server_process = started_process
        return client

    @staticmethod
    def _create_channel(endpoint: GrpcEndpoint) -> grpc.Channel:
        options = [
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ]
        if endpoint.secure:
            creds = grpc.ssl_channel_credentials()
            return grpc.secure_channel(endpoint.target, creds, options=options)
        return grpc.insecure_channel(endpoint.target, options=options)

    def close(self) -> None:
        self._channel.close()
        if self._server_process is not None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
            except Exception:
                pass

    # ---------------- Users ----------------

    def list_users(self) -> List[Dict[str, object]]:
        req = self._users_pb2.Empty()
        out: List[Dict[str, object]] = []
        for item in self._users_stub.ListUsers(req, metadata=self._metadata):
            out.append({"user_name": item.user_name, "initialized": item.initialized})
        return out

    def get_user(self, user_name: str) -> Dict[str, object]:
        req = self._users_pb2.GetUserRequest(user_name=user_name)
        resp = self._users_stub.GetUser(req, metadata=self._metadata)
        return {
            "user_name": resp.user_name,
            "initialized": resp.initialized,
            "access_key": resp.access_key,
        }

    def get_my_account(self) -> Dict[str, object]:
        req = self._users_pb2.GetMyAccountRequest()
        resp = self._users_stub.GetMyAccount(req, metadata=self._metadata)
        return {
            "user_name": resp.user_name,
            "initialized": resp.initialized,
            "access_key": resp.access_key,
        }

    # ---------------- Sharing ----------------

    def share(self, file_paths: Sequence[str], readers: Sequence[str]) -> List[Dict[str, str]]:
        req = self._sharing_pb2.ShareRequest(file_paths=list(file_paths), readers=list(readers))
        resp = self._sharing_stub.Share(req, metadata=self._metadata)
        return [self._status_to_dict(s) for s in resp.statuses]

    def revoke(self, file_paths: Sequence[str], readers: Sequence[str]) -> List[Dict[str, str]]:
        req = self._sharing_pb2.RevokeRequest(file_paths=list(file_paths), readers=list(readers))
        resp = self._sharing_stub.Revoke(req, metadata=self._metadata)
        return [self._status_to_dict(s) for s in resp.statuses]

    @staticmethod
    def _status_to_dict(status) -> Dict[str, str]:
        return {
            "file_path": status.file_path,
            "operation_state": status.operation_state,
            "error": status.error,
        }

    # ---------------- Attributes ----------------

    def get_attribute(self, file_path: str, name: str, snapshot_time: int = 0) -> Dict[str, str]:
        req = self._attributes_pb2.GetAttributeRequest(
            file_path=file_path,
            snapshot_time=snapshot_time,
            name=name,
        )
        resp = self._attributes_stub.GetAttribute(req, metadata=self._metadata)
        return {"name": resp.name, "value": resp.value}

    def get_attributes(
        self,
        file_path: str,
        names: Sequence[str],
        snapshot_time: int = 0,
    ) -> Dict[str, str]:
        req = self._attributes_pb2.GetAttributesRequest(
            file_path=file_path,
            snapshot_time=snapshot_time,
            names=list(names),
        )
        resp = self._attributes_stub.GetAttributes(req, metadata=self._metadata)
        return dict(resp.attributes)

    def set_attribute(
        self,
        file_path: str,
        name: str,
        value: str,
        snapshot_time: int = 0,
    ) -> None:
        req = self._attributes_pb2.SetAttributeRequest(
            file_path=file_path,
            snapshot_time=snapshot_time,
            name=name,
            value=value,
        )
        self._attributes_stub.SetAttribute(req, metadata=self._metadata)

    def delete_attribute(self, file_path: str, name: str, snapshot_time: int = 0) -> None:
        req = self._attributes_pb2.DeleteAttributeRequest(
            file_path=file_path,
            snapshot_time=snapshot_time,
            name=name,
        )
        self._attributes_stub.DeleteAttribute(req, metadata=self._metadata)

    # ---------------- FileOps ----------------

    def create_file(self, file_path: str, content: bytes) -> Dict[str, str]:
        req = self._fileops_pb2.CreateFileRequest(file_path=file_path, content=content)
        resp = self._fileops_stub.CreateFile(req, metadata=self._metadata)
        return self._status_to_dict(resp.status)

    def get_buffer(
        self,
        file_path: str,
        size: int,
        snapshot_time: int = 0,
        start_position: int = 0,
        parallel_chunks: int = 4,
    ) -> bytes:
        req = self._fileops_pb2.GetBufferRequest(
            file_path=file_path,
            snapshot_time=snapshot_time,
            start_position=start_position,
            parallel_chunks=parallel_chunks,
            size=size,
        )
        resp = self._fileops_stub.GetBuffer(req, metadata=self._metadata)
        return bytes(resp.data)

    def delete_files(
        self,
        cloud_path_prefix: str,
        including_subdirectories: bool = True,
        time_interval_start: Optional[str] = None,
        time_interval_end: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        req = self._fileops_pb2.DeleteRequest(
            cloud_path_prefix=cloud_path_prefix,
            including_subdirectories=including_subdirectories,
            time_interval_start=time_interval_start or "",
            time_interval_end=time_interval_end or "",
        )
        resp = self._fileops_stub.Delete(req, metadata=self._metadata)
        return [self._status_to_dict(s) for s in resp.statuses]

    def list_cloud_files_versions(
        self,
        cloud_path_prefix: str,
        including_subdirectories: bool = True,
        time_interval_start: Optional[str] = None,
        time_interval_end: Optional[str] = None,
    ) -> List[List[str]]:
        req = self._fileops_pb2.ListVersionsRequest(
            cloud_path_prefix=cloud_path_prefix,
            including_subdirectories=including_subdirectories,
            time_interval_start=time_interval_start or "",
            time_interval_end=time_interval_end or "",
        )
        out: List[List[str]] = []
        for entry in self._fileops_stub.ListVersions(req, metadata=self._metadata):
            out.append(list(entry.versions))
        return out

    def append_buffer_to_file(
        self,
        cloud_file_path: str,
        buffer: bytes,
        snapshot_time: Optional[int] = None,
    ) -> bool:
        req = self._fileops_pb2.AppendBufferToFileRequest(
            file_path=cloud_file_path,
            snapshot_time=0 if snapshot_time is None else snapshot_time,
            content=buffer,
        )
        resp = self._fileops_stub.AppendBufferToFile(req, metadata=self._metadata)
        return bool(resp.success)

    def store(
        self,
        local_files_or_directories: Sequence[str],
        local_fs_prefix: str,
        cloud_path_prefix: str,
        wait_until_done: bool,
    ) -> List[Dict[str, str]]:
        req = self._fileops_pb2.StoreRequest(
            local_files_or_directories=list(local_files_or_directories),
            local_fs_prefix=local_fs_prefix,
            cloud_path_prefix=cloud_path_prefix,
            wait_until_done=wait_until_done,
        )
        resp = self._fileops_stub.Store(req, metadata=self._metadata)
        return [self._status_to_dict(s) for s in resp.statuses]

    def retrieve_files(
        self,
        output_dir: str,
        cloud_path_prefix: str,
        including_subdirectories: bool,
        snapshot_time: Optional[int],
        is_streaming: bool,
        wait_until_done: bool,
    ) -> List[Dict[str, str]]:
        req = self._fileops_pb2.RetrieveRequest(
            output_dir=output_dir,
            cloud_path_prefix=cloud_path_prefix,
            including_subdirectories=including_subdirectories,
            snapshot_time=0 if snapshot_time is None else snapshot_time,
            is_streaming=is_streaming,
            wait_until_done=wait_until_done,
        )
        resp = self._fileops_stub.Retrieve(req, metadata=self._metadata)
        return [self._status_to_dict(s) for s in resp.statuses]

    def copy_file(self, from_cloud_file_path: str, to_cloud_file_path: str) -> Dict[str, str]:
        req = self._fileops_pb2.CopyFileRequest(
            from_cloud_file_path=from_cloud_file_path,
            to_cloud_file_path=to_cloud_file_path,
        )
        resp = self._fileops_stub.CopyFile(req, metadata=self._metadata)
        return self._status_to_dict(resp.status)

    def read_stream(
        self,
        cloud_file_path: str,
        snapshot_time: Optional[int] = None,
        start_position: int = 0,
        parallel_chunks: int = 4,
        chunk_size: int = 8 * 1024 * 1024,
    ):
        req = self._fileops_pb2.ReadStreamRequest(
            file_path=cloud_file_path,
            snapshot_time=0 if snapshot_time is None else snapshot_time,
            start_position=start_position,
            parallel_chunks=parallel_chunks,
            chunk_size=chunk_size,
        )
        for chunk in self._fileops_stub.ReadStream(req, metadata=self._metadata):
            yield bytes(chunk.data)

    # ----- Py4J-compat aliases -----

    def set_password(self, account_password: str):
        req = self._users_pb2.SetPasswordRequest(account_password=account_password)
        resp = self._users_stub.SetPassword(req, metadata=self._metadata)
        return bool(resp.success)

    def share_files(
        self,
        cloud_path_prefix: str,
        including_subdirectories: bool,
        time_interval_start: Optional[str],
        time_interval_end: Optional[str],
        users: Sequence[str],
    ) -> List[Dict[str, str]]:
        req = self._sharing_pb2.ShareByQueryRequest(
            cloud_path_prefix=cloud_path_prefix,
            including_subdirectories=including_subdirectories,
            time_interval_start=time_interval_start or "",
            time_interval_end=time_interval_end or "",
            readers=list(users),
        )
        resp = self._sharing_stub.ShareByQuery(req, metadata=self._metadata)
        return [self._status_to_dict(s) for s in resp.statuses]

    def revoke_reader_access(
        self,
        cloud_path_prefix: str,
        including_subdirectories: bool,
        time_interval_start: Optional[str],
        time_interval_end: Optional[str],
        readers_to_revoke: Sequence[str],
    ) -> List[Dict[str, str]]:
        req = self._sharing_pb2.RevokeByQueryRequest(
            cloud_path_prefix=cloud_path_prefix,
            including_subdirectories=including_subdirectories,
            time_interval_start=time_interval_start or "",
            time_interval_end=time_interval_end or "",
            readers=list(readers_to_revoke),
        )
        resp = self._sharing_stub.RevokeByQuery(req, metadata=self._metadata)
        return [self._status_to_dict(s) for s in resp.statuses]

    def get_file_attribute(
        self,
        cloud_file_path: str,
        snapshot_time: Optional[int],
        name: str,
    ) -> Optional[str]:
        try:
            result = self.get_attribute(cloud_file_path, name, 0 if snapshot_time is None else snapshot_time)
            return result.get("value")
        except grpc.RpcError:
            return None

    def get_java_input_stream(
        self,
        cloud_file_path: str,
        snapshot_time: Optional[int],
        start_position: int,
        how_many_chunks_in_parallel: int,
    ):
        return self.read_stream(
            cloud_file_path=cloud_file_path,
            snapshot_time=snapshot_time,
            start_position=start_position,
            parallel_chunks=how_many_chunks_in_parallel,
        )

    def subscribe_events(self):
        req = self._events_pb2.SubscribeRequest()
        for event in self._events_stub.Subscribe(req, metadata=self._metadata):
            yield {"event_name": event.event_name, "data": event.data}

    def add_event_listener(self, callback):
        """
        gRPC replacement for Py4J add_event_listener.
        Starts a daemon thread that consumes Subscribe stream and invokes callback.
        """
        stop_flag = {"stop": False}
        call_ref = {"call": None}

        def _worker():
            try:
                req = self._events_pb2.SubscribeRequest()
                call = self._events_stub.Subscribe(req, metadata=self._metadata)
                call_ref["call"] = call
                for event in call:
                    if stop_flag["stop"]:
                        break
                    callback(event.event_name, event.data)
            except Exception:
                # Match Py4J behavior: do not crash caller on background listener errors.
                pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        handle = {"thread": t, "stop": stop_flag, "call_ref": call_ref}
        self._listener_threads.append(t)
        return handle

    def remove_event_listener(self, listener):
        if isinstance(listener, dict) and "stop" in listener:
            listener["stop"]["stop"] = True
            call_ref = listener.get("call_ref")
            if isinstance(call_ref, dict):
                call = call_ref.get("call")
                if call is not None:
                    call.cancel()

    def remove_all_event_listeners(self):
        # Cooperative stop; stream closes naturally when call finishes/cancelled.
        for t in self._listener_threads:
            # best-effort no-op; handles returned by add_event_listener own stop flags
            _ = t
        self._listener_threads.clear()


def _find_user_properties_file(account_dir: str) -> str:
    candidates = [
        os.path.join(account_dir, name)
        for name in os.listdir(account_dir)
        if name.endswith(".user.properties")
    ]
    if not candidates:
        raise FileNotFoundError(f"No *.user.properties file found in: {account_dir}")
    if len(candidates) > 1:
        candidates.sort()
    return candidates[0]


def _infer_user_name(account_dir: str, user_properties_path: str) -> str:
    file_name = os.path.basename(user_properties_path)
    m = re.match(r".+-(?P<user>[^-]+)\.user\.properties$", file_name)
    if m:
        return m.group("user")

    dir_name = os.path.basename(account_dir)
    if "." in dir_name:
        maybe_user = dir_name.split(".")[-1]
        if maybe_user:
            return maybe_user
    if dir_name:
        return dir_name
    raise ValueError("Unable to infer user name from account directory")


def _bootstrap_via_grpc(
    endpoint: GrpcEndpoint,
    user_name: str,
    user_properties: str,
    private_key_encrypted: str,
    password: Optional[str],
) -> None:
    try:
        from .v1 import users_pb2, users_pb2_grpc
    except Exception as exc:
        raise ImportError(
            "gRPC stubs are missing. Run: python scripts/generate_grpc_stubs.py"
        ) from exc

    channel = AltaStataGrpcClient._create_channel(endpoint)
    try:
        users = users_pb2_grpc.UsersServiceStub(channel)
        users.SetUserProperties(users_pb2.SetUserPropertiesRequest(
            user_name=user_name,
            user_properties=user_properties,
        ))
        users.SetPrivateKey(users_pb2.SetPrivateKeyRequest(
            user_name=user_name,
            private_key_encrypted=private_key_encrypted,
        ))
        if password is not None:
            users.SetPasswordForUser(users_pb2.SetPasswordForUserRequest(
                user_name=user_name,
                account_password=password,
            ))
    finally:
        channel.close()


def _infer_user_name_from_properties_text(user_properties: str) -> str:
    for raw_line in user_properties.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip().lower()
        val = v.strip()
        if key in ("myuser", "user", "username") and val:
            return val
    raise ValueError("Unable to infer user_name from user_properties; pass user_name explicitly.")


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait_for_port(host: str, port: int, timeout_s: int = 45) -> None:
    started = time.time()
    while time.time() - started < timeout_s:
        if _is_port_open(host, port):
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def _start_local_grpc_service(
    grpc_server_command: Optional[Sequence[str]] = None,
    working_dir: Optional[str] = None,
):
    resolved_command, resolved_working_dir = _resolve_local_grpc_startup_command(
        grpc_server_command=grpc_server_command,
        working_dir=working_dir,
    )
    return subprocess.Popen(
        list(resolved_command),
        cwd=resolved_working_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _resolve_local_grpc_startup_command(
    grpc_server_command: Optional[Sequence[str]] = None,
    working_dir: Optional[str] = None,
) -> Tuple[Sequence[str], Optional[str]]:
    resolved_working_dir = working_dir
    if grpc_server_command is None:
        bundled_uber_jar = _find_bundled_grpc_uber_jar()
        if bundled_uber_jar is not None:
            grpc_server_command = ["java", "-cp", bundled_uber_jar, "com.altastata.grpc.GrpcApplication"]
            if resolved_working_dir is None:
                resolved_working_dir = os.path.dirname(bundled_uber_jar)
        else:
            grpc_server_command = ["./gradlew", ":altastata-grpc:run"]
            if resolved_working_dir is None:
                resolved_working_dir = _default_mycloud_dir()

    if resolved_working_dir is None and grpc_server_command[:2] == ["./gradlew", ":altastata-grpc:run"]:
        raise RuntimeError(
            "Unable to locate bundled altastata-grpc runtime jar and unable to determine mycloud "
            "directory for Gradle fallback. Package altastata-grpc-*-uber.jar under altastata/lib, "
            "or pass grpc_server_command/grpc_server_working_dir, or set ALTASTATA_MYCLOUD_DIR."
        )
    return list(grpc_server_command), resolved_working_dir


def _default_mycloud_dir() -> Optional[str]:
    env_dir = os.environ.get("ALTASTATA_MYCLOUD_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    repo_candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mycloud"))
    if os.path.isdir(repo_candidate):
        return repo_candidate
    return None


def _find_bundled_grpc_uber_jar() -> Optional[str]:
    try:
        jar_dir = pkg_resources.resource_filename("altastata", "lib")
    except Exception:
        return None
    if not os.path.isdir(jar_dir):
        return None

    candidates = sorted(
        os.path.join(jar_dir, f)
        for f in os.listdir(jar_dir)
        if f.startswith("altastata-grpc-") and f.endswith("-uber.jar")
    )
    if not candidates:
        return None
    return candidates[-1]
