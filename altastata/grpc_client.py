from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import grpc
import threading


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
