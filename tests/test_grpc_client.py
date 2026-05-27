import sys
import types
import unittest
from unittest.mock import patch


class DummyUsersStub:
    def __init__(self, channel):
        self._channel = channel


class DummySharingStub:
    def __init__(self, channel):
        self._channel = channel


class DummyAttributesStub:
    def __init__(self, channel):
        self._channel = channel


class DummyFileOpsStub:
    def __init__(self, channel):
        self._channel = channel

class DummyEventsStub:
    def __init__(self, channel):
        self._channel = channel


class GrpcClientTests(unittest.TestCase):
    def setUp(self):
        # Create fake generated modules so client can be instantiated in unit tests.
        pkg = types.ModuleType("altastata.v1")
        users_pb2 = types.ModuleType("users_pb2")
        sharing_pb2 = types.ModuleType("sharing_pb2")
        attributes_pb2 = types.ModuleType("attributes_pb2")
        fileops_pb2 = types.ModuleType("fileops_pb2")
        events_pb2 = types.ModuleType("events_pb2")
        users_pb2_grpc = types.ModuleType("users_pb2_grpc")
        sharing_pb2_grpc = types.ModuleType("sharing_pb2_grpc")
        attributes_pb2_grpc = types.ModuleType("attributes_pb2_grpc")
        fileops_pb2_grpc = types.ModuleType("fileops_pb2_grpc")
        events_pb2_grpc = types.ModuleType("events_pb2_grpc")

        users_pb2_grpc.UsersServiceStub = DummyUsersStub
        sharing_pb2_grpc.SharingServiceStub = DummySharingStub
        attributes_pb2_grpc.AttributesServiceStub = DummyAttributesStub
        fileops_pb2_grpc.FileOpsServiceStub = DummyFileOpsStub
        events_pb2_grpc.EventsServiceStub = DummyEventsStub

        sys.modules["altastata.v1"] = pkg
        sys.modules["altastata.v1.users_pb2"] = users_pb2
        sys.modules["altastata.v1.sharing_pb2"] = sharing_pb2
        sys.modules["altastata.v1.attributes_pb2"] = attributes_pb2
        sys.modules["altastata.v1.fileops_pb2"] = fileops_pb2
        sys.modules["altastata.v1.events_pb2"] = events_pb2
        sys.modules["altastata.v1.users_pb2_grpc"] = users_pb2_grpc
        sys.modules["altastata.v1.sharing_pb2_grpc"] = sharing_pb2_grpc
        sys.modules["altastata.v1.attributes_pb2_grpc"] = attributes_pb2_grpc
        sys.modules["altastata.v1.fileops_pb2_grpc"] = fileops_pb2_grpc
        sys.modules["altastata.v1.events_pb2_grpc"] = events_pb2_grpc

    def tearDown(self):
        for name in list(sys.modules.keys()):
            if name.startswith("altastata.v1"):
                sys.modules.pop(name, None)

    def test_token_from_local_user(self):
        from altastata.grpc_client import _token_from_params
        token = _token_from_params(None, "alice", None)
        self.assertEqual("local-alice", token)

    @patch("altastata.grpc_client.grpc.insecure_channel")
    def test_client_builds_bearer_metadata(self, mock_insecure_channel):
        mock_insecure_channel.return_value = object()

        from altastata.grpc_client import AltaStataGrpcClient, GrpcEndpoint
        client = AltaStataGrpcClient(
            endpoint=GrpcEndpoint(host="127.0.0.1", port=9877, secure=False),
            local_user="alice",
        )
        self.assertEqual([("authorization", "Bearer local-alice")], client._metadata)


if __name__ == "__main__":
    unittest.main()
