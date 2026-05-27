import sys
import types
import unittest
from unittest.mock import patch
import tempfile
import os


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

    def test_infer_user_name_from_properties_filename(self):
        from altastata.grpc_client import _infer_user_name
        user = _infer_user_name(
            "/tmp/amazon.rsa.bob123",
            "/tmp/amazon.rsa.bob123/altastata-myorgrsa444-bob123.user.properties",
        )
        self.assertEqual("bob123", user)

    def test_infer_user_name_from_properties_text(self):
        from altastata.grpc_client import _infer_user_name_from_properties_text
        props = """
        # comment
        region=us-east-1
        myuser=bob123
        accounttype=amazon-s3-secure
        """
        self.assertEqual("bob123", _infer_user_name_from_properties_text(props))

    @patch("altastata.grpc_client.grpc.insecure_channel")
    @patch("altastata.grpc_client._bootstrap_via_grpc")
    @patch("altastata.grpc_client._wait_for_port")
    @patch("altastata.grpc_client._start_local_grpc_service")
    @patch("altastata.grpc_client._is_port_open")
    def test_from_account_dir_starts_server_when_ports_down(
        self,
        mock_is_port_open,
        mock_start_server,
        _mock_wait_for_port,
        _mock_bootstrap,
        mock_insecure_channel,
    ):
        mock_insecure_channel.return_value = object()
        mock_is_port_open.return_value = False
        mock_start_server.return_value = object()

        with tempfile.TemporaryDirectory() as td:
            up = os.path.join(td, "altastata-org-bob123.user.properties")
            pk = os.path.join(td, "private.key")
            with open(up, "w", encoding="utf-8") as f:
                f.write("user.properties")
            with open(pk, "w", encoding="utf-8") as f:
                f.write("private.key")

            from altastata.grpc_client import AltaStataGrpcClient, GrpcEndpoint
            AltaStataGrpcClient.from_account_dir(
                td,
                password="123",
                endpoint=GrpcEndpoint(host="127.0.0.1", port=9877, secure=False),
                setup_port=9880,
                auto_start_server=True,
                grpc_server_command=["echo", "start"],
                grpc_server_working_dir=td,
            )
            mock_start_server.assert_called_once()

    @patch("altastata.grpc_client.grpc.insecure_channel")
    @patch("altastata.grpc_client._bootstrap_via_grpc")
    @patch("altastata.grpc_client._wait_for_port")
    @patch("altastata.grpc_client._start_local_grpc_service")
    @patch("altastata.grpc_client._is_port_open")
    def test_from_credentials_starts_server_when_ports_down(
        self,
        mock_is_port_open,
        mock_start_server,
        _mock_wait_for_port,
        _mock_bootstrap,
        mock_insecure_channel,
    ):
        mock_insecure_channel.return_value = object()
        mock_is_port_open.return_value = False
        mock_start_server.return_value = object()

        from altastata.grpc_client import AltaStataGrpcClient, GrpcEndpoint
        AltaStataGrpcClient.from_credentials(
            user_properties="myuser=bob123\nregion=us-east-1\n",
            private_key_encrypted="-----BEGIN RSA PRIVATE KEY-----\n...\n",
            password="123",
            endpoint=GrpcEndpoint(host="127.0.0.1", port=9877, secure=False),
            setup_port=9880,
            auto_start_server=True,
            grpc_server_command=["echo", "start"],
            grpc_server_working_dir="/tmp",
        )
        mock_start_server.assert_called_once()


if __name__ == "__main__":
    unittest.main()
