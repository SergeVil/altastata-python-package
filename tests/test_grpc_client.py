import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


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


class DummyAuthStub:
    def __init__(self, channel):
        self._channel = channel


class GrpcClientTests(unittest.TestCase):
    def setUp(self):
        # Create fake generated modules so client can be instantiated in unit tests.
        pkg = types.ModuleType("altastata.v1")
        auth_pb2 = types.ModuleType("auth_pb2")
        users_pb2 = types.ModuleType("users_pb2")
        sharing_pb2 = types.ModuleType("sharing_pb2")
        attributes_pb2 = types.ModuleType("attributes_pb2")
        fileops_pb2 = types.ModuleType("fileops_pb2")
        events_pb2 = types.ModuleType("events_pb2")
        auth_pb2_grpc = types.ModuleType("auth_pb2_grpc")
        users_pb2_grpc = types.ModuleType("users_pb2_grpc")
        sharing_pb2_grpc = types.ModuleType("sharing_pb2_grpc")
        attributes_pb2_grpc = types.ModuleType("attributes_pb2_grpc")
        fileops_pb2_grpc = types.ModuleType("fileops_pb2_grpc")
        events_pb2_grpc = types.ModuleType("events_pb2_grpc")

        auth_pb2_grpc.AuthServiceStub = DummyAuthStub
        users_pb2_grpc.UsersServiceStub = DummyUsersStub
        sharing_pb2_grpc.SharingServiceStub = DummySharingStub
        attributes_pb2_grpc.AttributesServiceStub = DummyAttributesStub
        fileops_pb2_grpc.FileOpsServiceStub = DummyFileOpsStub
        events_pb2_grpc.EventsServiceStub = DummyEventsStub

        sys.modules["altastata.v1"] = pkg
        sys.modules["altastata.v1.auth_pb2"] = auth_pb2
        sys.modules["altastata.v1.users_pb2"] = users_pb2
        sys.modules["altastata.v1.sharing_pb2"] = sharing_pb2
        sys.modules["altastata.v1.attributes_pb2"] = attributes_pb2
        sys.modules["altastata.v1.fileops_pb2"] = fileops_pb2
        sys.modules["altastata.v1.events_pb2"] = events_pb2
        sys.modules["altastata.v1.auth_pb2_grpc"] = auth_pb2_grpc
        sys.modules["altastata.v1.users_pb2_grpc"] = users_pb2_grpc
        sys.modules["altastata.v1.sharing_pb2_grpc"] = sharing_pb2_grpc
        sys.modules["altastata.v1.attributes_pb2_grpc"] = attributes_pb2_grpc
        sys.modules["altastata.v1.fileops_pb2_grpc"] = fileops_pb2_grpc
        sys.modules["altastata.v1.events_pb2_grpc"] = events_pb2_grpc

    def tearDown(self):
        for name in list(sys.modules.keys()):
            if name.startswith("altastata.v1"):
                sys.modules.pop(name, None)

    @patch("altastata.grpc_client.grpc.insecure_channel")
    def test_client_builds_bearer_metadata_from_session_token(self, mock_insecure_channel):
        mock_insecure_channel.return_value = object()

        from altastata.grpc_client import AltaStataGrpcClient, GrpcEndpoint
        client = AltaStataGrpcClient(
            endpoint=GrpcEndpoint(host="127.0.0.1", port=9877, secure=False),
            bearer_token="sess-abc123",
            user_name="alice",
        )
        self.assertEqual([("authorization", "Bearer sess-abc123")], client._metadata)

    def test_client_rejects_construction_without_token(self):
        from altastata.grpc_client import AltaStataGrpcClient, GrpcEndpoint
        with self.assertRaises(ValueError):
            AltaStataGrpcClient(
                endpoint=GrpcEndpoint(host="127.0.0.1", port=9877, secure=False),
                bearer_token="",
            )

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
    @patch("altastata.grpc_client._bootstrap_and_login")
    @patch("altastata.grpc_client._wait_for_port")
    @patch("altastata.grpc_client._start_local_grpc_service")
    @patch("altastata.grpc_client._is_port_open")
    def test_from_account_dir_starts_server_when_ports_down(
        self,
        mock_is_port_open,
        mock_start_server,
        _mock_wait_for_port,
        mock_bootstrap_and_login,
        mock_insecure_channel,
    ):
        mock_insecure_channel.return_value = object()
        mock_is_port_open.return_value = False
        mock_start_server.return_value = object()
        mock_bootstrap_and_login.return_value = "sess-test-token"

        with tempfile.TemporaryDirectory() as td:
            up = os.path.join(td, "altastata-org-bob123.user.properties")
            pk = os.path.join(td, "private.key")
            with open(up, "w", encoding="utf-8") as f:
                f.write("user.properties")
            with open(pk, "w", encoding="utf-8") as f:
                f.write("private.key")

            from altastata.grpc_client import AltaStataGrpcClient, GrpcEndpoint
            client = AltaStataGrpcClient.from_account_dir(
                td,
                password="123",
                endpoint=GrpcEndpoint(host="127.0.0.1", port=9877, secure=False),
                auto_start_server=True,
                grpc_server_command=["echo", "start"],
                grpc_server_working_dir=td,
            )
            mock_start_server.assert_called_once()
            mock_bootstrap_and_login.assert_called_once()
            self.assertEqual(
                [("authorization", "Bearer sess-test-token")],
                client._metadata,
            )

    @patch("altastata.grpc_client.grpc.insecure_channel")
    @patch("altastata.grpc_client._bootstrap_and_login")
    @patch("altastata.grpc_client._wait_for_port")
    @patch("altastata.grpc_client._start_local_grpc_service")
    @patch("altastata.grpc_client._is_port_open")
    def test_from_credentials_starts_server_when_ports_down(
        self,
        mock_is_port_open,
        mock_start_server,
        _mock_wait_for_port,
        mock_bootstrap_and_login,
        mock_insecure_channel,
    ):
        mock_insecure_channel.return_value = object()
        mock_is_port_open.return_value = False
        mock_start_server.return_value = object()
        mock_bootstrap_and_login.return_value = "sess-test-token"

        from altastata.grpc_client import AltaStataGrpcClient, GrpcEndpoint
        AltaStataGrpcClient.from_credentials(
            user_properties="myuser=bob123\nregion=us-east-1\n",
            private_key_encrypted="-----BEGIN RSA PRIVATE KEY-----\n...\n",
            password="123",
            endpoint=GrpcEndpoint(host="127.0.0.1", port=9877, secure=False),
            auto_start_server=True,
            grpc_server_command=["echo", "start"],
            grpc_server_working_dir="/tmp",
        )
        mock_start_server.assert_called_once()
        mock_bootstrap_and_login.assert_called_once()

    @patch("altastata.grpc_client.AltaStataGrpcClient._create_channel")
    def test_bootstrap_skips_set_private_key_for_hpcs_account(self, mock_create_channel):
        """HPCS / HSM-backed accounts pass private_key_encrypted="" because the
        key lives in the HSM. The server's SetPrivateKey validator rejects an
        empty PEM with INVALID_ARGUMENT, so the client must skip that RPC
        entirely. SetUserProperties and Login must still run."""
        users_stub_mock = MagicMock()
        auth_stub_mock = MagicMock()
        auth_stub_mock.Login.return_value = MagicMock(session_token="sess-hpcs-token")

        sys.modules["altastata.v1.users_pb2_grpc"].UsersServiceStub = MagicMock(
            return_value=users_stub_mock,
        )
        sys.modules["altastata.v1.auth_pb2_grpc"].AuthServiceStub = MagicMock(
            return_value=auth_stub_mock,
        )
        sys.modules["altastata.v1.users_pb2"].SetUserPropertiesRequest = MagicMock()
        sys.modules["altastata.v1.users_pb2"].SetPrivateKeyRequest = MagicMock()
        sys.modules["altastata.v1.auth_pb2"].LoginRequest = MagicMock()

        mock_create_channel.return_value = MagicMock()

        from altastata.grpc_client import _bootstrap_and_login, GrpcEndpoint
        token = _bootstrap_and_login(
            endpoint=GrpcEndpoint(host="127.0.0.1", port=9877, secure=False),
            user_name="serge678",
            user_properties="myuser=serge678\nkey-protection=HPCS\n",
            private_key_encrypted="",
            password="",
            client_hint="hpcs-test",
        )

        self.assertEqual("sess-hpcs-token", token)
        users_stub_mock.SetUserProperties.assert_called_once()
        users_stub_mock.SetPrivateKey.assert_not_called()
        auth_stub_mock.Login.assert_called_once()

    @patch("altastata.grpc_client.AltaStataGrpcClient._create_channel")
    def test_bootstrap_calls_set_private_key_for_rsa_account(self, mock_create_channel):
        """Regression guard: password-based / RSA accounts pass a real
        encrypted PEM and SetPrivateKey must still be called for them."""
        users_stub_mock = MagicMock()
        auth_stub_mock = MagicMock()
        auth_stub_mock.Login.return_value = MagicMock(session_token="sess-rsa-token")

        sys.modules["altastata.v1.users_pb2_grpc"].UsersServiceStub = MagicMock(
            return_value=users_stub_mock,
        )
        sys.modules["altastata.v1.auth_pb2_grpc"].AuthServiceStub = MagicMock(
            return_value=auth_stub_mock,
        )
        sys.modules["altastata.v1.users_pb2"].SetUserPropertiesRequest = MagicMock()
        sys.modules["altastata.v1.users_pb2"].SetPrivateKeyRequest = MagicMock()
        sys.modules["altastata.v1.auth_pb2"].LoginRequest = MagicMock()

        mock_create_channel.return_value = MagicMock()

        from altastata.grpc_client import _bootstrap_and_login, GrpcEndpoint
        token = _bootstrap_and_login(
            endpoint=GrpcEndpoint(host="127.0.0.1", port=9877, secure=False),
            user_name="bob123",
            user_properties="myuser=bob123\n",
            private_key_encrypted="-----BEGIN RSA PRIVATE KEY-----\n...\n",
            password="123",
            client_hint="rsa-test",
        )

        self.assertEqual("sess-rsa-token", token)
        users_stub_mock.SetUserProperties.assert_called_once()
        users_stub_mock.SetPrivateKey.assert_called_once()
        auth_stub_mock.Login.assert_called_once()

    @patch("altastata.grpc_client.subprocess.Popen")
    @patch("altastata.grpc_client._find_bundled_grpc_uber_jar")
    @patch("altastata.grpc_client._build_bundled_grpc_classpath")
    def test_start_local_grpc_service_prefers_bundled_uber_jar(
        self,
        mock_build_cp,
        mock_find_uber,
        mock_popen,
    ):
        mock_find_uber.return_value = "/tmp/altastata-grpc-1.0.0-uber.jar"
        mock_build_cp.return_value = "/tmp/a.jar:/tmp/b.jar:/tmp/altastata-grpc-1.0.0-uber.jar"
        mock_popen.return_value = MagicMock()

        from altastata.grpc_client import _start_local_grpc_service
        _start_local_grpc_service()

        # env is built by _build_grpc_subprocess_env (covered separately) and
        # is forwarded to Popen so the Java side can pick up the bundled SPA.
        mock_popen.assert_called_once_with(
            ["java", "-cp", "/tmp/a.jar:/tmp/b.jar:/tmp/altastata-grpc-1.0.0-uber.jar", "com.altastata.grpc.GrpcApplication"],
            cwd="/tmp",
            env=unittest.mock.ANY,
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
        )

    @patch("altastata.grpc_client.subprocess.Popen")
    @patch("altastata.grpc_client._default_mycloud_dir")
    @patch("altastata.grpc_client._find_bundled_grpc_uber_jar")
    def test_start_local_grpc_service_falls_back_to_gradle_when_uber_missing(
        self,
        mock_find_uber,
        mock_default_mycloud_dir,
        mock_popen,
    ):
        mock_find_uber.return_value = None
        mock_default_mycloud_dir.return_value = "/work/mycloud"
        mock_popen.return_value = MagicMock()

        from altastata.grpc_client import _start_local_grpc_service
        _start_local_grpc_service()

        mock_popen.assert_called_once_with(
            ["./gradlew", ":altastata-grpc:run"],
            cwd="/work/mycloud",
            env=unittest.mock.ANY,
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
        )

    @patch("altastata.grpc_client._find_bundled_console_ui_dir")
    def test_build_subprocess_env_exports_ui_dir_when_bundle_present(
        self, mock_find_ui
    ):
        mock_find_ui.return_value = "/wheel/altastata/lib/altastata-console-static"

        from altastata.grpc_client import _build_grpc_subprocess_env
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALTASTATA_WEB_UI_DIR", None)
            env = _build_grpc_subprocess_env()

        self.assertEqual(
            env["ALTASTATA_WEB_UI_DIR"],
            "/wheel/altastata/lib/altastata-console-static",
        )

    @patch("altastata.grpc_client._find_bundled_console_ui_dir")
    def test_build_subprocess_env_skips_ui_dir_when_no_bundle(self, mock_find_ui):
        mock_find_ui.return_value = None

        from altastata.grpc_client import _build_grpc_subprocess_env
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALTASTATA_WEB_UI_DIR", None)
            env = _build_grpc_subprocess_env()

        self.assertNotIn("ALTASTATA_WEB_UI_DIR", env)

    @patch("altastata.grpc_client._find_bundled_console_ui_dir")
    def test_build_subprocess_env_respects_caller_override(self, mock_find_ui):
        # If the caller explicitly set the variable (even to a different
        # path, or to empty to disable), the bundled lookup must not
        # silently override it.
        mock_find_ui.return_value = "/wheel/altastata/lib/altastata-console-static"

        from altastata.grpc_client import _build_grpc_subprocess_env
        with patch.dict(os.environ, {"ALTASTATA_WEB_UI_DIR": "/custom/path"}):
            env = _build_grpc_subprocess_env()

        self.assertEqual(env["ALTASTATA_WEB_UI_DIR"], "/custom/path")
        mock_find_ui.assert_not_called()

    def test_find_bundled_console_ui_dir_returns_none_for_missing_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_pkg = os.path.join(tmp, "altastata", "lib")
            os.makedirs(empty_pkg)
            with patch(
                "altastata.grpc_client.pkg_resources.resource_filename",
                return_value=os.path.join(empty_pkg, "altastata-console-static"),
            ):
                from altastata.grpc_client import _find_bundled_console_ui_dir
                self.assertIsNone(_find_bundled_console_ui_dir())

    def test_find_bundled_console_ui_dir_returns_path_when_index_html_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui_dir = os.path.join(tmp, "altastata-console-static")
            os.makedirs(ui_dir)
            with open(os.path.join(ui_dir, "index.html"), "w") as f:
                f.write("<html></html>")
            with patch(
                "altastata.grpc_client.pkg_resources.resource_filename",
                return_value=ui_dir,
            ):
                from altastata.grpc_client import _find_bundled_console_ui_dir
                self.assertEqual(_find_bundled_console_ui_dir(), os.path.abspath(ui_dir))


if __name__ == "__main__":
    unittest.main()
