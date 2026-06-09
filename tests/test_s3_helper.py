"""Unit tests for AltaStataFunctions S3 / boto3 helper surface.

These tests do not boot a real JVM. They mock out:
  * the py4j gateway construction (via AltaStataFunctions.__new__ + manual
    field assignment), so we do not depend on a running altastata-services
    process,
  * the bootstrap PUTs (via the module-level ``_http_put_text`` symbol),
  * and ``boto3.client`` (so ``boto3`` does not need to be installed for the
    test suite).

They exercise the public surface of s3_credentials(), boto3_s3(),
install_aws_env(), and the underlying _resolve_s3_endpoint() /
_read_bootstrap_material() helpers.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from altastata.altastata_functions import (
    AltaStataFunctions,
    _parse_user_name_from_properties,
    _http_put_text,
)


def _make_instance(transport: str = "py4j") -> AltaStataFunctions:
    """Build an AltaStataFunctions without touching any real JVM / gRPC.

    We bypass __init__ since it spins up a py4j gateway in py4j mode and we
    only care about the helper surface in unit tests.
    """
    inst = AltaStataFunctions.__new__(AltaStataFunctions)
    inst.transport = transport
    inst.gateway = None
    inst.altastata_file_system = None
    inst.grpc_client = None
    inst._event_listeners = []
    inst._account_dir_path = None
    inst._user_properties = None
    inst._private_key_encrypted = None
    inst._cached_password = None
    inst._s3_credentials_cache = {}
    return inst


class ParseUserNameTests(unittest.TestCase):
    def test_picks_myuser_line(self):
        self.assertEqual(
            "bob123",
            _parse_user_name_from_properties(
                "region=us-east-1\nmyuser=bob123\naccounttype=amazon\n"
            ),
        )

    def test_strips_whitespace(self):
        self.assertEqual(
            "alice",
            _parse_user_name_from_properties("myuser=  alice  \n"),
        )

    def test_ignores_commented_line(self):
        with self.assertRaises(ValueError):
            _parse_user_name_from_properties("# myuser=ghost\nfoo=bar\n")

    def test_missing_raises(self):
        with self.assertRaises(ValueError):
            _parse_user_name_from_properties("region=us-east-1\n")

    def test_empty_value_raises(self):
        with self.assertRaises(ValueError):
            _parse_user_name_from_properties("myuser=\n")


class ResolveS3EndpointTests(unittest.TestCase):
    def test_py4j_defaults_to_loopback(self):
        inst = _make_instance("py4j")
        self.assertEqual("http://127.0.0.1:9876", inst._resolve_s3_endpoint())

    def test_grpc_uses_grpc_host(self):
        inst = _make_instance("grpc")
        inst.grpc_client = MagicMock()
        inst.grpc_client.endpoint.host = "altastata.internal"
        self.assertEqual(
            "http://altastata.internal:9876", inst._resolve_s3_endpoint()
        )


class ReadBootstrapMaterialTests(unittest.TestCase):
    def test_from_credentials_mode_returns_stored_strings(self):
        inst = _make_instance("py4j")
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PRIVATE_KEY_BYTES"
        user_name, props, key = inst._read_bootstrap_material()
        self.assertEqual("bob", user_name)
        self.assertEqual("myuser=bob\n", props)
        self.assertEqual("PRIVATE_KEY_BYTES", key)

    def test_from_account_dir_mode_reads_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            props_path = os.path.join(
                tmpdir, "altastata-myorg-bob.user.properties"
            )
            with open(props_path, "w", encoding="utf-8") as f:
                f.write("myuser=bob\nregion=us-east-1\n")
            with open(os.path.join(tmpdir, "private.key"), "w", encoding="utf-8") as f:
                f.write("PK_PEM")

            inst = _make_instance("py4j")
            inst._account_dir_path = tmpdir
            user_name, props, key = inst._read_bootstrap_material()
            self.assertEqual("bob", user_name)
            self.assertIn("myuser=bob", props)
            self.assertEqual("PK_PEM", key)

    def test_missing_material_raises(self):
        inst = _make_instance("py4j")
        with self.assertRaises(RuntimeError):
            inst._read_bootstrap_material()

    def test_account_dir_missing_properties_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inst = _make_instance("py4j")
            inst._account_dir_path = tmpdir
            with self.assertRaises(FileNotFoundError):
                inst._read_bootstrap_material()


class S3CredentialsTests(unittest.TestCase):
    def _put_responder(self):
        """Return a ``_http_put_text`` mock that mimics the S3 admin PUTs."""
        def _put(url, body, timeout_s=30.0):
            if "/setUserProperties/" in url:
                return 200, b'{"status":"success"}'
            if "/setPrivateKey/" in url:
                return 200, b'{"status":"success"}'
            if "/setPassword/" in url:
                return 200, json.dumps({
                    "status": "success",
                    "accessKey": "AKIAFAKE",
                    "secretKey": "SECRETFAKE",
                }).encode("utf-8")
            return 404, b'{"error":"unknown"}'
        return _put

    @patch("altastata.altastata_functions._http_put_text")
    def test_returns_boto3_kwargs(self, mock_put):
        mock_put.side_effect = self._put_responder()
        inst = _make_instance("py4j")
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PRIVATE_KEY"
        inst._cached_password = "123"

        creds = inst.s3_credentials()

        self.assertEqual(
            {
                "endpoint_url": "http://127.0.0.1:9876",
                "aws_access_key_id": "AKIAFAKE",
                "aws_secret_access_key": "SECRETFAKE",
                "region_name": "us-east-1",
            },
            creds,
        )
        # Three PUTs in the expected order, and the first two carry
        # ?password=<urlencoded> so re-bootstrapping an already-known user
        # succeeds (see s3_credentials() body for the rationale).
        urls_called = [c.args[0] for c in mock_put.call_args_list]
        self.assertEqual(3, len(urls_called))
        self.assertIn("/setUserProperties/bob?password=123", urls_called[0])
        self.assertIn("/setPrivateKey/bob?password=123", urls_called[1])
        self.assertIn("/setPassword/bob", urls_called[2])
        # setPassword sends password in body, not query.
        self.assertNotIn("?password=", urls_called[2])

    @patch("altastata.altastata_functions._http_put_text")
    def test_password_is_url_encoded(self, mock_put):
        mock_put.side_effect = self._put_responder()
        inst = _make_instance("py4j")
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PK"
        # Password with characters that MUST be URL-escaped to survive a
        # query-string round-trip on S3Controller's validatePassword path.
        inst._cached_password = "p w&p=1?#/"

        inst.s3_credentials()

        urls = [c.args[0] for c in mock_put.call_args_list]
        # Reserved characters end up percent-encoded; literal `?` / `&` /
        # `=` from the password must not split the query string.
        encoded = "p%20w%26p%3D1%3F%23%2F"
        self.assertIn(f"?password={encoded}", urls[0])
        self.assertIn(f"?password={encoded}", urls[1])

    @patch("altastata.altastata_functions._http_put_text")
    def test_caches_result_per_endpoint(self, mock_put):
        mock_put.side_effect = self._put_responder()
        inst = _make_instance("py4j")
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PK"
        inst._cached_password = "p"

        inst.s3_credentials()
        inst.s3_credentials()
        # Three PUTs total, not six — the second call must hit the cache.
        self.assertEqual(3, mock_put.call_count)

    @patch("altastata.altastata_functions._http_put_text")
    def test_explicit_endpoint_override(self, mock_put):
        mock_put.side_effect = self._put_responder()
        inst = _make_instance("py4j")
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PK"
        inst._cached_password = "p"

        creds = inst.s3_credentials(endpoint="http://altastata.example:19876/")
        self.assertEqual("http://altastata.example:19876", creds["endpoint_url"])
        for call in mock_put.call_args_list:
            self.assertTrue(call.args[0].startswith("http://altastata.example:19876/"))

    @patch("altastata.altastata_functions._http_put_text")
    def test_already_bootstrapped_user_recovers(self, mock_put):
        """The S3 gateway rejects setUserProperties / setPrivateKey with 400
        once UserData exists, unless ?password= is supplied. The helper
        always passes it, so re-bootstrapping from a fresh wheel process
        against an already-warmed JVM must succeed."""
        calls = {"n": 0}

        def _put(url, body, timeout_s=30.0):
            calls["n"] += 1
            if "/setUserProperties/" in url:
                # Mimic S3Controller: reject when no ?password=, accept when
                # present (regardless of value — we already covered the
                # validation surface with a separate test).
                if "?password=" not in url:
                    return 400, (
                        b'{"error": "User properties already exist. '
                        b'Password required for modification."}'
                    )
                return 200, b'{"status":"success"}'
            if "/setPrivateKey/" in url:
                if "?password=" not in url:
                    return 400, b'{"error":"Private key already exists."}'
                return 200, b'{"status":"success"}'
            if "/setPassword/" in url:
                return 200, json.dumps({
                    "accessKey": "AK_REBOOT", "secretKey": "SK_REBOOT"
                }).encode("utf-8")
            return 404, b''

        mock_put.side_effect = _put

        inst = _make_instance("py4j")
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PK"
        inst._cached_password = "123"

        creds = inst.s3_credentials()
        self.assertEqual("AK_REBOOT", creds["aws_access_key_id"])
        self.assertEqual(3, calls["n"])

    def test_missing_password_raises(self):
        inst = _make_instance("py4j")
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PK"
        with self.assertRaises(ValueError):
            inst.s3_credentials()

    @patch("altastata.altastata_functions._http_put_text")
    def test_explicit_password_overrides_cache(self, mock_put):
        mock_put.side_effect = self._put_responder()
        inst = _make_instance("py4j")
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PK"
        inst._cached_password = "wrong"

        inst.s3_credentials(password="correct")
        last_call = mock_put.call_args_list[-1]
        # body of the third PUT is the password (second positional arg).
        self.assertEqual("correct", last_call.args[1])

    @patch("altastata.altastata_functions._http_put_text")
    def test_setPassword_500_surfaces_as_runtime_error(self, mock_put):
        def _put(url, body, timeout_s=30.0):
            if "/setPassword/" in url:
                return 500, b"server error"
            return 200, b'{"status":"success"}'
        mock_put.side_effect = _put

        inst = _make_instance("py4j")
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PK"
        inst._cached_password = "p"

        with self.assertRaises(RuntimeError):
            inst.s3_credentials()

    @patch("altastata.altastata_functions._http_put_text")
    def test_grpc_mode_default_endpoint_follows_grpc_host(self, mock_put):
        mock_put.side_effect = self._put_responder()
        inst = _make_instance("grpc")
        inst.grpc_client = MagicMock()
        inst.grpc_client.endpoint.host = "altastata.internal"
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PK"
        inst._cached_password = "p"

        creds = inst.s3_credentials()
        self.assertEqual("http://altastata.internal:9876", creds["endpoint_url"])


class Boto3ClientTests(unittest.TestCase):
    @patch("altastata.altastata_functions._http_put_text")
    def test_boto3_s3_passes_creds_and_overrides(self, mock_put):
        def _put(url, body, timeout_s=30.0):
            if "/setPassword/" in url:
                return 200, json.dumps({
                    "accessKey": "AK", "secretKey": "SK"
                }).encode("utf-8")
            return 200, b'{"status":"success"}'
        mock_put.side_effect = _put

        fake_boto3 = MagicMock()
        fake_boto3.client.return_value = "<s3-client>"
        with patch.dict(sys.modules, {"boto3": fake_boto3}):
            inst = _make_instance("py4j")
            inst._user_properties = "myuser=bob\n"
            inst._private_key_encrypted = "PK"
            inst._cached_password = "p"

            client = inst.boto3_s3(verify=False)

        self.assertEqual("<s3-client>", client)
        fake_boto3.client.assert_called_once()
        call_args, call_kwargs = fake_boto3.client.call_args
        self.assertEqual(("s3",), call_args)
        self.assertEqual("http://127.0.0.1:9876", call_kwargs["endpoint_url"])
        self.assertEqual("AK", call_kwargs["aws_access_key_id"])
        self.assertEqual("SK", call_kwargs["aws_secret_access_key"])
        self.assertIs(False, call_kwargs["verify"])

    def test_boto3_s3_missing_dependency_raises_importerror(self):
        inst = _make_instance("py4j")
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PK"
        inst._cached_password = "p"

        # Force `import boto3` inside boto3_s3() to fail.
        prev = sys.modules.get("boto3", None)
        sys.modules["boto3"] = None
        try:
            with self.assertRaises(ImportError):
                inst.boto3_s3()
        finally:
            if prev is not None:
                sys.modules["boto3"] = prev
            else:
                del sys.modules["boto3"]


class InstallAwsEnvTests(unittest.TestCase):
    @patch("altastata.altastata_functions._http_put_text")
    def test_install_aws_env_populates_environ(self, mock_put):
        def _put(url, body, timeout_s=30.0):
            if "/setPassword/" in url:
                return 200, json.dumps({
                    "accessKey": "AK_ENV", "secretKey": "SK_ENV"
                }).encode("utf-8")
            return 200, b'{"status":"success"}'
        mock_put.side_effect = _put

        inst = _make_instance("py4j")
        inst._user_properties = "myuser=bob\n"
        inst._private_key_encrypted = "PK"
        inst._cached_password = "p"

        # Snapshot env so the test does not bleed into siblings.
        captured = {
            k: os.environ.get(k)
            for k in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION",
                "AWS_ENDPOINT_URL_S3",
            )
        }
        try:
            aws_env = inst.install_aws_env()
            self.assertEqual("AK_ENV", os.environ["AWS_ACCESS_KEY_ID"])
            self.assertEqual("SK_ENV", os.environ["AWS_SECRET_ACCESS_KEY"])
            self.assertEqual("us-east-1", os.environ["AWS_DEFAULT_REGION"])
            self.assertEqual("http://127.0.0.1:9876", os.environ["AWS_ENDPOINT_URL_S3"])
            self.assertEqual("AK_ENV", aws_env["AWS_ACCESS_KEY_ID"])
        finally:
            for k, v in captured.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class SetPasswordCachesTests(unittest.TestCase):
    @patch("altastata.altastata_functions.AltaStataGrpcClient")
    def test_set_password_caches_value(self, mock_grpc_cls):
        mock_client = MagicMock()
        mock_grpc_cls.from_credentials.return_value = mock_client

        f = AltaStataFunctions.from_credentials(
            "myuser=bob\n",
            "PK",
            transport="grpc",
        )
        # Pre-condition: no cached password (no constructor kwarg).
        self.assertIsNone(f._cached_password)
        f.set_password("hunter2")
        self.assertEqual("hunter2", f._cached_password)


if __name__ == "__main__":
    unittest.main()
