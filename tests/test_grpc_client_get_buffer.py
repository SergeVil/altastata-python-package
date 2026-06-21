import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from altastata.grpc_client import (
    GRPC_UNARY_MAX_SIZE,
    AltaStataGrpcClient,
    GrpcEndpoint,
)

# Unit tests intentionally inject mocked internals.
# pylint: disable=protected-access


class AltaStataGrpcClientGetBufferTests(unittest.TestCase):
    def _make_client(self):
        client = AltaStataGrpcClient.__new__(AltaStataGrpcClient)
        client._metadata = [("authorization", "Bearer test")]
        client._fileops_stub = MagicMock()
        client._fileops_pb2 = SimpleNamespace(
            GetBufferRequest=lambda **kwargs: kwargs,
            ReadStreamRequest=lambda **kwargs: kwargs,
        )
        return client

    def test_get_buffer_uses_unary_for_small_known_size(self):
        client = self._make_client()
        payload = b"hello-grpc"
        client._fileops_stub.GetBuffer.return_value = SimpleNamespace(data=payload)

        out = client.get_buffer("a/b.txt", size=1024)

        self.assertIs(out, payload)
        client._fileops_stub.GetBuffer.assert_called_once()
        client._fileops_stub.ReadStream.assert_not_called()

    def test_get_buffer_uses_streaming_for_large_size(self):
        client = self._make_client()
        chunk_a = SimpleNamespace(data=b"a" * 1000)
        chunk_b = SimpleNamespace(data=b"b" * 500)
        client._fileops_stub.ReadStream.return_value = iter([chunk_a, chunk_b])

        out = client.get_buffer("f", size=GRPC_UNARY_MAX_SIZE + 1500)

        self.assertEqual(out, b"a" * 1000 + b"b" * 500)
        client._fileops_stub.GetBuffer.assert_not_called()
        client._fileops_stub.ReadStream.assert_called_once()

    def test_get_buffer_uses_streaming_for_unknown_size(self):
        client = self._make_client()
        chunk_a = SimpleNamespace(data=b"foo")
        chunk_b = SimpleNamespace(data=b"bar")
        client._fileops_stub.ReadStream.return_value = iter([chunk_a, chunk_b])

        out = client.get_buffer("f", size=0)

        self.assertEqual(out, b"foobar")
        client._fileops_stub.GetBuffer.assert_not_called()
        client._fileops_stub.ReadStream.assert_called_once()


class GrpcEndpointTests(unittest.TestCase):
    def test_default_target_is_tcp(self):
        self.assertEqual(GrpcEndpoint().target, "127.0.0.1:9877")

    def test_socket_path_target_uses_unix_scheme(self):
        self.assertEqual(
            GrpcEndpoint(socket_path="/tmp/altastata.sock").target,
            "unix:/tmp/altastata.sock",
        )


if __name__ == "__main__":
    unittest.main()
