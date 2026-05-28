import unittest
from unittest.mock import MagicMock, patch


class AltaStataFunctionsTransportTests(unittest.TestCase):
    @patch("altastata.altastata_functions.AltaStataGrpcClient")
    def test_from_credentials_grpc_uses_grpc_backend(self, mock_grpc_cls):
        mock_client = MagicMock()
        mock_grpc_cls.from_credentials.return_value = mock_client

        from altastata.altastata_functions import AltaStataFunctions

        f = AltaStataFunctions.from_credentials(
            "myuser=bob123\nregion=us-east-1\n",
            "-----BEGIN RSA PRIVATE KEY-----\n...\n",
            transport="grpc",
            password="123",
        )

        self.assertEqual("grpc", f.transport)
        self.assertIs(f.grpc_client, mock_client)
        f.set_password("abc")
        mock_client.set_password.assert_called_once_with("abc")

    @patch("altastata.altastata_functions.AltaStataGrpcClient")
    def test_get_file_attribute_delegates_to_grpc(self, mock_grpc_cls):
        mock_client = MagicMock()
        mock_client.get_file_attribute.return_value = "42"
        mock_grpc_cls.from_credentials.return_value = mock_client

        from altastata.altastata_functions import AltaStataFunctions

        f = AltaStataFunctions.from_credentials(
            "myuser=bob123\nregion=us-east-1\n",
            "-----BEGIN RSA PRIVATE KEY-----\n...\n",
            transport="grpc",
            password="123",
        )
        value = f.get_file_attribute("a/b.txt", None, "size")
        self.assertEqual("42", value)


if __name__ == "__main__":
    unittest.main()
