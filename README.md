# Altastata Python Package

A powerful Python package for secure, encrypted cloud storage with seamless integration for data processing, AI, machine learning, and RAG applications.

## Installation

```bash
pip install altastata
```

## Features

- **fsspec filesystem interface** - Use standard Python file operations with encrypted cloud storage
- **S3-compatible API** - Drive boto3 / `aws` CLI / `s3fs` / pyarrow against the bundled S3 gateway with one helper call
- **Real-time Event Notifications** - Listen for file share, delete, and create events
- **LangChain Integration** - Native support for document loaders and vector stores
- **PyTorch & TensorFlow Support** - Custom datasets for machine learning workflows
- **Multi-cloud Support** - Works with AWS, Azure, GCP, and more
- **End-to-end Encryption** - AES-256 encryption with zero-trust architecture

## Quick Start

```python
from altastata import AltaStataFunctions, AltaStataPyTorchDataset, AltaStataTensorFlowDataset
from altastata.altastata_tensorflow_dataset import register_altastata_functions_for_tensorflow
from altastata.altastata_pytorch_dataset import register_altastata_functions_for_pytorch

# Configuration parameters
user_properties = """#My Properties
#Sun Jan 05 12:10:23 EST 2025
AWSSecretKey=*****
AWSAccessKeyId=*****
myuser=bob123
accounttype=amazon-s3-secure
................................................................
region=us-east-1"""

private_key = """-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3,F26EBECE6DDAEC52

poe21ejZGZQ0GOe+EJjDdJpNvJcq/Yig9aYXY2rCGyxXLGVFeYJFg7z6gMCjIpSd
................................................................
wV5BUmp5CEmbeB4r/+BlFttRZBLBXT1sq80YyQIVLumq0Livao9mOg==
-----END RSA PRIVATE KEY-----"""

# Create an instance of AltaStataFunctions
altastata_functions = AltaStataFunctions.from_credentials(user_properties, private_key)
altastata_functions.set_password("my_password")
```

### gRPC transport (recommended for Python + browser integration)

```python
from altastata import AltaStataFunctions

f = AltaStataFunctions.from_account_dir(
    "/path/to/account",
    transport="grpc",
    password="123",
)
```

`transport="grpc"` auto-starts `altastata-grpc` when needed.

To run the gRPC server explicitly (for browser JS or local testing), use either:

```bash
altastata-grpc-server
# or (same entry point via Python module):
python -m altastata.grpc_server
```

The launcher binds gRPC to `127.0.0.1:9877` and, when the wheel ships with a
bundled AltaStata Console SPA at `altastata/lib/altastata-console-static/`,
serves the SPA from the same port. Open <http://127.0.0.1:9877> in a browser
to get a Finder-style file manager with auto-refresh on `SHARE` / `DELETE`
events from other users (see `mycloud/altastata-grpc/EventsService` and
`altastata-console`). The launcher exports `ALTASTATA_WEB_UI_DIR`
automatically; set `ALTASTATA_WEB_UI_DIR=` to disable the UI and keep the
port gRPC-only.

## PyTorch & TensorFlow Integration

```python
# Register the altastata functions for PyTorch or TensorFlow as a custom dataset
register_altastata_functions_for_pytorch(altastata_functions, "my_account")
register_altastata_functions_for_tensorflow(altastata_functions, "my_account")

# For PyTorch application use
torch_dataset = AltaStataPyTorchDataset(
    "my_account",
    root_dir=root_dir,
    file_pattern=pattern,
    transform=transform
)

# For TensorFlow application use
tensorflow_dataset = AltaStataTensorFlowDataset(
    "my_account",
    root_dir=root_dir,
    file_pattern=pattern,
    preprocess_fn=preprocess_fn
)
```

## fsspec Integration

Altastata implements the fsspec interface, making it compatible with any Python library that uses standard file operations:

```python
from altastata import AltaStataFunctions
from altastata.fsspec import create_filesystem

# Create AltaStata connection
altastata_functions = AltaStataFunctions.from_account_dir('/path/to/account')
altastata_functions.set_password("your_password")

# Create fsspec filesystem
fs = create_filesystem(altastata_functions, "my_account")

# Use it like any Python file system
files = fs.ls("Public/")
with fs.open("Public/Documents/file.txt", "r") as f:
    content = f.read()
    print(content)
```

This means you can use Altastata with pandas, dask, xarray, and hundreds of other libraries without any special configuration.

## boto3 / `aws` CLI / `s3fs` (S3-compatible API)

The bundled `altastata-services` JVM exposes an S3-compatible REST API on
port `9876` from inside the **same process** that backs the Python API.
Reads/writes from `boto3` resolve to the same `AltaStataFileSystem`
instance the Python API uses (shared `AccountRegistry`, no second cache).
Three helpers on `AltaStataFunctions` drive the admin bootstrap and
surface the access/secret pair the gateway generates:

```python
from altastata import AltaStataFunctions

alt = AltaStataFunctions.from_account_dir("/path/to/account")
alt.set_password("your_password")

# One-liner: ready-to-use boto3 S3 client.
s3 = alt.boto3_s3()
print(s3.list_buckets())
s3.put_object(Bucket="my-bucket", Key="hello.txt", Body=b"hi")

# Or: just the kwargs — pass to any AWS SDK / s3fs / pyarrow / awswrangler.
creds = alt.s3_credentials()
# {'endpoint_url': 'http://127.0.0.1:9876',
#  'aws_access_key_id': 'AKIA...',
#  'aws_secret_access_key': '...',
#  'region_name': 'us-east-1'}

import s3fs
fs = s3fs.S3FileSystem(
    key=creds["aws_access_key_id"],
    secret=creds["aws_secret_access_key"],
    client_kwargs={"endpoint_url": creds["endpoint_url"]},
)

# Or: install AWS_* env vars so `!aws s3 ls`, `!s3cmd ls`, and any SDK
# that reads the ambient environment all "just work" — handy from Jupyter
# notebook shell cells.
alt.install_aws_env()
```

Requirements:

- The S3 gateway must be enabled on the JVM
  (`ALTASTATA_SERVICES_S3GATEWAY_ENABLED=true`, which is the **default in the
  bundled `jupyter-datascience` docker compose**).
- `boto3` is **not** in the wheel's `install_requires` — install it
  separately (`pip install boto3`) only if you want the
  `alt.boto3_s3()` convenience. `s3_credentials()` and `install_aws_env()`
  use stdlib only.

## Event Listener

Get real-time notifications when file operations occur:

```python
from altastata import AltaStataFunctions

# Event handler
def event_handler(event_name, data):
    print(f"📢 Event: {event_name}, Data: {data}")
    if event_name == "SHARE":
        print("File was shared!")
    elif event_name == "DELETE":
        print("File was deleted!")

# Initialize with callback server
altastata = AltaStataFunctions.from_account_dir(
    '/path/to/account',
    enable_callback_server=True,
    callback_server_port=25334
)
altastata.set_password("your_password")

# Register listener
listener = altastata.add_event_listener(event_handler)

# Events will now be delivered in real-time!
# See examples/event-listener-example/ for complete demos
```

**Perfect for:**
- Data sharing among the users
- Audit logging and compliance
- Workflow automation

See [`examples/event-listener-example/`](examples/event-listener-example/) for complete documentation and working examples.

## LangChain Integration

Use Altastata as a document source for LangChain applications:

```python
from langchain.document_loaders import DirectoryLoader
from altastata.fsspec import create_filesystem
from altastata import AltaStataFunctions

# Create AltaStata connection
altastata_functions = AltaStataFunctions.from_account_dir('/path/to/account')
altastata_functions.set_password("your_password")

# Create fsspec filesystem
fs = create_filesystem(altastata_functions, "my_account")

# Use with LangChain document loaders
loader = DirectoryLoader("Public/Documents/", filesystem=fs)
documents = loader.load()

# Use with vector stores
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
```

**Perfect for:**
- RAG (Retrieval-Augmented Generation) applications
- Document processing pipelines
- Knowledge base construction
- Multi-modal AI applications

See the [full documentation](https://github.com/sergevil/altastata-python-package) for more examples and advanced usage.

This project is licensed under the MIT License. 