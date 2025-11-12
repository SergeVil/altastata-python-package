# Altastata Python Package

A powerful Python package for secure, encrypted cloud storage with seamless integration for data processing, AI, machine learning, and RAG applications.

## Installation

```bash
pip install altastata
```

## Features

- **fsspec filesystem interface** - Use standard Python file operations with encrypted cloud storage
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
# See event-listener-example/ for complete demos
```

**Perfect for:**
- Data sharing among the users
- Audit logging and compliance
- Workflow automation

See [`event-listener-example/`](event-listener-example/) for complete documentation and working examples.

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