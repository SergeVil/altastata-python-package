# Altastata Python Package v0.1.18

A powerful Python package for data processing and machine learning integration with Altastata.

## Installation

```bash
pip install altastata
```

## Features

- Seamless integration with PyTorch and TensorFlow
- **fsspec filesystem interface** for standard Python file operations
- **Real-time Event Notifications**: Listen for file share, delete, and create events
- Advanced data processing capabilities
- Java integration through Py4J with optimized memory management
- Support for large-scale data operations
- Improved garbage collection and memory optimization
- Enhanced error handling for cloud operations
- Optimized file reading with direct attribute access
- Comprehensive AWS IAM permission management
- **Confidential Computing Support**: Deploy on Google Cloud Platform with AMD SEV security
- Robust file operation status tracking

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

# Register the altastata functions for PyTorch or TensorFlow as a custom dataset
register_altastata_functions_for_pytorch(altastata_functions, "bob123_rsa")
register_altastata_functions_for_tensorflow(altastata_functions, "bob123_rsa")

# For PyTorch application use
torch_dataset = AltaStataPyTorchDataset(
    "bob123_rsa",
    root_dir=root_dir,
    file_pattern=pattern,
    transform=transform
)

# For TensorFlow application use
tensorflow_dataset = AltaStataTensorFlowDataset(
    "bob123_rsa",  # Using AltaStata account for testing
    root_dir=root_dir,
    file_pattern=pattern,
    preprocess_fn=preprocess_fn
)
```

## fsspec Integration

```python
from altastata import AltaStataFunctions
from altastata.fsspec import create_filesystem

# Create AltaStata connection
altastata_functions = AltaStataFunctions.from_account_dir('/path/to/account')
altastata_functions.set_password("your_password")

# Create fsspec filesystem
fs = create_filesystem(altastata_functions, "my_account")

# Use standard file operations
files = fs.ls("Public/")
with fs.open("Public/Documents/file.txt", "r") as f:
    content = f.read()
```

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
- Audit logging and compliance
- Real-time sync and backup
- Security monitoring
- RAG vector store updates
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

## Version Information

**Current Version**: 0.1.18

This version includes:
- **Event Listener Support**: Real-time notifications for file operations (share, delete, create)
- **fsspec Integration**: Standard Python filesystem interface for seamless file operations
- **LangChain Integration**: Native support for LangChain document loaders and vector stores
- Rebuilt `altastata-hadoop-all.jar` with latest improvements
- Enhanced error handling in `delete_files` operations
- Simplified `_read_file` method for better performance
- Updated AWS account configurations
- Improved memory management and garbage collection
- Comprehensive status tracking for cloud operations

## Docker Support

The package is available as a **multi-architecture Docker image** that works natively on both AMD64 and ARM64 platforms:

```bash
# Pull multi-architecture image (automatically selects correct architecture)
docker pull ghcr.io/sergevil/altastata/jupyter-datascience:latest

# Or use docker-compose
docker-compose -f docker-compose-ghcr.yml up -d
```

**Platform Support:**
- **Apple Silicon Macs**: Native ARM64 performance
- **Intel Macs**: Native AMD64 performance  
- **GCP Confidential GKE**: Native AMD64 performance
- **Other platforms**: Automatic architecture selection

## Confidential Computing Deployment

Deploy Altastata in a secure, confidential computing environment on Google Cloud Platform:

```bash
# Navigate to confidential GKE setup
cd confidential-gke

# Deploy confidential cluster with AMD SEV security
./setup-cluster.sh

# Access Jupyter Lab at the provided URL
# Stop cluster when not in use (saves costs)
gcloud container clusters delete altastata-confidential-cluster --zone=us-central1-a
```

**Features:**
- **Hardware-level security** with AMD SEV encryption
- **Memory encryption** during data processing
- **Multi-cloud storage** support (GCP, AWS, Azure)
- **Cost optimization** with easy stop/start commands
- **Multi-architecture support** for both AMD64 and ARM64 platforms

See `confidential-gke/README.md` for detailed setup instructions.

## Recent Improvements

- **Event Listener System**: Real-time notifications for file share, delete, and create events via Py4J callbacks
- **fsspec Integration**: Standard Python filesystem interface for seamless file operations with any Python library
- **LangChain Support**: Native integration with LangChain document loaders and vector stores for RAG applications
- **Multi-Architecture Support**: Docker images now work natively on both AMD64 and ARM64 platforms
- **Error Handling**: Enhanced `delete_files` method with detailed error reporting
- **Performance**: Optimized file reading operations
- **Compatibility**: Updated AWS IAM configurations for better permission management
- **Documentation**: Consistent version numbering across all components

This project is licensed under the MIT License - see the LICENSE file for details. 