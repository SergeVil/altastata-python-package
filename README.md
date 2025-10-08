# Altastata Python Package v0.1.17

A powerful Python package for data processing and machine learning integration with Altastata.

## Installation

```bash
pip install altastata
```

## Features

- Seamless integration with PyTorch and TensorFlow
- **fsspec filesystem interface** for standard Python file operations
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

## Version Information

**Current Version**: 0.1.17

This version includes:
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

- **Multi-Architecture Support**: Docker images now work natively on both AMD64 and ARM64 platforms
- **Error Handling**: Enhanced `delete_files` method with detailed error reporting
- **Performance**: Optimized file reading operations
- **Compatibility**: Updated AWS IAM configurations for better permission management
- **Documentation**: Consistent version numbering across all components

This project is licensed under the MIT License - see the LICENSE file for details. 