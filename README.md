from altastata.altastata_tensorflow_dataset import register_altastata_functions_for_tensorflowfrom altastata.altastata_pytorch_dataset import register_altastata_functions_for_pytorch

# Altastata Python Package

A powerful Python package for data processing and machine learning integration with Altastata.

## Installation

```bash
pip install altastata
```

## Features

- Seamless integration with PyTorch and TensorFlow
- Advanced data processing capabilities
- Java integration through Py4J
- Support for large-scale data operations

## Quick Start

```python
from altastata import AltaStataFunctions, AltaStataPyTorchDataset, AltaStataTensorFlowDataset

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

# register the altastata functions for PyTorch or TensorFlow as a custom dataset
register_altastata_functions_for_pytorch(altastata_functions, "bob123_rsa")
register_altastata_functions_for_tensorflow(altastata_functions, "bob123_rsa")

# for PyTorch application use
torch_dataset = AltaStataPyTorchDataset(
    "bob123_rsa",
    root_dir=root_dir,
    file_pattern=pattern,
    transform=transform
)

# for tensorflow application use
tensorflow_dataset = AltaStataTensorFlowDataset(
    "bob123_rsa",  # Using AltaStata account for testing
    root_dir=root_dir,
    file_pattern=pattern,
    preprocess_fn=preprocess_fn
)
```

This project is licensed under the MIT License - see the LICENSE file for details. 