# AltaStata fsspec Integration

Simple fsspec filesystem interface for AltaStata that automatically uses the latest version of files.

## Examples

This directory contains working examples and tests:

- **`example.py`** - Basic fsspec usage example
- **`test_simple.py`** - Simple fsspec functionality test (small file)
- **`test_large_file_fsspec.py`** - Performance test with 100MB files and data verification

## Running Tests

```bash
# Basic example
python example.py

# Simple functionality test (small file)
python test_simple.py

# Large file performance test (100MB with data verification)
python test_large_file_fsspec.py
```

## RAG Pipeline

For secure RAG (Retrieval-Augmented Generation) implementations with AltaStata, see the **[rag-example/](../rag-example/)** directory:

- Complete working RAG pipeline with LangChain
- Security architecture documentation
- Sample policy documents
- Real test output examples

## Installation

```bash
pip install altastata fsspec
```

## Usage

```python
import fsspec
from altastata import AltaStataFileSystem

# Register filesystem
AltaStataFileSystem.register()

# Use with fsspec
fs = fsspec.filesystem("altastata", account_id="my_account")
files = fs.ls("Public/")

# Read file (always latest version)
with fs.open("Public/Documents/file.txt", "r") as f:
    content = f.read()
```

## LangChain Integration

```python
from langchain_community.document_loaders import TextLoader

# Register filesystem
AltaStataFileSystem.register()

# Load document
loader = TextLoader("altastata://Public/Documents/file.txt")
documents = loader.load()
```

## Configuration

```python
from altastata import AltaStataFileSystem

# Option 1: Account directory
fs = AltaStataFileSystem(
    account_id="my_account",
    account_dir="/path/to/.altastata/accounts/my_account",
    password="your_password"
)

# Option 2: Credentials
fs = AltaStataFileSystem(
    account_id="my_account",
    user_properties="your_user_properties",
    private_key_encrypted="your_private_key",
    password="your_password"
)
```