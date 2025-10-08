# AltaStata fsspec Integration

Simple fsspec filesystem interface for AltaStata that automatically uses the latest version of files.

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