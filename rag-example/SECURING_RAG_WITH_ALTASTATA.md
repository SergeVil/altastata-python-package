# Securing RAG with AltaStata

## Overview

RAG (Retrieval-Augmented Generation) systems combine LLMs with external knowledge bases but often expose sensitive data through unencrypted storage and inadequate access controls. **AltaStata provides enterprise-grade security** for RAG pipelines using AES-256 encryption, confidential computing, and automatic version control—all with seamless LangChain integration via fsspec.

## Key Security Features

- **End-to-End Encryption**: AES-256 encrypted documents with zero-knowledge architecture
- **Confidential Computing**: Confidential Container for hardware-level memory protection
- **Automatic Versioning**: Immutable audit trail for compliance (GDPR, HIPAA, SOC 2)
- **Access Control**: Account-based authentication with encrypted credentials
- **Multi-Cloud Support**: Unified security across AWS, IBM, MiniIO, GCP, and Azure

## Quick Start

### Installation

```bash
pip install altastata fsspec langchain langchain-community sentence-transformers faiss-cpu
```

### Basic Document Loading with fsspec

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from altastata import AltaStataFunctions
from altastata.fsspec import create_filesystem

# Initialize AltaStata with encrypted credentials
altastata_functions = AltaStataFunctions.from_account_dir(
    "/path/to/.altastata/accounts/my_account"
)
altastata_functions.set_password("your_password")

# Create fsspec filesystem
fs = create_filesystem(altastata_functions, "my_account")

# Load document securely using fsspec
with fs.open("Public/Documents/policy.txt", "r") as f:
    content = f.read()
    doc = Document(page_content=content, metadata={"source": "policy.txt"})

# Split for RAG
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents([doc])
```

## Complete RAG Pipeline Example

See **[test_rag.py](test_rag.py)** for a complete, working implementation!

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from altastata import AltaStataFunctions
from altastata.fsspec import create_filesystem

# Initialize secure connection
altastata_functions = AltaStataFunctions.from_account_dir(
    "/path/to/.altastata/accounts/my_account"
)
altastata_functions.set_password("your_password")

# Create fsspec filesystem
fs = create_filesystem(altastata_functions, "my_account")

# Load documents from encrypted storage
documents = []
for filename in ["policy.txt", "guidelines.txt"]:
    with fs.open(f"Public/Documents/{filename}", "r") as f:
        content = f.read()
        doc = Document(page_content=content, metadata={"source": filename})
        documents.append(doc)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Query the secure RAG system
query = "What is our data retention policy?"
relevant_docs = retriever.invoke(query)

for i, doc in enumerate(relevant_docs, 1):
    print(f"[Chunk {i}] {doc.page_content[:200]}...")
    print(f"Source: {doc.metadata['source']}\n")
```

### Test Output

Running `python test_rag.py` produces:

```
✅ AltaStata initialized
✅ Uploaded: company_policy.txt - DONE
✅ Uploaded: security_guidelines.txt - DONE
✅ Loaded: company_policy.txt (1024 chars)
✅ Loaded: security_guidelines.txt (1024 chars)
✅ Total documents loaded: 4
✅ Created 12 text chunks
✅ Vector store created successfully

📊 Query: What are the password requirements?
Found 2 relevant chunks:
[Chunk 1] (from: security_guidelines.txt)
Enterprise Security Guidelines
Password Requirements:
- Minimum 12 characters
- Mix of uppercase, lowercase, numbers, and special characters
- Change passwords every 90 days

📊 Query: How long do we keep financial records?
[Chunk 1] (from: company_policy.txt)
Data Retention Periods:
- Financial Records: 10 years from the end of the fiscal year
```

## Using fsspec Directly

AltaStata implements the fsspec interface, allowing direct file operations:

```python
from altastata import AltaStataFunctions
from altastata.fsspec import create_filesystem

# Initialize connection
altastata_functions = AltaStataFunctions.from_account_dir(
    "/path/to/.altastata/accounts/my_account"
)
altastata_functions.set_password("your_password")

# Create filesystem
fs = create_filesystem(altastata_functions, "my_account")

# List files
files = fs.ls("Public/Documents/")

# Read file
with fs.open("Public/Reports/summary.txt", "r") as f:
    content = f.read()

# Check file existence
exists = fs.exists("Public/Documents/file.txt")

# Get file info
info = fs.info("Public/Documents/file.txt")
```

## Running the Example

Try it yourself with the included test:

```bash
cd rag-example
python test_rag.py
```

This demonstrates:
- ✅ Uploading documents to encrypted storage
- ✅ Loading documents via fsspec
- ✅ Creating embeddings and vector store
- ✅ Semantic search with similarity scoring
- ✅ Automatic cleanup

## Confidential Computing Deployment

For maximum security, deploy with Confidential Container on Red Hat OpenShift:

```bash
cd confidential-gke
./setup-cluster.sh
```

Your RAG pipeline runs in hardware-encrypted memory where even the cloud provider cannot access data during processing.

## Security Comparison

| Feature | Traditional RAG | AltaStata RAG |
|---------|----------------|---------------|
| **Data at Rest** | Often unencrypted | AES-256 encrypted |
| **Memory Protection** | None | Confidential Container available |
| **Version Control** | Manual/none | Automatic & immutable |
| **Cloud Provider Access** | Possible | Zero-knowledge (impossible) |
| **Compliance** | Complex setup | Built-in GDPR/HIPAA/SOC2 |

## Best Practices

### Use Environment Variables
```python
import os
from dotenv import load_dotenv

load_dotenv()
altastata_functions = AltaStataFunctions.from_account_dir(
    os.environ["ALTASTATA_ACCOUNT_DIR"]
)
altastata_functions.set_password(os.environ["ALTASTATA_PASSWORD"])
```

### Add Audit Logging
```python
import logging

logger = logging.getLogger(__name__)
logger.info(f"Loading documents from: {secure_path}")
documents = loader.load()
logger.info(f"Loaded {len(documents)} documents")
```

## Conclusion

AltaStata transforms RAG security with:
- ✅ **Zero-Knowledge Encryption** - Cloud providers can't access your data
- ✅ **Seamless Integration** - Works with existing LangChain code
- ✅ **Enterprise Compliance** - Built-in GDPR, HIPAA, SOC 2 support
- ✅ **Simple API** - Just use `altastata://` URLs with any LangChain loader

**Secure your RAG. Secure your future. 🔒**

---

For more details, see:
- [RAG Security Architecture](RAG_SECURITY_ARCHITECTURE.md) - Detailed security design
- [test_rag.py](test_rag.py) - Complete working RAG pipeline test
- [fsspec Integration Guide](../fsspec-example/README.md) - fsspec usage and examples
- [AltaStata Main Documentation](../README.md) - Full AltaStata documentation

