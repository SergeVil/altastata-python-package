# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Altastata Python package - a library for cloud-based data processing with machine learning integration. The package provides seamless integration with PyTorch and TensorFlow through custom datasets that can handle data from both local filesystems and AltaStata cloud storage.

## Core Architecture

### Java Gateway System
- **BaseGateway** (`altastata/base_gateway.py`): Manages Java subprocess and Py4J gateway connection
- **AltaStataFunctions** (`altastata/altastata_functions.py`): Main API class for cloud operations
- Java JARs in `altastata/lib/`: Contains required dependencies for cloud operations
- Port 25333: Default gateway port for Java communication

### Dataset Integration Pattern
- **Account Registry**: Global dictionaries store AltaStataFunctions instances by account ID
- **Registration Functions**: `register_altastata_functions_for_pytorch()` and `register_altastata_functions_for_tensorflow()`
- **Dual Mode Support**: Datasets automatically fall back to local filesystem when cloud functions unavailable

### Cloud File Versioning
- Files have version suffixes: `filename.ext✹timestamp_version`
- Version parsing: Split on `✹`, extract timestamp from suffix
- Cache system: Small files (<16MB) cached in memory with 1GB total limit

## Common Development Commands

### Package Building and Testing
```bash
# Install in development mode
pip install -e .

# Build package
python -m build

# Run basic tests
python test_script.py

# Test PyTorch example
cd examples/pytorch-example && python training_example.py

# Test TensorFlow example 
cd examples/tensorflow-example && python training_example.py
```

### Docker Development
```bash
# Build container (use image matching your platform: arm64 or amd64)
docker build -t altastata/jupyter-datascience-arm64:latest -f containers/jupyter/Dockerfile.arm64 .   # Apple Silicon
# or: docker buildx build --platform linux/amd64 -t altastata/jupyter-datascience-amd64:latest -f containers/jupyter/Dockerfile.amd64 --load .

# Run with volume mounts
docker run -d -p 8888:8888 \
  -v ~/.altastata:/opt/app-root/src/.altastata:rw \
  -v ~/Desktop:/opt/app-root/src/Desktop:rw \
  altastata/jupyter-datascience-arm64:latest   # or jupyter-datascience-amd64
```

### RAG Open LLM on IBM Z (s390x)
- **Build on server:** `./containers/rag-example/build-rag-s390x-on-server.sh` (syncs repo, builds image on LinuxONE; set `SSH_HOST`, `SSH_KEY` if needed).
- **Push to ICR:** `./containers/rag-example/push-rag-s390x-to-icr-from-server.sh` (after build). **Run:** `./containers/rag-example/pull-and-run-rag-s390x-from-icr.sh`.
- **Image:** `containers/rag-example/Dockerfile.open_llm_s390x` (base: ibmz-accelerated-for-pytorch; deps layer cached when requirements.txt unchanged). See `examples/rag-example/open_llm/README.md` for full s390x docs.

### RAG Open LLM on Mac (same layout as s390x)
- **Build and run:** `ALTASTATA_ACCOUNT_DIR=$HOME/.altastata/accounts/amazon.rsa.bob123 ./containers/rag-example/build-and-run-rag-mac.sh` (AltaStata only, index at startup).
- **Image:** `containers/rag-example/Dockerfile.open_llm_mac` (python:3.11-slim; no sample_documents, same entrypoint as s390x).
- **Faster answers:** Run Ollama on the host (e.g. `ollama run smollm2:360m`), then `LLM_PROVIDER=ollama` with same script; script passes `OLLAMA_BASE_URL=http://host.docker.internal:11434`.
- **Query path:** Chunk reads from AltaStata are parallelized (ThreadPoolExecutor). Default Transformers model is SmolLM2-360M-Instruct. Entrypoint logs account dir and indexer result for debugging.

### JAR Management
```bash
# Verify JAR integrity
jar tf altastata/lib/py4j0.10.9.5.jar | grep GatewayServer

# Download fresh py4j JAR if corrupted
wget https://repo1.maven.org/maven2/net/sf/py4j/py4j/0.10.9.5/py4j-0.10.9.5.jar -O altastata/lib/py4j0.10.9.5.jar

# Build AltaStata Hadoop JAR (external dependency)
# Note: -PnoGCP=true is no longer needed as the increased the size for altastata package
gradle clean build shadowJar -PexcludeBouncyCastle=true -PminimalBuild=true copyDeps
```

## Configuration Patterns

### Account Setup
```python
# Initialize with credentials
altastata_functions = AltaStataFunctions.from_credentials(user_properties, private_key)
altastata_functions.set_password("password")

# Register for framework
register_altastata_functions_for_pytorch(altastata_functions, "account_id")
register_altastata_functions_for_tensorflow(altastata_functions, "account_id")
```

### Dataset Usage
```python
# PyTorch dataset
dataset = AltaStataPyTorchDataset(
    "account_id",
    root_dir="data/images", 
    file_pattern="*.png",
    transform=transform
)

# TensorFlow dataset
dataset = AltaStataTensorFlowDataset(
    "account_id",
    root_dir="data/images",
    file_pattern="*.png", 
    preprocess_fn=preprocess_fn
)
```

## File Structure Guidelines

- `altastata/`: Main package code
- `examples/`: Runnable examples — `pytorch-example/`, `tensorflow-example/`, `fsspec-example/`, `event-listener-example/`, `rag-example/`
- `containers/jupyter/`: Jupyter DataScience Dockerfiles (arm64, amd64, s390x) and build scripts (`build-all-images.sh`, `push-to-ghcr.sh`)
- `containers/rag-example/`: RAG Open LLM Dockerfiles and scripts (Mac, s390x)
- `containers/confidential-gke/`: GKE deployment (jupyter-deployment.yaml, setup-cluster.sh)
- JAR files must be in `altastata/lib/` for Java gateway functionality

## Key Implementation Notes

- Always check account registry before using cloud functions
- File operations automatically fall back to local filesystem
- Memory mapping used for large files (>16MB) to optimize performance
- Custom callbacks save models directly to cloud storage
- Thread-safe design for multi-worker data loading