# AltaStata Python Package Developer Guide v0.1.15

## Prerequisites

### JAR Files Setup
1. Ensure you have the required JAR files in the `altastata/lib` directory:
   ```bash
   # For Windows example:
   cp /c/Users/serge/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/local-packages/share/py4j/py4j0.10.9.8.jar altastata/lib/
   ```

2. Build and copy the AltaStata Hadoop JAR:
   ```bash
   # Go to altastata-hadoop directory
   # gradle clean build shadowJar -PexcludeBouncyCastle=true copyDeps
   # Note: -PnoGCP=true is no longer needed as the increased the size for altastata package
   gradle clean build shadowJar -PexcludeBouncyCastle=true -PminimalBuild=true copyDeps
   
   # Copy the built JARs
   cp ../mycloud/altastata-hadoop/build/libs/altastata-hadoop-all.jar altastata/lib/
   cp ../mycloud/altastata-hadoop/build/libs_dependency/bc*-jdk18on-*.jar altastata/lib/
   ```

3. Verify JAR integrity:
   ```bash
   # Check if py4j JAR is valid
   jar tf altastata/lib/py4j0.10.9.5.jar | grep GatewayServer
   
   # If corrupted, download a fresh copy
   wget https://repo1.maven.org/maven2/net/sf/py4j/py4j/0.10.9.5/py4j-0.10.9.5.jar -O altastata/lib/py4j0.10.9.5.jar
   ```

### Logging Configuration
- To customize logging, copy and modify the logback configuration:
  ```bash
  cp ../mycloud/altastata-hadoop/src/main/resources/logback.xml altastata/lib/
  ```

## Local Development

### Installation
```bash
# Install the package in development mode
pip install -e .

# Install from pypi.org
pip install altastata

# Run tests
python test_script.py

## Build and Upload

Use the comprehensive build script for easy deployment:

```bash
# Complete build and upload process
./build-and-upload.sh

# Or run individual steps:
python -m build                    # Build Python package
twine upload dist/*               # Upload to PyPI
./build-all-images.sh             # Build Docker images
./push-to-ghcr.sh                # Push to GHCR
```
```## Docker Deployment

### Version Management

The Docker image version is centrally managed in `version.sh`. To update the version:

1. Edit `version.sh` and update the `VERSION` variable
2. Run `./update-version.sh` to sync the version to all configuration files

All build scripts automatically use the version from `version.sh`.

### Multi-Architecture Support

The project builds **multi-architecture images** for AMD64 and ARM64. The
s390x (IBM Z/LinuxONE) image is built separately using `openshift/Dockerfile.s390x`.

- **AMD64 (x86_64)**: Native performance on Intel/AMD processors and GCP nodes
- **ARM64**: Native performance on Apple Silicon Macs, with emulation support on other platforms
- **s390x**: Build and tag separately for IBM Z/LinuxONE

### Building Multi-Architecture Images
```bash
# Build multi-architecture image (AMD64 + ARM64)
./build-all-images.sh

# This creates:
# - ghcr.io/sergevil/altastata/jupyter-datascience:latest (multi-arch)
# - ghcr.io/sergevil/altastata/jupyter-datascience:${VERSION} (multi-arch, version from version.sh)
```

### Manual Multi-Architecture Build
```bash
# Build and push multi-architecture image
# Note: Use ./push-to-ghcr.sh instead, which automatically uses version from version.sh
source version.sh
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file openshift/Dockerfile.amd64 \
  --tag ghcr.io/sergevil/altastata/jupyter-datascience:${VERSION} \
  --push \
  .
```

### Running the Container

**Option A: Use pre-built image from GHCR (recommended — no build)**

The image is published to GitHub Container Registry, not Docker Hub. Use the GHCR compose file so the container runs without building:

```bash
# Run from GitHub Container Registry (no build, fast start)
docker-compose -f docker-compose-ghcr.yml up -d
```

Then open **http://localhost:8888**. Use the token from logs: `docker-compose -f docker-compose-ghcr.yml logs | grep token`.

**Option B: Single `docker run` with GHCR image**

```bash
source version.sh
docker run \
  --name altastata-jupyter \
  -d \
  -p 8888:8888 \
  -v /Users/sergevilvovsky/.altastata:/opt/app-root/src/.altastata:rw \
  -v /Users/sergevilvovsky/Desktop:/opt/app-root/src/Desktop:rw \
  ghcr.io/sergevil/altastata/jupyter-datascience:${VERSION}
```

**Option C: Local build with docker-compose**

`docker-compose up -d` uses image `altastata/jupyter-datascience:${VERSION}` (Docker Hub). That image does not exist there, so Compose will **build** the image locally (can take 10–20 minutes). To use this path:

```bash
./update-version.sh   # writes VERSION to .env
docker-compose up -d  # builds then runs
```

**IBM Z / LinuxONE (s390x)**

```bash
docker run \
  --name altastata-jupyter-s390x \
  -d \
  -p 8888:8888 \
  icr.io/altastata/jupyter-datascience-s390x:2026c
```

**If the container won’t run**

- **“pull access denied” for `altastata/jupyter-datascience`** — Use the GHCR image: `docker-compose -f docker-compose-ghcr.yml up -d` (see Option A).
- **“container name already in use”** — Remove the existing container: `docker rm -f altastata-jupyter`, then start again.
- **Port 8888 in use** — Stop the process using it or change the host port, e.g. `"8889:8888"` in the compose file.
- **Get Jupyter URL/token** — `docker logs altastata-jupyter 2>&1 | grep -E "http://|token="`

### Platform Compatibility
- **Apple Silicon Macs**: Native ARM64 performance
- **Intel Macs**: Native AMD64 performance  
- **GCP Confidential GKE**: Native AMD64 performance
- **IBM Z and LinuxONE**: Use the `jupyter-datascience-s390x` image
- **Other platforms**: Automatic architecture selection for AMD64/ARM64

## PyPI Package Management

### Prerequisites
```bash
# Install required tools
pip install --upgrade pip build twine
```

### Building the Package
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build the package
python -m build

# Verify the built package
twine check dist/*
```

### Uploading to PyPI
```bash
# Upload to PyPI Test (optional)
# twine upload --repository testpypi dist/*

# Upload to PyPI Production
twine upload dist/*
```

### PyPI Configuration
1. Create a PyPI account if you don't have one
2. Generate an API token:
   - Go to https://pypi.org/manage/account/token/
   - Create a new token with appropriate permissions
   - Store the token securely

3. Configure `.pypirc`:
   ```ini
   [pypi]
   username = __token__
   password = pypi-your-token-here
   ```

### Large File Handling
For large files (like JARs), you may need to request a file size limit increase:
- Documentation: https://docs.pypi.org/project-management/storage-limits/#requesting-a-file-size-limit-increase
- Support Issue: https://github.com/pypi/support/issues/6225

### Project Links
- PyPI Project: https://pypi.org/project/altastata/


