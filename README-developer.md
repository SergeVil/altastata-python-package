# AltaStata Python Package Developer Guide

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
   gradle clean build shadowJar -PexcludeBouncyCastle=true -PminimalBuild=true -PnoGCP=true copyDeps
   
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
```## Docker Deployment

### Building the Image
```bash

# build test
docker build -t altastata/jupyter-datascience:latest -f openshift/Dockerfile .

# Build multi-architecture image
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  -t ghcr.io/sergevil/altastata/jupyter-datascience:2025a_latest \
  -f openshift/Dockerfile .
```

### Pushing to Registry
```bash
# Push to GitHub Container Registry
docker push ghcr.io/sergevil/altastata/jupyter-datascience:2025a_latest
```

### Running the Container
```bash
docker run \
  --name altastata-jupyter \
  -d \
  -p 8888:8888 \
  -v /Users/sergevilvovsky/.altastata:/opt/app-root/src/.altastata:rw \
  -v /Users/sergevilvovsky/Desktop:/opt/app-root/src/Desktop:rw \
  ghcr.io/sergevil/altastata/jupyter-datascience:2025a_latest
```

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


