# AltaStata Python Package Developer Guide

> Package version is tracked in `setup.py` (`pip show altastata` to inspect).

## Prerequisites

### Bundled artifacts setup

The wheel ships with two binary artifacts that this repo deliberately does
not commit to git:

- `altastata/lib/altastata-grpc-<ver>-uber.jar` (built from `mycloud/altastata-grpc`)
- `altastata/lib/altastata-console-static/` (built from `altastata-console/frontend`)

Only `altastata/lib/py4j0.10.9.5.jar` is tracked in git, since it is a fixed
upstream artifact rather than something we build. Everything else under
`altastata/lib/` is gitignored (`lib/` rule in `.gitignore`) and must be
populated locally before `pip install -e .` or `python -m build`.

#### Quick start (recommended)

Run the helper script — it builds both artifacts in one shot and stages
them under `altastata/lib/`:

```bash
bash scripts/build-bundled-artifacts.sh
```

The script assumes the sibling layout `mycloud/` and `altastata-console/`
next to this repo. Override with `ALTASTATA_MYCLOUD_DIR` /
`ALTASTATA_CONSOLE_DIR` if your layout differs, or pass `SKIP_GRPC=1` /
`SKIP_UI=1` to leave one side untouched.

#### Manual fallback

If you prefer to drive each build yourself:

1. Build the gRPC uber jar in `mycloud/altastata-grpc`:
   ```bash
   (cd ../mycloud && ./gradlew :altastata-grpc:shadowJar)
   cp ../mycloud/altastata-grpc/build/libs/altastata-grpc-*-uber.jar altastata/lib/
   ```

2. Build the Console SPA in `altastata-console/frontend`:
   ```bash
   (cd ../altastata-console/frontend && npm install && npm run build)
   rm -rf altastata/lib/altastata-console-static
   mkdir -p altastata/lib/altastata-console-static
   cp -R ../altastata-console/frontend/dist/. altastata/lib/altastata-console-static/
   ```

3. Verify py4j integrity (only relevant if you suspect corruption):
   ```bash
   jar tf altastata/lib/py4j0.10.9.5.jar | grep GatewayServer
   # If corrupted, fetch a fresh copy
   wget https://repo1.maven.org/maven2/net/sf/py4j/py4j/0.10.9.5/py4j-0.10.9.5.jar \
        -O altastata/lib/py4j0.10.9.5.jar
   ```

4. Run the server (Java + bundled SPA on the same port):
   ```bash
   altastata-grpc-server
   # gRPC + UI both on http://127.0.0.1:9877
   ```

   The launcher exports `ALTASTATA_WEB_UI_DIR` automatically when the
   bundled `altastata/lib/altastata-console-static/` is present, so the Java
   gRPC gateway serves the SPA from the same port. Set
   `ALTASTATA_WEB_UI_DIR=` (empty) before running to disable the UI and
   keep gRPC-only routing for legacy testing.

### Logging Configuration
- To customize logging, copy and modify the logback configuration:
  ```bash
  # Optional: provide your own logback.xml in altastata/lib/
  ```

## Local Development

### Installation
```bash
# Install the package in development mode
pip install -e .

# Install from pypi.org
pip install altastata

# Run smoke tests (core API)
python examples/smoke-test/test_script.py

## Build and Upload

Use the comprehensive build script for easy deployment:

```bash
# Complete build and upload process
./containers/jupyter/build-and-upload.sh

# Or run individual steps:
python -m build                                    # Build Python package
twine upload dist/*                               # Upload to PyPI
./containers/jupyter/build-all-images.sh           # Build Docker images
./containers/jupyter/push-to-ghcr.sh               # Push to GHCR
```

## Docker Deployment

Full guide: [containers/jupyter/README-Docker.md](containers/jupyter/README-Docker.md)

### Version Management

The Docker image version is centrally managed in `version.sh`. To update the version:

1. Edit `version.sh` and update the `VERSION` variable
2. Run `./update-version.sh` to sync the version to all configuration files

All build scripts automatically use the version from `version.sh`.

### Multi-Architecture Support

The project builds **architecture-specific images** for AMD64 and ARM64.
Each architecture has its own GHCR package. The s390x (IBM Z/LinuxONE) image
is built separately using `containers/jupyter/Dockerfile.s390x`.

- **AMD64 (x86_64)**: `jupyter-datascience-amd64` — Intel/AMD processors and GCP nodes
- **ARM64**: `jupyter-datascience-arm64` — Apple Silicon Macs and ARM servers
- **s390x**: Build and tag separately for IBM Z/LinuxONE

### Building and Pushing Images
```bash
# Build and push architecture-specific images to GHCR
./containers/jupyter/push-to-ghcr.sh

# This creates:
# - ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION}
# - ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}
```

### Running the Container

**Option A: Use pre-built image from GHCR (recommended — no build)**

The image is published to GitHub Container Registry, not Docker Hub. Use the GHCR compose file so the container runs without building:

```bash
# Run from GitHub Container Registry (no build, fast start)
docker compose -f containers/jupyter/docker-compose-ghcr.yml up -d
```

Then open **http://localhost:8888/lab**. The token is generated at startup; get it from the logs:

```bash
docker compose -f containers/jupyter/docker-compose-ghcr.yml logs altastata-jupyter 2>&1 | grep -E "127.0.0.1:8888|token"
# or
docker exec altastata-jupyter jupyter server list
```
Use the URL with `?token=...` from the output, or paste the token on the login page. If you still see an old fixed token, rebuild the image and recreate the container (see containers/jupyter/README-Docker.md).

**Option B: Single `docker run` with GHCR image**

```bash
source version.sh
# Use image matching your platform (arm64 for Apple Silicon, amd64 for Intel/AMD)
docker run \
  --name altastata-jupyter \
  -d \
  -p 8888:8888 \
  -v ~/.altastata:/opt/app-root/src/.altastata:rw \
  -v ~/Desktop:/opt/app-root/src/Desktop:rw \
  ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION}   # or jupyter-datascience-amd64
```

**Option C: Local build with docker-compose**

`docker compose -f containers/jupyter/docker-compose.yml up -d` uses image `altastata/jupyter-datascience-${ARCH}:${VERSION}` (ARCH is arm64 or amd64). Compose may **build** the image locally (can take 10–20 minutes). To use this path:

```bash
./update-version.sh   # writes VERSION and ARCH to .env (detects your platform)
docker compose -f containers/jupyter/docker-compose.yml up -d  # builds then runs
```

Then open **http://localhost:8888/lab**. Get the generated token from the logs: `docker compose -f containers/jupyter/docker-compose.yml logs altastata-jupyter 2>&1 | grep -E "127.0.0.1:8888|token"` or `docker exec altastata-jupyter jupyter server list`.

**IBM Z / LinuxONE (s390x)**

```bash
docker run \
  --name altastata-jupyter-s390x \
  -d \
  -p 8888:8888 \
  icr.io/altastata/jupyter-datascience-s390x:${VERSION}
```

**If the container won’t run**

- **“pull access denied” **— Use the GHCR image for your platform: `ghcr.io/sergevil/altastata/jupyter-datascience-arm64` or `jupyter-datascience-amd64` with your version.
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


