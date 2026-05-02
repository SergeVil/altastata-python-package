#!/bin/bash

# Altastata Python Package GHCR Push Script
# This script builds and pushes Docker images to GitHub Container Registry
# Run from repo root: ./containers/jupyter/push-to-ghcr.sh
# Default: arm64 (Apple Silicon) only — native build + push.
# SKIP_JUPYTER_ARM64_BUILD=1 skips `docker build` and only runs `docker push` (tag must exist locally).
# PUSH_JUPYTER_AMD64_TO_GHCR=1 also builds linux/amd64 via buildx (slow on Mac).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/version.sh"

if [ -z "$JUPYTER_VERSION" ]; then
    echo "Error: JUPYTER_VERSION not set in version.sh"
    exit 1
fi
if [ -z "$ALTASTATA_PYPI_VERSION" ]; then
    echo "Error: ALTASTATA_PYPI_VERSION not extracted from setup.py (see version.sh)"
    exit 1
fi
# Local alias so the rest of the script (and echo lines) can keep using $VERSION.
VERSION="$JUPYTER_VERSION"

# Check if GitHub token is set
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable not set"
    echo "Please set your GitHub Personal Access Token:"
    echo "export GITHUB_TOKEN=your_token_here"
    echo "Or get it from: https://github.com/settings/tokens"
    exit 1
fi

# Login to GHCR
echo "Logging in to GitHub Container Registry..."
echo $GITHUB_TOKEN | docker login ghcr.io -u sergevil --password-stdin

if [[ "${PUSH_JUPYTER_AMD64_TO_GHCR:-}" == "1" ]]; then
    echo "🔧 Setting up Docker buildx (needed for amd64 cross-build)..."
    docker buildx create --name altastata-builder --use 2>/dev/null || docker buildx use altastata-builder
fi

echo "Building and pushing jupyter-datascience image(s) to GHCR..."
if [[ "${PUSH_JUPYTER_AMD64_TO_GHCR:-}" == "1" ]]; then
    echo "Packages: jupyter-datascience-arm64, jupyter-datascience-amd64"
else
    echo "Packages: jupyter-datascience-arm64 only (set PUSH_JUPYTER_AMD64_TO_GHCR=1 to include amd64)"
fi
if [[ "${SKIP_JUPYTER_ARM64_BUILD:-}" == "1" ]]; then
    echo "SKIP_JUPYTER_ARM64_BUILD=1: will not rebuild arm64; push only if tag exists locally."
fi
echo ""

cd "$REPO_ROOT"

ARM64_IMG="ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION}"

if [[ "${SKIP_JUPYTER_ARM64_BUILD:-}" == "1" ]]; then
    echo "Pushing existing jupyter-datascience-arm64 (altastata ${ALTASTATA_PYPI_VERSION}), tag ${VERSION}..."
    if ! docker image inspect "${ARM64_IMG}" >/dev/null 2>&1; then
        echo "Error: no local image ${ARM64_IMG}"
        echo "Build first, or retag your image:"
        echo "  docker tag <your-image-id> ${ARM64_IMG}"
        exit 1
    fi
    docker push "${ARM64_IMG}"
else
    echo "Building and pushing jupyter-datascience-arm64 (altastata ${ALTASTATA_PYPI_VERSION})..."
    docker build -f containers/jupyter/Dockerfile.arm64 \
        --build-arg ALTASTATA_VERSION=${ALTASTATA_PYPI_VERSION} \
        -t "${ARM64_IMG}" . && \
        docker push "${ARM64_IMG}"
fi

if [[ "${PUSH_JUPYTER_AMD64_TO_GHCR:-}" == "1" ]]; then
    echo ""
    echo "Building and pushing jupyter-datascience-amd64 (altastata ${ALTASTATA_PYPI_VERSION})..."
    docker buildx build --platform linux/amd64 --file containers/jupyter/Dockerfile.amd64 \
        --build-arg ALTASTATA_VERSION=${ALTASTATA_PYPI_VERSION} \
        --tag ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION} \
        --push .
fi

echo ""
echo "✅ Push to GHCR finished."
echo ""
echo "Images:"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION}"
if [[ "${PUSH_JUPYTER_AMD64_TO_GHCR:-}" == "1" ]]; then
    echo "- ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}"
fi
echo ""

echo "Pull:"
echo "  ARM64 (Apple Silicon): docker pull ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION}"
if [[ "${PUSH_JUPYTER_AMD64_TO_GHCR:-}" == "1" ]]; then
    echo "  AMD64:                 docker pull ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}"
fi
