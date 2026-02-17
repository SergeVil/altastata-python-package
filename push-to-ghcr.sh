#!/bin/bash

# Altastata Python Package GHCR Push Script
# This script builds and pushes Docker images to GitHub Container Registry
# Separate packages: jupyter-datascience-arm64, jupyter-datascience-amd64

# Load version configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/version.sh"

# Validate version is set
if [ -z "$VERSION" ]; then
    echo "Error: VERSION is not set in version.sh"
    echo "Please ensure version.sh contains: VERSION=\"your_version\""
    exit 1
fi

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

# Create and use buildx builder instance if it doesn't exist
echo "🔧 Setting up Docker buildx..."
docker buildx create --name altastata-builder --use 2>/dev/null || docker buildx use altastata-builder

echo "Building and pushing jupyter-datascience images to GHCR..."
echo "Packages: jupyter-datascience-arm64, jupyter-datascience-amd64"
echo ""

# Build and push jupyter-datascience-arm64
echo "Building and pushing jupyter-datascience-arm64..."
docker build -f openshift/Dockerfile.arm64 -t ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION} . && \
    docker push ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION}

# Build and push jupyter-datascience-amd64
echo ""
echo "Building and pushing jupyter-datascience-amd64..."
docker buildx build --platform linux/amd64 --file openshift/Dockerfile.amd64 \
    --tag ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION} \
    --push .

echo ""
echo "✅ Images pushed successfully to GHCR!"
echo ""
echo "Images available at:"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION}"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}"
echo ""

echo "To deploy from GHCR, pull the image for your platform:"
echo "  ARM64 (Apple Silicon): docker pull ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION}"
echo "  AMD64:                 docker pull ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}" 