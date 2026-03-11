#!/bin/bash

# Altastata Python Package Docker Image Build Script
# This script builds both ARM64 and AMD64 images for local development/testing (Mac).
# Run from repo root: ./containers/jupyter/build-all-images.sh
# For pushing to GHCR, use containers/jupyter/push-to-ghcr.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/version.sh"

# Validate version is set
if [ -z "$VERSION" ]; then
    echo "Error: VERSION is not set in version.sh"
    echo "Please ensure version.sh contains: VERSION=\"your_version\""
    exit 1
fi

echo "🚀 Building Altastata Python Package Docker Images (ARM64 + AMD64)..."

# Create Docker network if it doesn't exist (shared with main altastata project)
echo "Creating shared Docker network..."
docker network create altastata-network 2>/dev/null || echo "Network altastata-network already exists (shared with main project)"

# Create and use buildx builder
echo "🔧 Setting up Docker buildx..."
docker buildx create --name altastata-builder --use 2>/dev/null || docker buildx use altastata-builder

# Build ARM64 image (context = repo root)
echo ""
echo "🏗️  Building jupyter-datascience-arm64..."
cd "$REPO_ROOT"
docker build -f containers/jupyter/Dockerfile.arm64 \
    -t altastata/jupyter-datascience-arm64:latest \
    -t altastata/jupyter-datascience-arm64:${VERSION} \
    .

# Build AMD64 image
echo ""
echo "🏗️  Building jupyter-datascience-amd64..."
docker buildx build --platform linux/amd64 -f containers/jupyter/Dockerfile.amd64 \
    -t altastata/jupyter-datascience-amd64:latest \
    -t altastata/jupyter-datascience-amd64:${VERSION} \
    --load \
    .

echo ""
echo "✅ Both images built successfully!"
echo ""
echo "📦 Local Docker daemon:"
echo "   - altastata/jupyter-datascience-arm64:latest and :${VERSION}"
echo "   - altastata/jupyter-datascience-amd64:latest and :${VERSION}"
echo ""
echo "🚀 To push both to GHCR, run: ./containers/jupyter/push-to-ghcr.sh"
echo "🔧 To run locally, use: docker-compose up -d"
echo "🌐 To run from GHCR, use: docker-compose -f docker-compose-ghcr.yml up -d"
