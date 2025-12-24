#!/bin/bash

# Altastata Python Package Docker Image Build Script
# This script builds a local Docker image for development/testing (AMD64)
# For multi-architecture builds and pushing to GHCR, use push-to-ghcr.sh

# Load version configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/version.sh"

# Validate version is set
if [ -z "$VERSION" ]; then
    echo "Error: VERSION is not set in version.sh"
    echo "Please ensure version.sh contains: VERSION=\"your_version\""
    exit 1
fi

echo "🚀 Building Altastata Python Package Docker Image..."

# Create Docker network if it doesn't exist (shared with main altastata project)
echo "Creating shared Docker network..."
docker network create altastata-network 2>/dev/null || echo "Network altastata-network already exists (shared with main project)"

# Create and use a new builder instance for builds
echo "🔧 Setting up Docker buildx for builds..."
docker buildx create --name altastata-builder --use 2>/dev/null || docker buildx use altastata-builder

# Build image for local development use
echo "🏗️  Building image for local development (AMD64)..."

# Build for local use only (AMD64 for local Docker daemon)
# Note: For multi-arch builds and pushing to GHCR, use push-to-ghcr.sh instead
echo "📦 Building jupyter-datascience image for local use (AMD64)..."
docker buildx build \
    --platform linux/amd64 \
    --file openshift/Dockerfile.amd64 \
    --tag altastata/jupyter-datascience:latest \
    --tag altastata/jupyter-datascience:${VERSION} \
    --load \
    .

echo ""
echo "✅ Local image built successfully!"
echo ""
echo "📦 Local Docker daemon:"
echo "   - altastata/jupyter-datascience:latest"
echo "   - altastata/jupyter-datascience:${VERSION}"
echo ""
echo "🚀 To build and push multi-architecture images (AMD64, ARM64, s390x) to GHCR, run: ./push-to-ghcr.sh"
echo "🔧 To run locally, use: docker-compose up -d"
echo "🌐 To run from GHCR, use: docker-compose -f docker-compose-ghcr.yml up -d" 