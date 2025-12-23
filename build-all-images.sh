#!/bin/bash

# Altastata Python Package Docker Image Build Script
# This script builds multi-architecture Docker images (AMD64, ARM64, s390x)

echo "🚀 Building Altastata Python Package Docker Image..."

# Create Docker network if it doesn't exist (shared with main altastata project)
echo "Creating shared Docker network..."
docker network create altastata-network 2>/dev/null || echo "Network altastata-network already exists (shared with main project)"

# Create and use a new builder instance for builds
echo "🔧 Setting up Docker buildx for builds..."
docker buildx create --name altastata-builder --use 2>/dev/null || docker buildx use altastata-builder

# Build multi-architecture image locally first
echo "🏗️  Building multi-architecture image locally (AMD64 + ARM64 + s390x)..."

# Build for amd64, arm64, and s390x using the AMD64-specific Dockerfile
# Note: Building all architectures - they will be stored in buildx cache
echo "📦 Building jupyter-datascience image for all architectures (AMD64, ARM64, s390x)..."
docker buildx build \
    --platform linux/amd64,linux/arm64,linux/s390x \
    --file openshift/Dockerfile.amd64 \
    --tag altastata/jupyter-datascience:latest \
    --tag altastata/jupyter-datascience:2025i_latest \
    --tag ghcr.io/sergevil/altastata/jupyter-datascience:latest \
    --tag ghcr.io/sergevil/altastata/jupyter-datascience:2025i_latest \
    .

echo ""
echo "✅ Multi-architecture images built successfully!"
echo ""
echo "📦 Built architectures (all in buildx cache, ready to push):"
echo "   - linux/amd64"
echo "   - linux/arm64"
echo "   - linux/s390x"
echo ""
echo "🏷️  GHCR images (tagged and ready to push):"
echo "   - ghcr.io/sergevil/altastata/jupyter-datascience:latest"
echo "   - ghcr.io/sergevil/altastata/jupyter-datascience:2025i_latest"
echo ""
echo "🚀 To push to GHCR, run: ./push-to-ghcr.sh"
echo "🌐 To run from GHCR, use: docker-compose -f docker-compose-ghcr.yml up -d" 