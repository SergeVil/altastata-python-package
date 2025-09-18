#!/bin/bash

# Altastata Python Package Docker Image Build Script
# This script builds AMD64 Docker image (works on all platforms)

echo "🚀 Building Altastata Python Package Docker Image..."

# Create Docker network if it doesn't exist (shared with main altastata project)
echo "Creating shared Docker network..."
docker network create altastata-network 2>/dev/null || echo "Network altastata-network already exists (shared with main project)"

# Create and use a new builder instance for builds
echo "🔧 Setting up Docker buildx for builds..."
docker buildx create --name altastata-builder --use 2>/dev/null || docker buildx use altastata-builder

# Build multi-architecture image
echo "🏗️  Building multi-architecture image (AMD64 + ARM64)..."

# Build for both amd64 and arm64 using the AMD64-specific Dockerfile
echo "📦 Building jupyter-datascience image..."
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --file openshift/Dockerfile.amd64 \
    --tag altastata/jupyter-datascience:latest \
    --push \
    .

# Tag for GHCR (already pushed above)
echo "🏷️  Tagging additional versions for GHCR..."
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --file openshift/Dockerfile.amd64 \
    --tag ghcr.io/sergevil/altastata/jupyter-datascience:2025d_latest \
    --push \
    .

echo ""
echo "✅ Multi-architecture image built and pushed successfully!"
echo ""
echo "🏷️  GHCR images (pushed):"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience:latest (AMD64 + ARM64)"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience:2025d_latest (AMD64 + ARM64)"
echo ""
echo "🚀 To push to GHCR, run: ./push-to-ghcr.sh"
echo "🔧 To run locally, use: docker-compose up -d (local image)"
echo "🌐 To run from GHCR, use: docker-compose -f docker-compose-ghcr.yml up -d" 