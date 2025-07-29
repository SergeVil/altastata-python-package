#!/bin/bash

# Altastata Python Package Docker Images Build Script
# This script builds separate Docker images for amd64 and arm64 architectures

echo "🚀 Building Altastata Python Package Docker Images for separate architectures..."

# Create Docker network if it doesn't exist (shared with main altastata project)
echo "Creating shared Docker network..."
docker network create altastata-network 2>/dev/null || echo "Network altastata-network already exists (shared with main project)"

# Create and use a new builder instance for multi-architecture builds
echo "🔧 Setting up Docker buildx for multi-architecture builds..."
docker buildx create --name altastata-builder --use 2>/dev/null || docker buildx use altastata-builder

# Build separate architecture images locally
echo "🏗️  Building separate architecture images locally..."

# Build for amd64 using the AMD64-specific Dockerfile
echo "📦 Building jupyter-datascience-amd64 image..."
docker buildx build \
    --platform linux/amd64 \
    --file openshift/Dockerfile.amd64 \
    --tag altastata/jupyter-datascience-amd64:latest \
    --load \
    .

# Build for arm64 using the ARM64-specific Dockerfile
echo "📦 Building jupyter-datascience-arm64 image..."
docker buildx build \
    --platform linux/arm64 \
    --file openshift/Dockerfile.arm64 \
    --tag altastata/jupyter-datascience-arm64:latest \
    --load \
    .

# Tag for GHCR (separate architectures)
echo "🏷️  Tagging images for GHCR..."
docker tag altastata/jupyter-datascience-amd64:latest ghcr.io/sergevil/altastata/jupyter-datascience-amd64:latest
docker tag altastata/jupyter-datascience-amd64:latest ghcr.io/sergevil/altastata/jupyter-datascience-amd64:2025a_latest
docker tag altastata/jupyter-datascience-arm64:latest ghcr.io/sergevil/altastata/jupyter-datascience-arm64:latest
docker tag altastata/jupyter-datascience-arm64:latest ghcr.io/sergevil/altastata/jupyter-datascience-arm64:2025a_latest

echo ""
echo "✅ Separate architecture images built successfully!"
echo ""
echo "📦 Local images:"
echo "- altastata/jupyter-datascience-amd64:latest (AMD64)"
echo "- altastata/jupyter-datascience-arm64:latest (ARM64)"
echo ""
echo "🏷️  GHCR images (tagged but not pushed):"
echo "AMD64:"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience-amd64:latest"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience-amd64:2025a_latest"
echo ""
echo "ARM64:"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience-arm64:latest"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience-arm64:2025a_latest"
echo ""
echo "🚀 To push to GHCR, run: ./push-to-ghcr.sh"
echo "🔧 To run locally, use: docker-compose up -d (local image)"
echo "🌐 To run from GHCR, use: docker-compose -f docker-compose-ghcr.yml up -d" 