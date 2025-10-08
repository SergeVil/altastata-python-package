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

# Build multi-architecture image locally first
echo "🏗️  Building multi-architecture image locally (AMD64 + ARM64)..."

# Build for both amd64 and arm64 using the AMD64-specific Dockerfile
echo "📦 Building jupyter-datascience image..."
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --file openshift/Dockerfile.amd64 \
    --tag altastata/jupyter-datascience:latest \
    --load \
    .

# Tag for GHCR
echo "🏷️  Tagging image for GHCR..."
docker tag altastata/jupyter-datascience:latest ghcr.io/sergevil/altastata/jupyter-datascience:latest
docker tag altastata/jupyter-datascience:latest ghcr.io/sergevil/altastata/jupyter-datascience:2025f_latest

echo ""
echo "✅ Multi-architecture image built successfully!"
echo ""
echo "📦 Local image:"
echo "- altastata/jupyter-datascience:latest (works on all platforms)"
echo ""
echo "🏷️  GHCR images (tagged but not pushed):"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience:latest"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience:2025f_latest"
echo ""
echo "🚀 To push to GHCR, run: ./push-to-ghcr.sh"
echo "🔧 To run locally, use: docker-compose up -d (local image)"
echo "🌐 To run from GHCR, use: docker-compose -f docker-compose-ghcr.yml up -d" 