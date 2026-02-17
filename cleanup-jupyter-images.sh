#!/bin/bash

# Altastata Python Package Docker Images Cleanup Script
# This script removes jupyter-datascience images (arm64, amd64, legacy) from local Docker repository

echo "🧹 Cleaning up jupyter-datascience Docker images..."

# List existing images before cleanup
echo "📋 Current jupyter-datascience images:"
docker images | grep jupyter-datascience || echo "No jupyter-datascience images found"

echo ""
echo "🗑️  Removing jupyter-datascience images..."

# Remove altastata/jupyter-datascience-* (arm64, amd64) and legacy altastata/jupyter-datascience
echo "Removing altastata/jupyter-datascience* images..."
docker rmi $(docker images --format "{{.Repository}}:{{.Tag}}" | grep "altastata/jupyter-datascience") 2>/dev/null || echo "  None found"

# Remove ghcr.io/sergevil/altastata/jupyter-datascience-* (arm64, amd64) and legacy
echo "Removing ghcr.io/sergevil/altastata/jupyter-datascience* images..."
docker rmi $(docker images --format "{{.Repository}}:{{.Tag}}" | grep "ghcr.io/sergevil/altastata/jupyter-datascience") 2>/dev/null || echo "  None found"

# Remove dangling images
echo "Removing dangling images..."
docker image prune -f

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "📋 Remaining jupyter-datascience images (if any):"
docker images | grep jupyter-datascience || echo "No jupyter-datascience images remaining"

echo ""
echo "🚀 You can now build fresh images with:"
echo "   ./build-all-images.sh   # Builds for your platform (arm64 or amd64)"