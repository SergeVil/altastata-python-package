#!/bin/bash

# Altastata Python Package Docker Images Cleanup Script
# This script removes existing jupyter-datascience images from local Docker repository

echo "🧹 Cleaning up existing jupyter-datascience Docker images..."

# List existing jupyter-datascience images before cleanup
echo "📋 Current jupyter-datascience images:"
docker images | grep jupyter-datascience || echo "No jupyter-datascience images found"

echo ""
echo "🗑️  Removing jupyter-datascience images..."

# Remove images with altastata/jupyter-datascience prefix
echo "Removing altastata/jupyter-datascience images..."
docker rmi $(docker images --format "{{.Repository}}:{{.Tag}}" | grep "altastata/jupyter-datascience") 2>/dev/null || echo "No altastata/jupyter-datascience images to remove"

# Remove images with ghcr.io/sergevil/altastata/jupyter-datascience prefix
echo "Removing ghcr.io/sergevil/altastata/jupyter-datascience images..."
docker rmi $(docker images --format "{{.Repository}}:{{.Tag}}" | grep "ghcr.io/sergevil/altastata/jupyter-datascience") 2>/dev/null || echo "No ghcr.io/sergevil/altastata/jupyter-datascience images to remove"

# Remove any dangling images that might be related
echo "Removing dangling images..."
docker image prune -f

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "📋 Remaining jupyter-datascience images (if any):"
docker images | grep jupyter-datascience || echo "No jupyter-datascience images remaining"

echo ""
echo "🚀 You can now build fresh images with:"
echo "./build-all-images.sh"