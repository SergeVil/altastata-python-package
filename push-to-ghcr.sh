#!/bin/bash

# Altastata Python Package GHCR Push Script
# This script pushes already-built local Docker images for amd64 and arm64 architectures to GitHub Container Registry

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

echo "Pushing Altastata Python Package Docker Images for separate architectures to GitHub Container Registry..."

# Push AMD64 images
echo "Pushing jupyter-datascience-amd64 images..."
docker push ghcr.io/sergevil/altastata/jupyter-datascience-amd64:latest
docker push ghcr.io/sergevil/altastata/jupyter-datascience-amd64:2025a_latest

# Push ARM64 images
echo "Pushing jupyter-datascience-arm64 images..."
docker push ghcr.io/sergevil/altastata/jupyter-datascience-arm64:latest
docker push ghcr.io/sergevil/altastata/jupyter-datascience-arm64:2025a_latest

# Create legacy tag by tagging the AMD64 image
echo "Creating legacy tag jupyter-datascience:2025a_latest linked to AMD64 image..."
docker tag ghcr.io/sergevil/altastata/jupyter-datascience-amd64:2025a_latest ghcr.io/sergevil/altastata/jupyter-datascience:2025a_latest

# Push the legacy tag
echo "Pushing legacy tag to GHCR..."
docker push ghcr.io/sergevil/altastata/jupyter-datascience:2025a_latest

echo ""
echo "✅ Separate architecture images pushed successfully to GHCR!"
echo ""
echo "Images available at:"
echo "AMD64 (x86_64):"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience-amd64:latest"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience-amd64:2025a_latest"
echo ""
echo "ARM64:"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience-arm64:latest"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience-arm64:2025a_latest"
echo ""
echo "Legacy compatibility:"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience:2025a_latest (linked to amd64)"
echo ""
echo "To deploy from GHCR, run:"
echo "docker-compose -f docker-compose-ghcr.yml up -d" 