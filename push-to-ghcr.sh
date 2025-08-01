#!/bin/bash

# Altastata Python Package GHCR Push Script
# This script pushes already-built local Docker image for amd64 architecture to GitHub Container Registry

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

echo "Pushing Altastata Python Package Docker Image to GitHub Container Registry..."

# Push AMD64 images
echo "Pushing jupyter-datascience images..."
docker push ghcr.io/sergevil/altastata/jupyter-datascience:latest
docker push ghcr.io/sergevil/altastata/jupyter-datascience:2025b_latest





echo ""
echo "✅ AMD64 image pushed successfully to GHCR!"
echo ""
echo "Images available at:"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience:latest"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience:2025b_latest"
echo ""

echo ""
echo "To deploy from GHCR, run:"
echo "docker-compose -f docker-compose-ghcr.yml up -d" 