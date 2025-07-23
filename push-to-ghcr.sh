#!/bin/bash

# Altastata Python Package GHCR Push Script
# This script pushes the Jupyter DataScience Docker image to GitHub Container Registry

# Check if GitHub token is set
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable not set"
    echo "Please set your GitHub Personal Access Token:"
    echo "export GITHUB_TOKEN=your_token_here"
    echo "Or get it from: https://github.com/settings/tokens"
    exit 1
fi

echo $GITHUB_TOKEN | docker login ghcr.io -u sergevil --password-stdin

echo "Pushing Altastata Python Package Docker Image to GitHub Container Registry..."

# Push Jupyter DataScience image
echo "Pushing ghcr.io/sergevil/altastata/jupyter-datascience:2025a_latest..."
docker push ghcr.io/sergevil/altastata/jupyter-datascience:2025a_latest

echo ""
echo "Image pushed successfully to GHCR!"
echo ""
echo "Image available at:"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience:2025a_latest"
echo ""
echo "To deploy from GHCR, run:"
echo "docker-compose -f docker-compose-ghcr.yml up -d" 