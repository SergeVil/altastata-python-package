#!/bin/bash

# Altastata Python Package GHCR Push Script
# This script pushes the Jupyter DataScience Docker image to GitHub Container Registry

export YOUR_PAT=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
echo $YOUR_PAT | docker login ghcr.io -u sergevil --password-stdin

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