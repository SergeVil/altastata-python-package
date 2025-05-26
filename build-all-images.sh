#!/bin/bash

# Altastata Python Package Docker Images Build Script
# This script builds the Jupyter DataScience image with local altastata/ naming, then tags it for GHCR

echo "Building Altastata Python Package Docker Image..."

# Create Docker network if it doesn't exist (shared with main altastata project)
echo "Creating shared Docker network..."
docker network create altastata-network 2>/dev/null || echo "Network altastata-network already exists (shared with main project)"

# Build Jupyter DataScience image
echo "Building altastata/jupyter-datascience:latest..."
docker build -f openshift/Dockerfile -t altastata/jupyter-datascience:latest .

echo ""
echo "✅ Local image built successfully!"
echo ""

# Now tag it for GHCR
echo "Tagging image for GHCR..."
docker tag altastata/jupyter-datascience:latest ghcr.io/sergevil/altastata/jupyter-datascience:2025a_latest

echo ""
echo "✅ Image tagged for GHCR!"
echo ""
echo "Local image:"
echo "- altastata/jupyter-datascience:latest"
echo ""
echo "GHCR image:"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience:2025a_latest"
echo ""
echo "To push to GHCR, run: ./push-to-ghcr.sh"
echo "To run locally, use: docker-compose up -d (local image)"
echo "To run from GHCR, use: docker-compose -f docker-compose-ghcr.yml up -d" 