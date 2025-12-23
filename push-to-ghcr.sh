#!/bin/bash

# Altastata Python Package GHCR Push Script
# This script builds and pushes multi-architecture Docker images (AMD64, ARM64, s390x) to GitHub Container Registry

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

# Create and use buildx builder instance if it doesn't exist
echo "🔧 Setting up Docker buildx for multi-architecture builds..."
docker buildx create --name altastata-builder --use 2>/dev/null || docker buildx use altastata-builder

echo "Building and pushing multi-architecture Docker images to GitHub Container Registry..."
echo "Architectures: AMD64, ARM64, s390x"

# Build and push multi-architecture images using buildx
echo "Building and pushing jupyter-datascience images..."
docker buildx build \
    --platform linux/amd64,linux/arm64,linux/s390x \
    --file openshift/Dockerfile.amd64 \
    --tag ghcr.io/sergevil/altastata/jupyter-datascience:latest \
    --tag ghcr.io/sergevil/altastata/jupyter-datascience:2025i_latest \
    --push \
    .

echo ""
echo "✅ Multi-architecture images (AMD64, ARM64, s390x) pushed successfully to GHCR!"
echo ""
echo "Images available at:"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience:latest (multi-arch: amd64, arm64, s390x)"
echo "- ghcr.io/sergevil/altastata/jupyter-datascience:2025i_latest (multi-arch: amd64, arm64, s390x)"
echo ""

echo ""
echo "To deploy from GHCR, run:"
echo "docker-compose -f docker-compose-ghcr.yml up -d"

echo ""
echo "⚠️  IMPORTANT: If this is a new image, you may need to make it public:"
echo "1. Go to: https://github.com/users/SergeVil/packages/container/package/altastata%2Fjupyter-datascience"
echo "2. Click 'Package settings' (gear icon)"
echo "3. Scroll to 'Danger Zone'"
echo "4. Click 'Change visibility'"
echo "5. Select 'Public'"
echo "6. Confirm the change"
echo ""
echo "Note: Private packages require authentication to pull, while public packages can be pulled by anyone." 