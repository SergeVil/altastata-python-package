#!/bin/bash
# Script to update version in all configuration files
# This script reads version.sh and updates:
# - .env (for docker-compose)
# - confidential-gke/jupyter-deployment.yaml (Kubernetes manifest)

# Load version configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/version.sh"

# Validate version is set
if [ -z "$VERSION" ]; then
    echo "Error: VERSION is not set in version.sh"
    echo "Please ensure version.sh contains: VERSION=\"your_version\""
    exit 1
fi

echo "Updating version to: ${VERSION}"

# Update .env file for docker-compose
echo "VERSION=${VERSION}" > .env
echo "✅ Updated .env file"

# Update Kubernetes deployment manifest
if [ -f "confidential-gke/jupyter-deployment.yaml" ]; then
    # Use sed to replace the version in the image line
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS requires -i '' for in-place editing
        sed -i '' "s|ghcr.io/sergevil/altastata/jupyter-datascience:[^[:space:]]*|ghcr.io/sergevil/altastata/jupyter-datascience:${VERSION}|g" confidential-gke/jupyter-deployment.yaml
    else
        # Linux
        sed -i "s|ghcr.io/sergevil/altastata/jupyter-datascience:[^[:space:]]*|ghcr.io/sergevil/altastata/jupyter-datascience:${VERSION}|g" confidential-gke/jupyter-deployment.yaml
    fi
    echo "✅ Updated confidential-gke/jupyter-deployment.yaml"
fi

echo ""
echo "Version update complete! All files now use version: ${VERSION}"

