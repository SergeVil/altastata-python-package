#!/bin/bash
# Script to update version in all configuration files
# This script reads version.sh and updates:
# - .env (for docker-compose: VERSION, ARCH)
# - confidential-gke/jupyter-deployment.yaml (Kubernetes manifest, uses jupyter-datascience-amd64)

# Load version configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/version.sh"

# Validate version is set
if [ -z "$VERSION" ]; then
    echo "Error: VERSION is not set in version.sh"
    echo "Please ensure version.sh contains: VERSION=\"your_version\""
    exit 1
fi

# Detect architecture for docker-compose
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    ARCH="arm64"
elif [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
    ARCH="amd64"
else
    ARCH="amd64"  # default
fi

echo "Updating version to: ${VERSION} (ARCH: ${ARCH})"

# Update .env file for docker-compose
echo "VERSION=${VERSION}" > .env
echo "ARCH=${ARCH}" >> .env
echo "✅ Updated .env file (VERSION=${VERSION}, ARCH=${ARCH})"

# Update Kubernetes deployment manifest (GKE runs AMD64)
if [ -f "confidential-gke/jupyter-deployment.yaml" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|ghcr.io/sergevil/altastata/jupyter-datascience[^:]*:[^[:space:]]*|ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}|g" confidential-gke/jupyter-deployment.yaml
    else
        sed -i "s|ghcr.io/sergevil/altastata/jupyter-datascience[^:]*:[^[:space:]]*|ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}|g" confidential-gke/jupyter-deployment.yaml
    fi
    echo "✅ Updated confidential-gke/jupyter-deployment.yaml (jupyter-datascience-amd64:${VERSION})"
fi

echo ""
echo "Version update complete! All files now use version: ${VERSION}"

