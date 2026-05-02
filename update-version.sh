#!/bin/bash
# Script to update version in all configuration files
# This script reads version.sh and updates:
# - .env (for docker-compose: VERSION, ARCH)
# - containers/confidential-gke/jupyter-deployment.yaml (Kubernetes manifest, uses jupyter-datascience-amd64)

# Load version configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/version.sh"

if [ -z "$JUPYTER_VERSION" ]; then
    echo "Error: JUPYTER_VERSION not set in version.sh"
    exit 1
fi
# This script writes .env (docker-compose) and the GKE deployment manifest, both
# of which deploy the Jupyter image (jupyter-datascience-amd64). RAG has no
# docker-compose / k8s deployment yet, so we only forward JUPYTER_VERSION here.
VERSION="$JUPYTER_VERSION"

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
if [ -f "containers/confidential-gke/jupyter-deployment.yaml" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|ghcr.io/sergevil/altastata/jupyter-datascience[^:]*:[^[:space:]]*|ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}|g" containers/confidential-gke/jupyter-deployment.yaml
    else
        sed -i "s|ghcr.io/sergevil/altastata/jupyter-datascience[^:]*:[^[:space:]]*|ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}|g" containers/confidential-gke/jupyter-deployment.yaml
    fi
    echo "✅ Updated containers/confidential-gke/jupyter-deployment.yaml (jupyter-datascience-amd64:${VERSION})"
fi

echo ""
echo "Version update complete! All files now use version: ${VERSION}"

