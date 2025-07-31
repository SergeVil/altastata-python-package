#!/bin/bash

# AltaStata Python Package - Complete Build and Upload Script
# This script builds the Python package and Docker images, then uploads them

set -e  # Exit on any error

echo "🚀 Starting AltaStata Python Package build and upload process..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check if required tools are installed
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v twine &> /dev/null; then
    print_warning "twine not found, installing..."
    pip install --upgrade twine
fi

# Check for PyPI credentials
if [ ! -f ~/.pypirc ]; then
    print_warning "PyPI credentials not found (~/.pypirc)"
    print_warning "You may need to configure PyPI upload manually"
fi

# Check for GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
    print_warning "GITHUB_TOKEN environment variable not set"
    print_warning "Docker images will be built but not pushed to GHCR"
fi

# Step 1: Build Python Package
print_status "Step 1: Building Python package..."

# Clean previous builds
print_status "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/ 2>/dev/null || true

# Build the package
print_status "Building package with python -m build..."
python -m build

# Verify the built package
print_status "Verifying built package..."
twine check dist/*

print_success "Python package built successfully!"

# Step 2: Upload Python Package to PyPI
read -p "Do you want to upload the Python package to PyPI? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Step 2: Uploading Python package to PyPI..."
    twine upload dist/*
    print_success "Python package uploaded to PyPI!"
else
    print_warning "Skipping PyPI upload"
fi

# Step 3: Build Docker Images
print_status "Step 3: Building Docker images..."
./build-all-images.sh

# Step 4: Push Docker Images to GHCR
if [ -n "$GITHUB_TOKEN" ]; then
    read -p "Do you want to push Docker images to GHCR? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Step 4: Pushing Docker images to GHCR..."
        ./push-to-ghcr.sh
        print_success "Docker images pushed to GHCR!"
    else
        print_warning "Skipping GHCR push"
    fi
else
    print_warning "GITHUB_TOKEN not set, skipping GHCR push"
fi

print_success "Build and upload process completed!"
echo ""
echo "📦 Summary:"
echo "- Python package: Built and optionally uploaded to PyPI"
echo "- Docker images: Built for AMD64 and ARM64 architectures"
echo "- Docker images: Optionally pushed to GHCR"
echo ""
echo "🔧 Next steps:"
echo "- Test the uploaded package: pip install altastata"
echo "- Test Docker images: docker-compose up -d"
echo "- Update version in setup.py for next release" 