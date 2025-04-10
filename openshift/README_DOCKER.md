# Docker Instructions for Altastata Workbench

This document provides instructions for building and running the Altastata workbench Docker container.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Instructions](#detailed-instructions)
  - [Building the Image](#building-the-image)
  - [Running the Container](#running-the-container)
  - [Container Management](#container-management)
  - [Shell Access and Logs](#shell-access-and-logs)
  - [Local Registry Setup](#local-registry-setup)
- [Verification and Troubleshooting](#verification-and-troubleshooting)
- [Notes and Considerations](#notes-and-considerations)

## Directory Structure

```
altastata-python-package/
├── altastata/              # Package source code
│   ├── altastata_functions.py
│   ├── base_gateway.py
│   ├── __init__.py
│   └── lib/                # Java libraries
├── openshift/              # OpenShift/Docker related files
│   ├── Dockerfile          # Container build instructions
│   ├── requirements.txt    # Python package dependencies
│   ├── .dockerignore       # Files to exclude from Docker build
│   └── README_DOCKER.md    # This file
├── setup.py                # Package installation configuration
├── test_script.py          # Test scripts
└── README.md               # Main project documentation
```

## Prerequisites

- Docker installed on your system
- Git repository cloned
- Python package dependencies in `openshift/requirements.txt`

## Quick Start

```bash

# Build the image
docker build -t altastata:latest -f openshift/Dockerfile .

# Kill the existing one
docker rm -f altastata-jupyter

# Run the container with .altastata volume mount
docker run --platform linux/amd64 \
  --name altastata-jupyter \
  -p 8888:8888 \
  -v /Users/sergevilvovsky/.altastata:/opt/app-root/src/.altastata:rw \
  -v /Users/sergevilvovsky/Desktop:/opt/app-root/src/Desktop:rw \
  altastata:latest

# Access Jupyter Lab at http://127.0.0.1:8888/lab

# In your Python code, use the container path:
from altastata import AltaStataFunctions
altastata_functions = AltaStataFunctions('/opt/app-root/src/.altastata/accounts/amazon.pqc.alice786')
altastata_functions.set_password("123")

# Store
result = altastata_functions.store(['/opt/app-root/src/Desktop/serge.png',
                                    '/opt/app-root/src/Desktop/meeting_saved_chat.txt'],
                                   '/opt/app-root/src/Desktop', 'StoreTest', True)
```