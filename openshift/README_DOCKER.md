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
# Kill the existing one
docker ps | grep 8888
docker stop <id>

# Build the image
docker build -t altastata:latest -f openshift/Dockerfile .

# Run the container
docker run --platform linux/amd64 -p 8888:8888 altastata:latest

# Access Jupyter Lab at http://127.0.0.1:8888/lab
```

## Detailed Instructions

### Building the Image

1. Navigate to the project root directory:
```bash
cd /path/to/altastata-python-package
```

2. Build the Docker image:
```bash
docker build -t altastata:latest -f openshift/Dockerfile .
```

### Running the Container

1. Run Jupyter Lab (interactive mode):
```bash
docker run --platform linux/amd64 -p 8888:8888 altastata:latest
```

2. Run Jupyter Lab in detached mode (background):
```bash
docker run -d --platform linux/amd64 -p 8888:8888 altastata:latest
```

3. Run with persistent storage for notebooks:
```bash
docker run --platform linux/amd64 -p 8888:8888 -v $(pwd)/notebooks:/home/jovyan/notebooks altastata:latest
```

### Container Management

#### Basic Operations
```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop container gracefully
docker stop <container_id>

# Force stop container
docker kill <container_id>

# Remove container
docker rm <container_id>

# Stop and remove in one command
docker rm -f <container_id>
```

#### Bulk Operations
```bash
# Stop all running containers
docker stop $(docker ps -q)

# Kill all running containers
docker kill $(docker ps -q)

# Remove all stopped containers
docker container prune
```

### Shell Access and Logs

#### Shell Access
```bash
# Get interactive shell in running container
docker exec -it <container_id> /bin/bash

# Start new container with shell
docker run -it --rm altastata:latest /bin/bash
```

#### Logs
```bash
# View container logs
docker logs <container_id>

# View real-time logs
docker logs -f <container_id>

# View last N lines
docker logs --tail 100 <container_id>
```

### Local Registry Setup

1. Start local registry (using port 5001 to avoid conflicts):
```bash
docker run -d -p 5001:5000 --restart=always --name registry registry:2
```

2. Tag image for local registry:
```bash
docker tag altastata:latest localhost:5001/altastata:latest
```

3. Push to local registry:
```bash
docker push localhost:5001/altastata:latest
```

## Verification and Troubleshooting

### Package Verification
```bash
# Check Java version
java -version

# Check Java home
echo $JAVA_HOME

# Check Python packages
pip list | grep altastata

# Check Python path
python -c "import altastata; print(altastata.__file__)"
```

### Common Issues
1. Port 8888 already in use:
```bash
# Find container using port 8888
docker ps | grep 8888

# Stop the container
docker stop <container_id>
```

2. Container won't start:
```bash
# Check container logs
docker logs <container_id>

# Verify image exists
docker images | grep altastata
```

## Notes and Considerations

- The container runs Jupyter Lab on port 8888
- Default user is jovyan (UID: 1001)
- Java 17 is installed and configured
- The Altastata package is installed in development mode
- All Python dependencies from requirements.txt are installed
- The container uses the linux/amd64 platform
- Jupyter authentication is disabled for easier access
- Use `--platform linux/amd64` flag when running on ARM-based systems (e.g., M1/M2 Macs) 