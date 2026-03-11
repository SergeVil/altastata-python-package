# Altastata Python Package Docker Guide

This comprehensive guide covers building, running, and deploying the Altastata Python Package Jupyter DataScience environment using Docker containers, including local development and GitHub Container Registry (GHCR) deployment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Building Images](#building-images)
- [Local Development](#local-development)
- [GitHub Container Registry (GHCR)](#github-container-registry-ghcr)
- [Confidential Computing Deployment](#confidential-computing-deployment)
- [Volume Management](#volume-management)
- [Container Management](#container-management)
- [Troubleshooting](#troubleshooting)
- [Service URLs](#service-urls)

## Prerequisites

- Docker installed and running
- Docker Compose installed
- GitHub account with access to the repository (for GHCR)
- GitHub Personal Access Token with `write:packages` permission (for GHCR)

## Version Management

The Docker image version is centrally managed in `version.sh`. To update the version across all scripts and configuration files:

1. Edit `version.sh` and update the `VERSION` variable:
   ```bash
   VERSION="2026b_latest"  # Change to your desired version
   ```

2. Run the update script to sync the version to all configuration files:
   ```bash
   ./update-version.sh
   ```

The version will be automatically used by:
- Build scripts (`containers/jupyter/build-all-images.sh`, `containers/jupyter/push-to-ghcr.sh`)
- Docker Compose files (`containers/jupyter/docker-compose.yml`, `containers/jupyter/docker-compose-ghcr.yml`)
- Kubernetes manifests (`containers/confidential-gke/jupyter-deployment.yaml`)

**Note**: All scripts validate that `VERSION` is set and will exit with an error if it's missing.

## System Architecture

The Altastata Python Package system consists of a single Jupyter DataScience environment:

| **Service** | **Description** | **Port** | **Image** |
|-------------|-----------------|----------|-----------|
| **Jupyter DataScience** | Jupyter Lab with PyTorch, TensorFlow, and Altastata integration | 8888 | `altastata/jupyter-datascience` |

## Multi-Architecture Support

The project builds **architecture-specific Docker images** for AMD64 and ARM64.
Each architecture has its own GHCR package. The s390x (IBM Z/LinuxONE) image is built
separately using `containers/jupyter/Dockerfile.s390x`.

### Available Images

| **Architecture** | **Package** | **Image Tag** | **Size** | **Use Case** |
|------------------|-------------|---------------|----------|--------------|
| **ARM64** | `jupyter-datascience-arm64` | `${VERSION}` (from `version.sh`) | ~7GB | Apple Silicon, ARM servers |
| **AMD64** | `jupyter-datascience-amd64` | `${VERSION}` | ~7GB | Intel/AMD x86_64, GCP |
| **s390x (IBM Z/LinuxONE)** | `jupyter-datascience-s390x` | `${VERSION}` | ~4-5GB | IBM Z and LinuxONE |

### Dockerfile

- `containers/jupyter/Dockerfile.amd64` - AMD64 build
- `containers/jupyter/Dockerfile.arm64` - ARM64 build
- `containers/jupyter/Dockerfile.s390x` - IBM Z/LinuxONE build

### Build Scripts

- `containers/jupyter/build-all-images.sh` - Build local AMD64 image
- `containers/jupyter/push-to-ghcr.sh` - Build and push ARM64 + AMD64 images to GHCR
- `containers/jupyter/cleanup-jupyter-images.sh` - Clean up Docker images

### Usage (AMD64 + ARM64)

```bash
# Pull image for your platform (version from version.sh, currently: 2026b_latest)
source version.sh

# Apple Silicon (ARM64):
docker pull ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION}
docker run -p 8888:8888 ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION}

# Intel/AMD (AMD64):
docker pull ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}
docker run -p 8888:8888 ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}
```

### Platform Compatibility

- **Apple Silicon Macs**: Use `jupyter-datascience-arm64`
- **Intel Macs**: Use `jupyter-datascience-amd64`
- **GCP Confidential GKE**: Use `jupyter-datascience-amd64`
- **IBM Z and LinuxONE**: Use `jupyter-datascience-s390x`



## Quick Start

### Option 1: Build and Run Everything

```bash
# 1. Build the image
./containers/jupyter/build-all-images.sh

# 2. Run locally (from repo root)
docker compose -f containers/jupyter/docker-compose.yml up -d

# 3. Access Jupyter Lab at http://localhost:8888/lab
# Get the token from logs: docker logs altastata-jupyter 2>&1 | grep -E "127.0.0.1:8888|token"
# Or: docker exec altastata-jupyter jupyter server list
```

### Option 2: Use Pre-built GHCR Images

```bash
# Run from GitHub Container Registry (from repo root)
docker compose -f containers/jupyter/docker-compose-ghcr.yml up -d
```

Then open **http://localhost:8888/lab**. Get the generated token from the logs: `docker logs altastata-jupyter 2>&1 | grep -E "127.0.0.1:8888|token"` or `docker exec altastata-jupyter jupyter server list`.

## Building Images

### Build and Push to GHCR

```bash
# Build and push architecture-specific images to GHCR
./containers/jupyter/push-to-ghcr.sh
```

This script will:
1. Build ARM64 image using `containers/jupyter/Dockerfile.arm64` and push to `jupyter-datascience-arm64`
2. Build AMD64 image using `containers/jupyter/Dockerfile.amd64` and push to `jupyter-datascience-amd64`

### Local Build

```bash
# Build ARM64 image locally (Apple Silicon):
docker build -f containers/jupyter/Dockerfile.arm64 -t altastata/jupyter-datascience-arm64:latest .

# Build AMD64 image locally:
docker buildx build --platform linux/amd64 -f containers/jupyter/Dockerfile.amd64 -t altastata/jupyter-datascience-amd64:latest --load .
```

### Manual Build and Push

```bash
# Create network first
docker network create altastata-network

# Build and push to GHCR (use ./containers/jupyter/push-to-ghcr.sh for automated push)
source version.sh
./containers/jupyter/push-to-ghcr.sh
```

## Local Development

### Using Docker Compose

#### Full System

```bash
# Start Jupyter service (from repo root)
docker compose -f containers/jupyter/docker-compose.yml up -d

# View logs
docker compose -f containers/jupyter/docker-compose.yml logs -f

# Stop service
docker compose -f containers/jupyter/docker-compose.yml down
```

### Individual Container

```bash
# Create network
docker network create altastata-network

# Jupyter DataScience (use image matching your platform: arm64 or amd64)
docker run -d \
  --name altastata-jupyter \
  -p 8888:8888 \
  --network altastata-network \
  -v $(pwd)/examples/pytorch-example:/home/jovyan/pytorch-example \
  -v $(pwd)/examples/tensorflow-example:/home/jovyan/tensorflow-example \
  -v $(pwd)/altastata:/home/jovyan/altastata-source \
  -e JUPYTER_ENABLE_LAB=yes \
  ghcr.io/sergevil/altastata/jupyter-datascience-arm64:latest   # or jupyter-datascience-amd64 on Intel/AMD
```

### Development Workflow

```bash
# 1. Make code changes to Python package...

# 2. Rebuild image
docker compose -f containers/jupyter/docker-compose.yml build altastata-jupyter

# 3. Restart service
docker compose -f containers/jupyter/docker-compose.yml up -d altastata-jupyter

# 4. View logs
docker compose -f containers/jupyter/docker-compose.yml logs -f altastata-jupyter
```

## GitHub Container Registry (GHCR)

### Authentication

```bash
# Set your GitHub Personal Access Token
export YOUR_PAT=your_github_token_here          # see containers/jupyter/push-to-ghcr.sh

# Login to GitHub Container Registry
echo $YOUR_PAT | docker login ghcr.io -u your_username --password-stdin
```

### Pushing Images

#### Automated Push

```bash
# Build and push architecture-specific images to GHCR
./containers/jupyter/push-to-ghcr.sh
```

This builds and pushes both `jupyter-datascience-arm64` and `jupyter-datascience-amd64` with the version from `version.sh`.

#### Manual Push

```bash
# Login first
export GITHUB_TOKEN=your_github_token_here
./containers/jupyter/push-to-ghcr.sh




```

### Pulling Images

```bash
# Pull image for your platform (version from version.sh)
source version.sh

# Apple Silicon (ARM64):
docker pull ghcr.io/sergevil/altastata/jupyter-datascience-arm64:${VERSION}

# Intel/AMD (AMD64):
docker pull ghcr.io/sergevil/altastata/jupyter-datascience-amd64:${VERSION}




```

### Deploying from GHCR

```bash
# Deploy using GHCR image
docker compose -f containers/jupyter/docker-compose-ghcr.yml up -d

# Update deployment with latest image
docker compose -f containers/jupyter/docker-compose-ghcr.yml pull
docker compose -f containers/jupyter/docker-compose-ghcr.yml up -d
```

### Making Packages Public

After pushing images to GHCR, you may need to make them public for external access:

1. **ARM64**: https://github.com/users/SergeVil/packages/container/package/altastata%2Fjupyter-datascience-arm64
2. **AMD64**: https://github.com/users/SergeVil/packages/container/package/altastata%2Fjupyter-datascience-amd64
3. **Click "Package settings"** (gear icon) for each
4. **Scroll to "Danger Zone"** → **"Change visibility"** → **"Public"**
5. **Confirm the change**

**Note**: Private packages require authentication to pull, while public packages can be pulled by anyone.

## Confidential Computing Deployment

Deploy Altastata in a secure, confidential computing environment on Google Cloud Platform with hardware-level memory encryption:

### Quick Start

```bash
# Navigate to confidential GKE setup
cd containers/confidential-gke

# Deploy confidential cluster with AMD SEV security
./setup-cluster.sh

# Access Jupyter Lab at the provided URL
# Stop cluster when not in use (saves costs)
gcloud container clusters delete altastata-confidential-cluster --zone=us-central1-a
```

### Features

- **Hardware-level security** with AMD SEV encryption
- **Memory encryption** during data processing
- **Multi-cloud storage** support (GCP, AWS, Azure)
- **Cost optimization** with easy stop/start commands
- **AMD64 architecture** support for optimal performance

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Confidential GKE Node                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Jupyter Container                      │   │
│  │  ┌─────────────────┐  ┌─────────────────────────┐  │   │
│  │  │   Jupyter Lab   │  │   Altastata Package     │  │   │
│  │  │   (Port 8888)   │  │   - Storage Management  │  │   │
│  │  │                 │  │   - Data Processing     │  │   │
│  │  └─────────────────┘  │   - Security            │  │   │
│  │                       └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Cloud Storage   │
                    │ - GCP Storage   │
                    │ - AWS S3        │
                    │ - Azure Blob    │
                    └─────────────────┘
```

### Management Commands

| Action | Command |
|--------|---------|
| **Deploy Cluster** | `./setup-cluster.sh` |
| **Stop Cluster** | `gcloud container clusters delete altastata-confidential-cluster --zone=us-central1-a` |
| **Stop Container** | `kubectl scale deployment altastata-jupyter-confidential --replicas=0` |
| **Start Container** | `kubectl scale deployment altastata-jupyter-confidential --replicas=1` |
| **Check Status** | `kubectl get pods -l app=altastata-jupyter` |

See `containers/confidential-gke/README.md` for detailed setup instructions.

## Volume Management

### Volume Mounts

The Jupyter container uses several volume mounts for development:

```bash
# Volume mounts
-v ./examples/pytorch-example:/home/jovyan/pytorch-example      # PyTorch examples
-v ./examples/tensorflow-example:/home/jovyan/tensorflow-example # TensorFlow examples
-v ./altastata:/home/jovyan/altastata-source           # Source code
-v jupyter-data:/home/jovyan/work                      # Persistent workspace
```

### Volume Commands

```bash
# List volumes
docker volume ls

# Remove unused volumes
docker volume prune

# Inspect volume
docker volume inspect jupyter-data
```

## Container Management

### Useful Commands

```bash
# List running containers
docker ps

# Stop container
docker stop altastata-jupyter

# Remove container
docker rm altastata-jupyter

# View container logs
docker logs altastata-jupyter

# Follow logs in real-time
docker logs -f altastata-jupyter

# Execute command in running container
docker exec -it altastata-jupyter /bin/bash

# Access Jupyter terminal
docker exec -it altastata-jupyter jupyter lab list
```

### Image Management

```bash
# List images
docker images

# Remove image
docker rmi altastata/jupyter-datascience-arm64:latest   # or jupyter-datascience-amd64

# Remove unused images
docker image prune
```

## Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check what's using port 8888
lsof -i :8888

# Kill processes using port
sudo kill -9 $(lsof -t -i:8888)
```

#### 2. Jupyter Access Issues
```bash
# Check if Jupyter is running
docker exec altastata-jupyter jupyter lab list

# Get Jupyter URL with token
docker logs altastata-jupyter | grep "http://127.0.0.1:8888"
```

#### 3. Package Import Issues
```bash
# Check if altastata package is installed
docker exec altastata-jupyter python -c "import altastata; print('Package imported successfully')"

# Check Python path
docker exec altastata-jupyter python -c "import sys; print('\n'.join(sys.path))"
```

#### 4. Java/JAR Issues
```bash
# Check Java installation
docker exec altastata-jupyter java -version

# Check JAR files (altastata installed via pip)
docker exec altastata-jupyter ls -la $(docker exec altastata-jupyter python -c "import altastata; import os; print(os.path.dirname(altastata.__file__))")/lib/
```

#### 5. Build Failures
```bash
# Build with verbose output (use image matching your platform)
docker build --no-cache --progress=plain -f containers/jupyter/Dockerfile.arm64 -t altastata/jupyter-datascience-arm64:latest .   # ARM64
docker buildx build --no-cache --progress=plain --platform linux/amd64 -f containers/jupyter/Dockerfile.amd64 -t altastata/jupyter-datascience-amd64:latest --load .   # AMD64

# Check build context
ls -la .
```

### Health Checks

```bash
# Check if Jupyter is responding
curl http://localhost:8888/lab

# Check container health
docker inspect altastata-jupyter | grep -A 10 "Health"
```

### Reset Everything

```bash
# Stop container
docker stop altastata-jupyter

# Remove container
docker rm altastata-jupyter

# Remove image
docker rmi altastata/jupyter-datascience-arm64:latest   # or jupyter-datascience-amd64

# Remove volumes
docker volume rm jupyter-data

# Remove network
docker network rm altastata-network
```

## Service URLs

When deployed, the service is available at:

- **Jupyter Lab**: http://localhost:8888 (Main development environment)

## Complete Workflows

### Development Workflow

```bash
# 1. Build image
./containers/jupyter/build-all-images.sh

# 2. Start development environment
docker compose -f containers/jupyter/docker-compose.yml up -d

# 3. Access Jupyter Lab at http://localhost:8888

# 4. Make changes and rebuild if needed
docker compose -f containers/jupyter/docker-compose.yml build altastata-jupyter
docker compose -f containers/jupyter/docker-compose.yml up -d altastata-jupyter
```

### Production Deployment Workflow

```bash
# 1. Build and tag image
./containers/jupyter/build-all-images.sh

# 2. Push to GHCR
./containers/jupyter/push-to-ghcr.sh

# 3. Deploy from GHCR
docker compose -f containers/jupyter/docker-compose-ghcr.yml up -d

# 4. Monitor deployment
docker compose -f containers/jupyter/docker-compose-ghcr.yml logs -f
```

### Package Development Workflow

```bash
# 1. Start development environment
docker compose -f containers/jupyter/docker-compose.yml up -d

# 2. Access Jupyter Lab
open http://localhost:8888

# 3. Edit Python files in mounted directories:
#    - ./altastata/ (source code)
#    - ./examples/pytorch-example/ (PyTorch examples)
#    - ./examples/tensorflow-example/ (TensorFlow examples)

# 4. Test changes in Jupyter notebooks

# 5. Rebuild package if needed
docker exec altastata-jupyter pip install -e /home/jovyan/altastata-source
```

## Integration Scenarios

### Scenario 1: Standalone Python Package Development

**Use Case**: Developing and testing the Altastata Python package in isolation.

```bash
# Navigate to Python package project
cd /Users/sergevilvovsky/eclipse-workspace/mcloud/altastata-python-package

# Build and start Jupyter environment
./containers/jupyter/build-all-images.sh
docker compose -f containers/jupyter/docker-compose.yml up -d

# Access Jupyter Lab
open http://localhost:8888

# Available services:
# - Jupyter Lab: http://localhost:8888
# - Network: altastata-network (created)
```

**Benefits:**
- Fast startup (single container)
- Focused development environment
- No port conflicts with main project

### Scenario 2: Full Stack Development (Both Projects)

**Use Case**: Developing machine learning applications that need to interact with the main Altastata APIs.

```bash
# Terminal 1: Start main Altastata project
cd /Users/sergevilvovsky/eclipse-workspace/mcloud/mycloud
./containers/jupyter/build-all-images.sh
docker compose -f containers/jupyter/docker-compose.yml up -d

# Terminal 2: Start Python package
cd /Users/sergevilvovsky/eclipse-workspace/mcloud/altastata-python-package
./containers/jupyter/build-all-images.sh
docker compose -f containers/jupyter/docker-compose.yml up -d

# Access all services:
# - Web UI: http://localhost:3000
# - Admin UI: http://localhost:3500
# - Jupyter Lab: http://localhost:8888
# - Web API: http://localhost:8080
# - Admin API: http://localhost:8084
```

**Cross-Service Communication:**
```python
# In Jupyter notebook, test API connectivity
import requests

# Test Web API health
web_api_health = requests.get('http://altastata-web-api:8080/health')
print(f"Web API Status: {web_api_health.status_code}")

# Test Admin API health  
admin_api_health = requests.get('http://altastata-admin-api:8084/health')
print(f"Admin API Status: {admin_api_health.status_code}")

# Use Altastata functions with live backend
from altastata import AltaStataFunctions
# Configure to use running admin API for certificate management
```

### Scenario 3: Machine Learning Pipeline with Live Data

**Use Case**: Training ML models using data from the running Altastata system.

```bash
# 1. Start both projects (as in Scenario 2)

# 2. Initialize organization and users via Admin UI
open http://localhost:3500

# 3. Upload training data via Web UI
open http://localhost:3000

# 4. Develop ML pipeline in Jupyter
open http://localhost:8888
```

**Jupyter Notebook Example:**
```python
# Connect to live Altastata system
from altastata import AltaStataFunctions, AltaStataPyTorchDataset

# Use credentials from running admin system
altastata_functions = AltaStataFunctions.from_credentials(
    user_properties, private_key
)

# Create dataset pointing to live data
dataset = AltaStataPyTorchDataset(
    "myorg_user",
    root_dir="s3://my-bucket/training-data/",
    file_pattern="*.jpg",
    transform=transforms
)

# Train model with live data
for batch in DataLoader(dataset):
    # Training loop...
```

### Scenario 4: Production ML Inference

**Use Case**: Running inference services alongside the main Altastata system.

```bash
# Deploy both projects from GHCR
cd /Users/sergevilvovsky/eclipse-workspace/mcloud/mycloud
docker compose -f containers/jupyter/docker-compose-ghcr.yml up -d

cd /Users/sergevilvovsky/eclipse-workspace/mcloud/altastata-python-package
docker compose -f containers/jupyter/docker-compose-ghcr.yml up -d

# Configure Jupyter for inference workloads
docker exec altastata-jupyter pip install fastapi uvicorn

# Deploy inference API from Jupyter
```

### Scenario 5: Development with Hot Reloading

**Use Case**: Rapid development with automatic code reloading.

```bash
# Start with volume mounts for live editing
docker compose -f containers/jupyter/docker-compose.yml up -d

# Edit Python files directly on host:
# - ./altastata/ → /home/jovyan/altastata-source
# - ./examples/pytorch-example/ → /home/jovyan/pytorch-example
# - ./examples/tensorflow-example/ → /home/jovyan/tensorflow-example

# Reload package in Jupyter without restart
docker exec altastata-jupyter pip install -e /home/jovyan/altastata-source

# Test changes immediately in notebooks
```

### Scenario 6: Multi-User Development

**Use Case**: Multiple developers working on different aspects.

```bash
# Developer 1: Backend APIs
cd /Users/sergevilvovsky/eclipse-workspace/mcloud/mycloud
docker-compose up -d altastata-web-api altastata-admin-api

# Developer 2: Frontend UIs  
docker-compose up -d altastata-web-ui altastata-admin-ui

# Developer 3: ML/Python package
cd /Users/sergevilvovsky/eclipse-workspace/mcloud/altastata-python-package
docker compose -f containers/jupyter/docker-compose.yml up -d

# All share the same altastata-network for communication
```

## Network Connectivity Testing

### Verify Cross-Project Communication

```bash
# From Jupyter container, test all main project services
docker exec altastata-jupyter ping altastata-web-api
docker exec altastata-jupyter ping altastata-admin-api
docker exec altastata-jupyter ping altastata-web-ui
docker exec altastata-jupyter ping altastata-admin-ui

# Test HTTP connectivity
docker exec altastata-jupyter curl -s http://altastata-web-api:8080/health
docker exec altastata-jupyter curl -s http://altastata-admin-api:8084/health

# Test from main project to Jupyter (if needed)
docker exec altastata-web-api ping altastata-jupyter
```

### Network Inspection

```bash
# View all containers on the shared network
docker network inspect altastata-network

# See which containers are connected
docker network inspect altastata-network --format='{{range .Containers}}{{.Name}} {{.IPv4Address}}{{"\n"}}{{end}}'

# Monitor network traffic (if needed)
docker exec altastata-jupyter netstat -tuln
```

## Network Integration

### Shared Network with Main Altastata Project

Both the main Altastata project and this Python package use the same Docker network (`altastata-network`). This allows:

- **Cross-project communication**: Jupyter notebooks can potentially communicate with the main Altastata APIs
- **Shared network resources**: Both projects use the same network infrastructure
- **Simplified deployment**: One network for all Altastata services

If you have both projects running simultaneously:
```bash
# Main project services will be available from Jupyter as:
# - altastata-web-api:8080
# - altastata-admin-api:8084
# - altastata-web-ui:28081
# - altastata-admin-ui:3500

# Test connectivity from Jupyter container
docker exec altastata-jupyter ping altastata-web-api
docker exec altastata-jupyter curl -s http://altastata-admin-api:8084/health
```

## Jupyter Token Authentication

The Jupyter container does **not** use a hardcoded token; a token is **generated at startup** (all images: arm64, amd64, s390x). If you still see an old fixed token, rebuild the image and recreate the container. Use one of these to get the current token:

1. **From container logs** (look for a line with `http://127.0.0.1:8888` and `?token=...`):
   ```bash
   docker logs altastata-jupyter 2>&1 | grep -E "127.0.0.1:8888|token"
   ```

2. **From inside the container**:
   ```bash
   docker exec altastata-jupyter jupyter server list
   ```
   Example output: `http://localhost:8888/?token=abc123... :: /home/jovyan`

3. **Copy the token** (the string after `?token=`) and either open that full URL in the browser or go to **http://localhost:8888/lab** and paste the token when prompted.

4. **Optional**: On first login, you can set a password for future access.

For more details, see:
- [RunPod Token Authentication Guide](https://docs.runpod.io/references/troubleshooting/token-authentication-enabled)
- [Jupyter Notebook Security Documentation](https://jupyter-notebook.readthedocs.io/en/6.2.0/public_server.html#securing-a-notebook-server)

## Notes

- The container runs as user `1001` for OpenShift compatibility
- Java 17 is installed for Py4J integration with JAR files
- PyTorch CPU-only version is installed to reduce image size
- Jupyter token is auto-generated at container startup (see [Jupyter Token Authentication](#jupyter-token-authentication))
- All examples and source code are mounted as volumes for live editing
- The package includes large JAR files (83MB altastata-hadoop-all.jar)
- Symbolic links provide easy access to examples from Jupyter interface
- **Shares `altastata-network` with the main Altastata project for cross-service communication** 