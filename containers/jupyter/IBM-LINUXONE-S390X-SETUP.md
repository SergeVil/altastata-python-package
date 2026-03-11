# IBM LinuxONE (s390x) Setup Guide

This guide explains how to set up an IBM LinuxONE (s390x) virtual machine on IBM Cloud and build the Jupyter DataScience Docker image for s390x architecture.

**To save money:** You do **not** need a Linux host (LinuxONE VM) for building or publishing the s390x image. Build on Mac with `docker buildx build --platform linux/s390x` and push to ICR (see [README-ICR-BUILD-AND-PUSH.md](README-ICR-BUILD-AND-PUSH.md)). Create a LinuxONE VM only when you need to run or test the image natively on s390x hardware. When not needed, stop or delete the VM in IBM Cloud to avoid ongoing cost.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Creating an IBM LinuxONE VM on IBM Cloud](#creating-an-ibm-linuxone-vm-on-ibm-cloud)
- [Configuring the VM](#configuring-the-vm)
- [Deploying S3 Gateway on s390x](#deploying-s3-gateway-on-s390x)
- [Filestash Installation (Building from Source)](#filestash-installation-building-from-source)
- [Building the Docker Image](#building-the-docker-image)
- [Running the Container](#running-the-container)
- [RAG Open LLM on s390x](#rag-open-llm-on-s390x)
- [Deploying S3 Gateway on s390x](#deploying-s3-gateway-on-s390x)
- [Filestash Installation (Building from Source)](#filestash-installation-building-from-source)
- [Architecture Support Notes](#architecture-support-notes)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- IBM Cloud account
- SSH key pair for accessing the VM
- Source code available locally or on the VM

## Creating an IBM LinuxONE VM on IBM Cloud

### Step 1: Access IBM Cloud Console

1. Log in to [IBM Cloud Console](https://cloud.ibm.com/)
2. Navigate to **Virtual Servers for VPC**
3. Select your region

### Step 2: Create Virtual Server

1. Click **Create** → **Virtual server for VPC**
2. Configure the instance:
   - **Name**: `altastata-linuxone` (or your preferred name)
   - **Resource group**: Select your resource group
   - **Location**: Select a region with LinuxONE availability (e.g., `us-south`)
   - **Architecture**: **IBM Z (s390x)** or **LinuxONE**

### Step 3: Image Selection

1. **Operating system**: Choose **Ubuntu 24.04 Minimal** (recommended)
   - Alternative: RHEL 9 if you have Red Hat subscriptions
2. **Profile**: Select a profile based on your needs:
   - **General purpose**: Good for development/testing
   - **Balanced**: Good balance of CPU and memory
   - **Memory optimized**: If you need more RAM
3. **Recommended minimums**:
   - **vCPUs**: 2-4
   - **RAM**: 8 GB (16 GB recommended for Docker builds)
   - **Boot volume**: 50 GB (larger if building many images)

### Step 4: Network Configuration

1. **VPC**: Select or create a VPC
2. **Subnet**: Select a subnet in your VPC
3. **Floating IP**: 
   - **Important**: Select **Create new floating IP** or attach an existing one
   - This is required for SSH access from outside the VPC

### Step 5: SSH Key

1. **SSH keys**: Upload or select your SSH public key
2. Save your private key securely (you'll need it to access the VM)

### Step 6: Review and Create

1. Review all settings
2. Click **Create virtual server instance**
3. Wait for the instance to be provisioned (2-5 minutes)

### Step 7: Note Important Information

After creation, note:
- **Floating IP address**: `xxx.xxx.xxx.xxx` (for SSH access)
- **Instance ID**: For reference
- **Cost**: LinuxONE instances are typically $5-10/month for basic configurations

## Configuring the VM

### Step 1: Connect via SSH

```bash
ssh -i /path/to/your/private-key.pem ubuntu@<FLOATING_IP>
```

Replace:
- `/path/to/your/private-key.pem` with your SSH private key path
- `<FLOATING_IP>` with the floating IP address from Step 7

### Step 2: Update System Packages

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 3: Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (to run docker without sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker installation
docker --version
docker info
```

### Step 4: Configure Docker Storage (if needed)

If you need more disk space for Docker images:

```bash
# Check disk space
df -h

# If using a separate disk (e.g., /dev/vdb)
# Format the disk (WARNING: This will erase all data)
sudo mkfs.ext4 /dev/vdb

# Mount the disk
sudo mkdir -p /mnt/docker-data
sudo mount /dev/vdb /mnt/docker-data

# Update /etc/fstab for permanent mounting
echo '/dev/vdb /mnt/docker-data ext4 defaults 0 2' | sudo tee -a /etc/fstab

# Configure Docker to use the new location
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "data-root": "/mnt/docker-data"
}
EOF

# Restart Docker
sudo systemctl restart docker
```

### Step 5: Verify Architecture

```bash
# Verify you're on s390x
uname -m
# Should output: s390x
```

## Building the Docker Image

### Step 1: Transfer Source Code to VM

**Option A: Using tar over SSH (recommended)**

From your local machine:

```bash
cd /path/to/altastata-python-package
tar czf - . | ssh -i /path/to/private-key.pem ubuntu@<FLOATING_IP> "cd ~ && tar xzf -"
```

**Option B: Using Git (if code is in a repository)**

On the VM:

```bash
git clone <your-repository-url>
cd altastata-python-package
```

**Option C: Using scp (for smaller directories)**

```bash
scp -i /path/to/private-key.pem -r altastata-python-package ubuntu@<FLOATING_IP>:~/
```

**Option D: Transfer a prebuilt image from macOS**

If you built the s390x image on macOS, you can export and copy it to the VM:

```bash
docker save altastata/jupyter-datascience-s390x:2026c -o /tmp/jupyter-2026c.tar
scp -i /path/to/private-key.pem /tmp/jupyter-2026c.tar ubuntu@<FLOATING_IP>:~/
```

On the VM, load the image:

```bash
sudo docker load -i /home/ubuntu/jupyter-2026c.tar
```

### Step 2: Navigate to Project Directory

```bash
cd ~/altastata-python-package
```

### Step 3: Verify Dockerfile Exists

```bash
ls -la containers/jupyter/Dockerfile.s390x
ls -la containers/jupyter/requirements-ibm.txt
```

### Step 4: Build the Docker Image

```bash
# Build the image (this will take 30-60+ minutes)
sudo docker build -f containers/jupyter/Dockerfile.s390x \
  -t altastata/jupyter-datascience-s390x:latest \
  . 2>&1 | tee /tmp/docker-build.log
```

**Note**: The build process will:
- Install Python 3.9, Java 17, and build dependencies
- Install Jupyter Lab and data science packages
- Install altastata package
- Take 30-60+ minutes due to compiling packages from source

**Important**: TensorFlow and PyTorch are not available from PyPI for s390x architecture. To use these frameworks, you need access to IBM's **AI Toolkit for IBM Z and LinuxONE** which provides optimized containers for TensorFlow and PyTorch. Contact IBM to request access to the AI Toolkit.

### Step 5: Monitor Build Progress

In another terminal (or using `screen`/`tmux`):

```bash
# Watch build progress
tail -f /tmp/docker-build.log

# Check current step
tail -20 /tmp/docker-build.log | grep -E '(Step|Installing|Complete)'
```

### Step 6: Verify Image Creation

```bash
# List images
sudo docker images | grep jupyter-datascience-s390x

# Check image details
sudo docker inspect altastata/jupyter-datascience-s390x:latest | grep -A 5 Architecture
```

## Running the Container

### Step 1: Run the Container

```bash
sudo docker run -d \
  --name altastata-jupyter-s390x \
  -p 8888:8888 \
  -v ~/jupyter-data:/home/jovyan/work \
  altastata/jupyter-datascience-s390x:latest
```

### Step 2: Check Container Status

```bash
# Check if container is running
sudo docker ps | grep jupyter

# View logs
sudo docker logs altastata-jupyter-s390x
```

### Step 3: Get Jupyter Token

The s390x image uses **token authentication** for production security. To get the token:

```bash
# Method 1: Using jupyter server list (recommended)
sudo docker exec -it altastata-jupyter-s390x jupyter server list

# Example output:
# Currently running servers:
# http://localhost:8888/?token=ua5nw5fwkdzseqpp5apj :: /home/jovyan

# Method 2: From container logs
sudo docker logs altastata-jupyter-s390x | grep token
```

The token is the string after `token=` (e.g., `ua5nw5fwkdzseqpp5apj`).

For more details, see [RunPod Token Authentication Guide](https://docs.runpod.io/references/troubleshooting/token-authentication-enabled).

### Step 4: Access Jupyter Lab

1. Get the Floating IP address of your VM
2. Open browser: `http://<FLOATING_IP>:8888`
3. Enter the token from Step 3 in the Token field
4. **Optional**: On first login, you can set a password for future access (see [Jupyter Security Documentation](https://jupyter-notebook.readthedocs.io/en/6.2.0/public_server.html#securing-a-notebook-server))

### Step 5: Verify Python Packages

In a Jupyter notebook, test imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from altastata import AltaStataFunctions

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Note: TensorFlow and PyTorch are not available on s390x from PyPI
# Contact IBM for AI Toolkit access if you need these frameworks
```

## RAG Open LLM on s390x

The **Open LLM RAG** app (simple vector store + Transformers + AltaStata) can be built and run on a LinuxONE/s390x VM.

### Build and run

**Option A – Pull from ICR and run:** If the image is published to ICR, from your Mac run:
```bash
./containers/rag-example/pull-and-run-rag-s390x-from-icr.sh
```
- SSHs to the server, **stops/removes** any existing `rag-s390x-test` container **before** pulling, pulls `icr.io/altastata/rag-open-llm-s390x:$VERSION`, runs the container, runs a test query, then **leaves the container running** (no stop/rm at the end).
- Set **`ICR_TOKEN`** on your Mac so the script can log in to icr.io on the server before pull.
- **Account:** Default is **HPCS** (`amazon.rsa.hpcs.serge678`; no password). For **bob123** (password-based):
  ```bash
  ACCOUNT_NAME=amazon.rsa.bob123 ./containers/rag-example/pull-and-run-rag-s390x-from-icr.sh
  ```
- Optional: `SSH_HOST`, `SSH_KEY`, `HF_LLM_MODEL=gpt2` (8 GB VMs). See [README-ICR-BUILD-AND-PUSH.md](README-ICR-BUILD-AND-PUSH.md) for full pull-and-run options.

Or manually on the server:
```bash
source version.sh 2>/dev/null || VERSION="2026c_latest"
docker pull icr.io/altastata/rag-open-llm-s390x:${VERSION}
# Then run with your account dir; see "Run the container" below.
```

**Option B – Build on the server from your Mac** (syncs repo, builds Docker image):
```bash
./containers/rag-example/build-rag-s390x-on-server.sh
```
Set `SSH_HOST` and `SSH_KEY` if needed (see script defaults). The script uses `GSSAPIAuthentication=no` and `PreferredAuthentications=publickey` to speed up SSH.

**Run the container** (after build or pull):
- After pushing to ICR, from your Mac run `./containers/rag-example/pull-and-run-rag-s390x-from-icr.sh` to pull and run on the server (or run `docker run` manually on the server).
- Manually on the server: use the same `docker run` as in [examples/rag-example/open_llm/README.md](../../examples/rag-example/open_llm/README.md) (env `ALTASTATA_ACCOUNT_DIR`, `HF_LLM_MODEL`, volume mount for `$HOME/.altastata/accounts`). For HPCS accounts add `-e ALTASTATA_USE_HPCS=1`. Open `http://<host>:8000/`.

On **8 GB VMs** use **gpt2** (`HF_LLM_MODEL=gpt2`); on 16+ GB use `HF_LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0`. See [examples/rag-example/open_llm/README.md](../../examples/rag-example/open_llm/README.md) for full s390x options (watsonx, llama.cpp, etc.).

**Check logs** (on the server):
```bash
ssh -i $SSH_KEY root@<FLOATING_IP> "docker logs rag-s390x-test 2>&1 | tail -80"
```

**Push RAG s390x to ICR:** See [README-ICR-BUILD-AND-PUSH.md](README-ICR-BUILD-AND-PUSH.md) (build, tag, login, push).

## Deploying S3 Gateway on s390x

### Step 1: Pull S3 Gateway Image

```bash
# Pull the s390x S3 Gateway image
docker pull ghcr.io/sergevil/altastata/s3-gateway-s390x:2026b_latest

# Tag as latest (optional)
docker tag ghcr.io/sergevil/altastata/s3-gateway-s390x:2026b_latest altastata/s3-gateway-s390x:latest
```

### Step 2: Run S3 Gateway Container

```bash
# Run the container
docker run -d \
  --name altastata-s3-gateway \
  -p 9876:9876 \
  -v ~/altastata-data:/app/data \
  ghcr.io/sergevil/altastata/s3-gateway-s390x:2026b_latest
```

### Step 3: Configure Security Group

**Important**: You must configure the IBM Cloud VPC Security Group to allow inbound traffic:

1. Go to IBM Cloud Console → **VPC Infrastructure** → **Security Groups**
2. Find the security group attached to your VM
3. Add an inbound rule:
   - **Type**: Custom TCP
   - **Port**: 9876
   - **Source**: 0.0.0.0/0 (or your specific IP for security)
   - **Action**: Allow

### Step 4: Verify S3 Gateway

```bash
# Check container status
docker ps | grep s3-gateway

# Check logs
docker logs altastata-s3-gateway

# Test status endpoint (on the VM)
curl http://localhost:9876/status

# Test from external machine (use your Floating IP)
curl http://169.63.190.94:9876/status
```

Expected response:
```json
{"status": "running", "message": "S3 Gateway is running. Set password via PUT /setPassword if needed."}
```

### Step 5: Configure User for Production Use

To use the S3 Gateway with real credentials (not test mode), you need to set up a user with properties, private key, and password:

**Option A: Using the setup script (Recommended)**

```bash
# On the IBM machine (ubuntu@169.63.190.94)
# First, transfer the setup script to the VM (or clone the repository)

# Source the setup script
source altastata-s3-gateway/scripts/realuser/setup-user-properties.sh

# Set up a user with properties, private key, and password
# Replace with your actual user properties and private key
setup_user_properties http://localhost:9876 your_username your_password

# Or use environment variables
GATEWAY_URL="http://localhost:9876" \
USER_NAME="your_username" \
PASSWORD="your_password" \
setup_user_properties
```

**Option B: Manual setup using curl**

```bash
# Step 1: Set user properties
curl -X PUT "http://localhost:9876/setUserProperties/your_username" \
  -H "Content-Type: text/plain" \
  -d "#your_account properties
myuser=youruser
accounttype=your-account-type
..."

# Step 2: Set private key
curl -X PUT "http://localhost:9876/setPrivateKey/your_username" \
  -H "Content-Type: text/plain" \
  -d "-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----"

# Step 3: Set password (this generates S3 credentials)
curl -X PUT "http://localhost:9876/setPassword/your_username?regenerateKeys=true" \
  -H "Content-Type: text/plain" \
  -d "your_password"
```

**Important Notes:**
- After any container restart, you must run `set_password` again to re-initialize the runtime services
- The setup script is located at `altastata-s3-gateway/scripts/realuser/setup-user-properties.sh`
- For production use, replace the example credentials with your actual user properties and private key
- The S3 Gateway will generate access keys and secret keys after setting the password

## Filestash Installation (Building from Source)

Filestash does not provide pre-built Docker images for s390x. To use Filestash, you must build it from source:

### Step 1: Install Build Dependencies

```bash
# Install required packages (including all C library dependencies)
sudo apt update
sudo apt install -y golang-go nodejs npm make \
  libbrotli-dev libsqlite3-dev gcc libc6-dev \
  libraw-dev libgif-dev libwebp-dev libjpeg-dev libpng-dev
```

### Step 2: Clone Filestash Repository

```bash
cd ~
git clone https://github.com/mickael-kerjean/filestash.git
cd filestash
```

### Step 3: Build Filestash

```bash
# Initialize dependencies
make init

# Build (this may take 10-30 minutes)
# Note: The build process may require installing additional dependencies as errors occur
make build
```

**Build Output**: The build creates a binary at `dist/filestash` (approximately 86MB).

### Step 4: Run Filestash

**Option A: Run as Docker Container (Recommended)**

```bash
# Build Docker image from the built binary
cd ~/filestash
docker build -f docker/Dockerfile -t filestash:latest .

# Run Filestash container connected to S3 Gateway
docker run -d \
  --name filestash-testing \
  --network altastata-network \
  -p 8334:8334 \
  -e S3_ENDPOINT_URL=http://altastata-s3-gateway:9876 \
  filestash:latest
```

**Option B: Run Binary Directly**

```bash
# Run Filestash binary directly
cd ~/filestash
./dist/filestash
```

**Note**: When running as a Docker container, use `http://altastata-s3-gateway:9876` as the S3 endpoint (Docker network name). When running the binary directly, use `http://localhost:9876` or `http://169.63.190.94:9876` (Floating IP).

### Step 5: Configure Filestash to Use S3 Gateway

**If running as Docker container:**
- Filestash will automatically use `http://altastata-s3-gateway:9876` (via Docker network)

**If running binary directly:**
- Configure Filestash to use `http://localhost:9876` or `http://169.63.190.94:9876`
- Set the `S3_ENDPOINT_URL` environment variable accordingly

### Step 6: Access Filestash

```bash
# From the IBM machine
curl http://localhost:8334

# From external machine (use your Floating IP)
# Note: You may need to configure security group to allow port 8334
curl http://169.63.190.94:8334
```

**Build Status**: ✅ Filestash has been successfully built from source on s390x. The build process requires several C library development packages (brotli, sqlite3, libraw, gif, webp, jpeg, png) which are installed as part of Step 1.

## Architecture Support Notes

### Supported Packages (s390x)

- ✅ **NumPy**: Builds from source (supported)
- ✅ **Pandas**: Builds from source (supported)
- ✅ **scikit-learn**: Builds from source (supported)
- ✅ **Jupyter Lab**: Works on s390x
- ✅ **Java 17**: Available in UBI9/RHEL repositories
- ✅ **Altastata**: Python package (works on any architecture)
- ✅ **S3 Gateway**: Full s390x Docker image available

### Requires IBM AI Toolkit Access (s390x)

- 🔒 **TensorFlow**: Not available from PyPI for s390x
  - **Solution**: Request access to IBM's AI Toolkit for IBM Z and LinuxONE
  - The AI Toolkit provides optimized TensorFlow containers for s390x
  - Contact IBM to request access
  
- 🔒 **PyTorch**: Not available from PyPI for s390x
  - **Solution**: Request access to IBM's AI Toolkit for IBM Z and LinuxONE
  - The AI Toolkit provides optimized PyTorch containers for s390x
  - Contact IBM to request access

- ✅ **Filestash**: Successfully builds from source on s390x
  - No pre-built Docker image available, but can be built from source
  - Requires: Go, Node.js, and C development libraries (brotli, sqlite3, libraw, gif, webp, jpeg, png)
  - Build time: 10-30 minutes

### IBM AI Toolkit for IBM Z and LinuxONE

For production ML workloads with TensorFlow and PyTorch, you need access to IBM's **AI Toolkit for IBM Z and LinuxONE**, which includes:
- Optimized TensorFlow containers for s390x
- Optimized PyTorch containers for s390x
- IBM-vetted, security-scanned containers
- Access through IBM Container Registry (icr.io)

**How to request access**: Contact IBM to request access to the AI Toolkit for IBM Z and LinuxONE. You will need:
- IBM Cloud account
- Access credentials for IBM Container Registry (icr.io)
- May require IBM support contracts depending on your use case

Once you have access, you can use the optimized containers from the AI Toolkit in your Docker images.

### Base Image

- **UBI9 Minimal (s390x)**: `registry.access.redhat.com/ubi9/ubi-minimal:latest`
  - Supports s390x architecture
  - Includes Python 3.9 by default
  - Lightweight base image

## Troubleshooting

### Issue: SSH Connection Timeout

**Problem**: Cannot connect to VM via SSH

**Solutions**:
1. Verify Floating IP is attached to the VM
2. Check Security Group rules allow SSH (port 22) and that this security group is **attached** to the instance's network interface
3. Verify VPC routing and subnet configuration; try from another network (e.g. VPN) if the VM is only reachable from certain IPs
4. Check VM is running (not stopped). If the VM was force-stopped, start it again and wait 2–3 minutes before retrying SSH
5. To speed up SSH once it works, use `-o GSSAPIAuthentication=no -o PreferredAuthentications=publickey` (the RAG build/test scripts do this automatically)

### Issue: Docker Build Fails with "No Space Left on Device"

**Problem**: Not enough disk space for Docker images

**Solutions**:
1. Check disk space: `df -h`
2. Clean up old Docker images: `sudo docker system prune -a`
3. Resize boot volume in IBM Cloud console
4. Use separate disk for Docker data (see Configuring the VM, Step 4)

### Issue: Docker Build Fails on Package Installation

**Problem**: Packages fail to install (network issues, compilation errors)

**Solutions**:
1. Check network connectivity: `ping 8.8.8.8`
2. Verify DNS resolution: `nslookup pypi.org`
3. Check build log: `tail -50 /tmp/docker-build.log`
4. Some packages may need additional build dependencies

### Issue: TensorFlow/PyTorch Installation Fails

**Problem**: TensorFlow or PyTorch installation errors (expected on s390x)

**Solution**: TensorFlow and PyTorch are not available from PyPI for s390x. To use these frameworks:
- **Request access to IBM's AI Toolkit for IBM Z and LinuxONE** (recommended)
- The AI Toolkit provides optimized containers for TensorFlow and PyTorch on s390x
- Once you have access, you can use the containers from IBM Container Registry (icr.io)

### Issue: Container Won't Start

**Problem**: Container exits immediately

**Solutions**:
1. Check logs: `sudo docker logs altastata-jupyter-s390x`
2. Verify port 8888 is available: `sudo netstat -tlnp | grep 8888`
3. Check container status: `sudo docker ps -a`
4. Try running interactively: `sudo docker run -it --rm altastata/jupyter-datascience-s390x:latest /bin/bash`

### Issue: Cannot Access Jupyter from Browser

**Problem**: Browser cannot connect to Jupyter

**Solutions**:
1. Verify Security Group allows inbound traffic on port 8888
2. Check Floating IP is correct
3. Verify container is running: `sudo docker ps`
4. Check Jupyter is listening: `sudo docker logs altastata-jupyter-s390x`
5. Try accessing from VM: `curl http://localhost:8888`

## Additional Resources

- [IBM Cloud Virtual Servers Documentation](https://cloud.ibm.com/docs/vpc)
- [Red Hat UBI Documentation](https://developers.redhat.com/products/rhel/ubi)
- [TensorFlow s390x Support](https://www.tensorflow.org/)
- [IBM LinuxONE Community](https://www.ibm.com/community/z/)

## Summary

This guide covers:
1. ✅ Creating an s390x LinuxONE VM on IBM Cloud
2. ✅ Configuring Docker on the VM
3. ✅ Building the Jupyter DataScience Docker image
4. ✅ Running and accessing the container
5. ✅ Understanding architecture support limitations

The resulting image includes:
- Python 3.9
- Java 17
- Jupyter Lab
- NumPy, Pandas, Matplotlib, scikit-learn
- Altastata package
- All other required dependencies

**TensorFlow and PyTorch require IBM AI Toolkit access** (not available from PyPI for s390x). To use these frameworks, request access to IBM's AI Toolkit for IBM Z and LinuxONE.

**S3 Gateway** is fully supported on s390x and can be deployed using the pre-built Docker image from GHCR.

