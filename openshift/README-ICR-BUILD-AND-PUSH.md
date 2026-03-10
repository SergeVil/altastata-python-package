# Build and Push Docker Images (ICR)

This guide covers building the IBM s390x image and the local arm64 test image,
then tagging and pushing the s390x image to IBM Container Registry (ICR).

## Prerequisites

- Docker Desktop installed and running
- Access token for ICR (provided by IBM/ICR admin)
- Access to the `altastata` ICR namespace

## Build s390x (IBM base)

On macOS, use buildx to produce a real s390x image:

```bash
docker buildx build --platform linux/s390x -f openshift/Dockerfile.s390x \
  -t altastata/jupyter-datascience-s390x:2026c_latest --load .
```

If you are already on an s390x host, a normal build is fine:

```bash
docker build -f openshift/Dockerfile.s390x -t altastata/jupyter-datascience-s390x:2026c_latest .
```

## Build arm64 (local testing on macOS)

```bash
docker build -f openshift/Dockerfile.arm64 .
```

Run locally:

```bash
docker tag <IMAGE_ID> altastata/jupyter-datascience-arm64:arm64-local
docker run --rm -p 8889:8888 altastata/jupyter-datascience-arm64:arm64-local
```

Then open `http://localhost:8889/?token=altastata-dev-token` (fixed token for local development).

## Tag s390x image for ICR

```bash
# Version from version.sh (e.g. 2026c_latest)
source version.sh 2>/dev/null || VERSION="2026c_latest"
docker tag altastata/jupyter-datascience-s390x:${VERSION} icr.io/altastata/jupyter-datascience-s390x:${VERSION}
```

## Login to ICR

We use IBM Container Registry (icr.io) only—no Docker Hub. Log in with your ICR API key (iamapikey), not a Docker ID:

```bash
export ICR_TOKEN="PASTE_NICOLAS_TOKEN_HERE"
echo "$ICR_TOKEN" | docker login -u iamapikey --password-stdin icr.io
```

## Push to ICR

```bash
docker push icr.io/altastata/jupyter-datascience-s390x:${VERSION}
```

Optional logout:

```bash
docker logout icr.io
```

---

## RAG Open LLM (s390x)

Same pattern for the RAG s390x image: build (on s390x host or with buildx), tag for ICR, push.

### Build RAG s390x image

**On an s390x host (recommended:** dependency layer is cached; use `./openshift/rag/build-rag-s390x-on-server.sh` from your Mac to sync repo and build on the server):

```bash
# On the s390x server (or from Mac via build-rag-s390x-on-server.sh)
source version.sh 2>/dev/null || true
docker build -f openshift/rag/Dockerfile.open_llm_s390x -t altastata/rag-open-llm-s390x:latest -t altastata/rag-open-llm-s390x:${VERSION:-2026c_latest} .
```

**On macOS with buildx** (cross-build; no cache from previous s390x builds):

```bash
source version.sh 2>/dev/null || true
docker buildx build --platform linux/s390x -f openshift/rag/Dockerfile.open_llm_s390x \
  -t altastata/rag-open-llm-s390x:latest -t altastata/rag-open-llm-s390x:${VERSION:-2026c_latest} --load .
```

### Tag and push RAG s390x to ICR

Uses the same **VERSION** as the Jupyter notebook image (from `version.sh`, currently **2026c_latest**).

**Push from server** (image was built with `build-rag-s390x-on-server.sh`):
```bash
export ICR_TOKEN="PASTE_NICOLAS_TOKEN_HERE"
./openshift/rag/push-rag-s390x-to-icr-from-server.sh
```

**Or manual (on the server):**
```bash
source version.sh 2>/dev/null || VERSION="2026c_latest"
docker tag altastata/rag-open-llm-s390x:latest icr.io/altastata/rag-open-llm-s390x:${VERSION}
echo "$ICR_TOKEN" | docker login -u iamapikey --password-stdin icr.io
docker push icr.io/altastata/rag-open-llm-s390x:${VERSION}
```

### Pull and run (for users who pull from ICR)

**From your Mac (script SSHs to server, pulls, runs, and tests):**
```bash
./openshift/rag/pull-and-run-rag-s390x-from-icr.sh
```
- Set **`ICR_TOKEN`** on your Mac so the script can log in to icr.io on the server before pull.
- The script **stops and removes** any existing `rag-s390x-test` container **before** pulling, then **leaves the new container running** when it finishes (no stop/rm at the end).
- **Default account:** `amazon.rsa.hpcs.serge678` (HPCS; no password). To use the password-based bob123 account:
  ```bash
  ACCOUNT_NAME=amazon.rsa.bob123 ./openshift/rag/pull-and-run-rag-s390x-from-icr.sh
  ```
- **Optional env (can set before running the script):**
  - `ACCOUNT_NAME` – AltaStata account subdir name (default: `amazon.rsa.hpcs.serge678`). Use `amazon.rsa.bob123` for password-based; the script passes `ALTASTATA_USE_HPCS=1` only when the name contains `hpcs`.
  - `SSH_HOST`, `SSH_KEY` – server and key path (see script defaults).
  - `HF_LLM_MODEL=gpt2` – for 8 GB VMs to avoid OOM (weaker answers).
- The account directory must exist on the **server** at `$REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME` (default `/root/.altastata/accounts/...`).

If you see **"manifest not found"**, the image is not in ICR yet—build on the server then push: `./openshift/rag/build-rag-s390x-on-server.sh` then `./openshift/rag/push-rag-s390x-to-icr-from-server.sh`.

**Or manually on the s390x server:**
```bash
source version.sh 2>/dev/null || VERSION="2026c_latest"
docker pull icr.io/altastata/rag-open-llm-s390x:${VERSION}
# Password-based account (e.g. bob123):
docker run -d -p 8000:8000 --name rag \
  -e ALTASTATA_ACCOUNT_DIR=/root/.altastata/accounts/amazon.rsa.bob123 \
  -e HF_LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  -v $HOME/.altastata/accounts:/root/.altastata/accounts:ro \
  icr.io/altastata/rag-open-llm-s390x:${VERSION}
# HPCS account (no password): add -e ALTASTATA_USE_HPCS=1
# Open http://<host>:8000/
```
