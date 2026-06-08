# Build and Push Docker Images (ICR)

This guide covers building the IBM s390x image and the local arm64 test image,
then tagging and pushing the s390x image to IBM Container Registry (ICR).

## Prerequisites

- Docker Desktop installed and running
- Access token for ICR (provided by IBM/ICR admin)
- Access to the `altastata` ICR namespace

## Versioning at a glance

`version.sh` exposes two image tags (different release cadences):

- `JUPYTER_VERSION` (currently `2026e_latest`) — used for `jupyter-datascience-{arm64,amd64,s390x}`.
- `RAG_VERSION` (currently `2026j_latest`) — used for `rag-open-llm-s390x`. The `:${RAG_VERSION}_zdnn` variant is the research/zDNN-on build; the plain tag is the default CPU build.

Source `version.sh` once and the rest of these examples just use `${JUPYTER_VERSION}` / `${RAG_VERSION}`.

## Build s390x (IBM base)

On macOS, use buildx to produce a real s390x image:

```bash
source version.sh
docker buildx build --platform linux/s390x -f containers/jupyter/Dockerfile.s390x \
  -t altastata/jupyter-datascience-s390x:${JUPYTER_VERSION} --load .
```

If you are already on an s390x host, a normal build is fine:

```bash
source version.sh
docker build -f containers/jupyter/Dockerfile.s390x -t altastata/jupyter-datascience-s390x:${JUPYTER_VERSION} .
```

## Build arm64 (local testing on macOS)

```bash
docker build -f containers/jupyter/Dockerfile.arm64 .
```

Run locally:

```bash
docker tag <IMAGE_ID> altastata/jupyter-datascience-arm64:arm64-local
docker run --rm -p 8889:8888 altastata/jupyter-datascience-arm64:arm64-local
```

Then open http://localhost:8889/lab and use the token from logs: `docker logs <container> 2>&1 | grep -E "127.0.0.1:8889|token"` or `docker exec <container> jupyter server list`.

## Tag s390x image for ICR

```bash
source version.sh
docker tag altastata/jupyter-datascience-s390x:${JUPYTER_VERSION} icr.io/altastata/jupyter-datascience-s390x:${JUPYTER_VERSION}
```

## Login to ICR

We use IBM Container Registry (icr.io) only—no Docker Hub. Log in with your ICR API key (iamapikey), not a Docker ID:

```bash
export ICR_TOKEN="PASTE_YOUR_ICR_API_KEY_HERE"
echo "$ICR_TOKEN" | docker login -u iamapikey --password-stdin icr.io
```

## Push to ICR

```bash
docker push icr.io/altastata/jupyter-datascience-s390x:${JUPYTER_VERSION}
```

Optional logout:

```bash
docker logout icr.io
```

---

## RAG Open LLM (s390x)

Same pattern for the RAG s390x image: build (on s390x host or with buildx), tag for ICR, push.

### Build RAG s390x image

**On an s390x host (recommended:** dependency layer is cached; use `./containers/rag-example/build-rag-s390x-on-server.sh` from your Mac to sync repo and build on the server):

```bash
# On the s390x server (or from Mac via build-rag-s390x-on-server.sh)
source version.sh
docker build -f containers/rag-example/Dockerfile.open_llm_s390x -t altastata/rag-open-llm-s390x:latest -t altastata/rag-open-llm-s390x:${RAG_VERSION} .
```

**On macOS with buildx** (cross-build; no cache from previous s390x builds):

```bash
source version.sh
docker buildx build --platform linux/s390x -f containers/rag-example/Dockerfile.open_llm_s390x \
  -t altastata/rag-open-llm-s390x:latest -t altastata/rag-open-llm-s390x:${RAG_VERSION} --load .
```

(`ENABLE_ZDNN=1 ./containers/rag-example/build-rag-s390x-on-server.sh` tags only `:${RAG_VERSION}_zdnn`, not `:latest`; use the scripts for that.)

### Tag and push RAG s390x to ICR

Uses **`RAG_VERSION`** from `version.sh` (currently **`2026j_latest`** — different from Jupyter’s **`JUPYTER_VERSION`** / `2026e_latest`).

**Push from server** (image was built with `build-rag-s390x-on-server.sh`):
```bash
export ICR_TOKEN="PASTE_YOUR_ICR_API_KEY_HERE"
./containers/rag-example/push-rag-s390x-to-icr-from-server.sh
```

**Or manual (on the server):**
```bash
source version.sh
docker tag altastata/rag-open-llm-s390x:latest icr.io/altastata/rag-open-llm-s390x:${RAG_VERSION}
echo "$ICR_TOKEN" | docker login -u iamapikey --password-stdin icr.io
docker push icr.io/altastata/rag-open-llm-s390x:${RAG_VERSION}
```

For the **`${RAG_VERSION}_zdnn`** research image, set **`ENABLE_ZDNN=1`** when running **`push-rag-s390x-to-icr-from-server.sh`** (see script header).

### Pull and run (for users who pull from ICR)

**From your Mac (script SSHs to server, pulls, runs, and tests):**
```bash
./containers/rag-example/pull-and-run-rag-s390x-from-icr.sh
```
- Set **`ICR_TOKEN`** on your Mac so the script can log in to icr.io on the server before pull.
- The script **stops and removes** any existing `rag-s390x-test` container **before** pulling, then **leaves the new container running** when it finishes (no stop/rm at the end).
- **Default account:** `amazon.rsa.hpcs.serge678` (HPCS; no password). To use the password-based bob123 account:
  ```bash
  ACCOUNT_NAME=amazon.rsa.bob123 ./containers/rag-example/pull-and-run-rag-s390x-from-icr.sh
  ```
- **Optional env (can set before running the script):**
  - `ACCOUNT_NAME` – AltaStata account subdir name (default: `amazon.rsa.hpcs.serge678`). Use `amazon.rsa.bob123` for password-based; the script passes `ALTASTATA_USE_HPCS=1` only when the name contains `hpcs`.
  - `SSH_HOST`, `SSH_KEY` – server and key path (see script defaults).
  - `HF_LLM_MODEL=gpt2` – for 8 GB VMs to avoid OOM (weaker answers).
- The account directory must exist on the **server** at `$REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME` (default `/root/.altastata/accounts/...`).

If you see **"manifest not found"**, the image is not in ICR yet—build on the server then push: `./containers/rag-example/build-rag-s390x-on-server.sh` then `./containers/rag-example/push-rag-s390x-to-icr-from-server.sh`.

**Or manually on the s390x server:**
```bash
source version.sh
docker pull icr.io/altastata/rag-open-llm-s390x:${RAG_VERSION}
# Password-based account (e.g. bob123):
docker run -d -p 8000:8000 --name rag \
  -e ALTASTATA_ACCOUNT_DIR=/root/.altastata/accounts/amazon.rsa.bob123 \
  -e HF_LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  -v $HOME/.altastata/accounts:/root/.altastata/accounts:ro \
  icr.io/altastata/rag-open-llm-s390x:${RAG_VERSION}
# HPCS account (no password): add -e ALTASTATA_USE_HPCS=1
# Open http://<host>:8000/
```
