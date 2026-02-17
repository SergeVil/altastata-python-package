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
  -t altastata/jupyter-datascience-s390x:2026a_latest --load .
```

If you are already on an s390x host, a normal build is fine:

```bash
docker build -f openshift/Dockerfile.s390x -t altastata/jupyter-datascience-s390x:2026a_latest .
```

## Build arm64 (local testing on macOS)

```bash
docker build -f openshift/Dockerfile.arm64 .
```

Run locally:

```bash
docker tag <IMAGE_ID> altastata/jupyter-datascience:arm64-local
docker run --rm -p 8889:8888 altastata/jupyter-datascience:arm64-local
```

Then open `http://localhost:8889/?token=altastata-dev-token` (fixed token for local development).

## Tag s390x image for ICR

```bash
docker tag altastata/jupyter-datascience-s390x:2026a_latest icr.io/altastata/jupyter-datascience-s390x:2026a_latest
```

## Login to ICR

```bash
export ICR_TOKEN="PASTE_NICOLAS_TOKEN_HERE"
echo "$ICR_TOKEN" | docker login -u iamapikey --password-stdin icr.io
```

## Push to ICR

```bash
docker push icr.io/altastata/jupyter-datascience-s390x:2026a_latest
```

Optional logout:

```bash
docker logout icr.io
```
