#!/usr/bin/env bash
# Jupyter then RAG s390x Docker images on THIS MACHINE (LinuxONE / s390x only).
#
# This is the build entrypoint ON the VM — do NOT run containers/build-s390x-jupyter-and-rag-on-server.sh
# here (that script is workstation-only: SSH + rsync, then invokes this runner remotely).
#
# Prerequisites: synced repo tree on this VM (typically rsync from Mac; git on VM not required),
#   Docker, and optionally /root/llama_models/llama-3.2-1b-instruct-be.f16.gguf when ENABLE_ZDNN=1
#
# From repo root (after updating the tree on disk):
#   ./containers/linuxone/build-jupyter-and-rag-on-linuxone.sh
#   ENABLE_ZDNN=1 ./containers/linuxone/build-jupyter-and-rag-on-linuxone.sh
# RAG only (skip Jupyter), e.g. after zDNN model change:
#   ENABLE_ZDNN=1 SKIP_JUPYTER=1 ./containers/linuxone/build-jupyter-and-rag-on-linuxone.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

ENABLE_ZDNN="${ENABLE_ZDNN:-0}"
SKIP_JUPYTER="${SKIP_JUPYTER:-0}"
F16_FILE="llama-3.2-1b-instruct-be.f16.gguf"
RAG_MODEL_PATH="containers/rag-example/$F16_FILE"

source ./version.sh

if [ "$ENABLE_ZDNN" = "1" ]; then
  if [ ! -f "/root/llama_models/$F16_FILE" ]; then
    echo "ERROR: /root/llama_models/$F16_FILE missing (needed for ENABLE_ZDNN=1)." >&2
    exit 2
  fi
  ln -f "/root/llama_models/$F16_FILE" "$RAG_MODEL_PATH" 2>/dev/null \
    || cp -f "/root/llama_models/$F16_FILE" "$RAG_MODEL_PATH"
else
  : >"$RAG_MODEL_PATH"
fi

if [ "$SKIP_JUPYTER" = "1" ]; then
  echo "SKIP_JUPYTER=1: skipping Jupyter image."
else
  echo "Building Jupyter (jupyter-datascience-s390x) ..."
  docker build --build-arg ALTASTATA_VERSION="$ALTASTATA_PYPI_VERSION" \
    -f containers/jupyter/Dockerfile.s390x \
    -t altastata/jupyter-datascience-s390x:latest \
    -t "altastata/jupyter-datascience-s390x:${JUPYTER_VERSION}" .
fi

echo "Building RAG (rag-open-llm-s390x) ..."
if [ "$ENABLE_ZDNN" = "1" ]; then
  RAG_TAGS="-t altastata/rag-open-llm-s390x:${RAG_VERSION}_zdnn"
else
  RAG_TAGS="-t altastata/rag-open-llm-s390x:latest -t altastata/rag-open-llm-s390x:${RAG_VERSION}"
fi
# shellcheck disable=SC2086
docker build --build-arg ENABLE_ZDNN="$ENABLE_ZDNN" \
  -f containers/rag-example/Dockerfile.open_llm_s390x \
  ${RAG_TAGS} .

if [ "$SKIP_JUPYTER" = "1" ]; then
  echo "Build finished OK (RAG image only)."
else
  echo "Build finished OK (Jupyter + RAG)."
fi
