#!/usr/bin/env bash
# Jupyter then RAG s390x Docker images on THIS MACHINE (LinuxONE / s390x only).
#
# Prerequisites: git checkout, Docker, optionally
#   /root/llama_models/llama-3.2-1b-instruct-be.f16.gguf when ENABLE_ZDNN=1
#
# From repo root (after git pull):
#   ./containers/linuxone/build-jupyter-and-rag-on-linuxone.sh
#   ENABLE_ZDNN=1 ./containers/linuxone/build-jupyter-and-rag-on-linuxone.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

ENABLE_ZDNN="${ENABLE_ZDNN:-0}"
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

echo "Building Jupyter (jupyter-datascience-s390x) ..."
docker build --build-arg ALTASTATA_VERSION="$ALTASTATA_PYPI_VERSION" \
  -f containers/jupyter/Dockerfile.s390x \
  -t altastata/jupyter-datascience-s390x:latest \
  -t "altastata/jupyter-datascience-s390x:${JUPYTER_VERSION}" .

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

echo "Both sequential builds finished OK."
