#!/usr/bin/env bash
# Build the RAG open_llm image for Mac (same layout as s390x: AltaStata only, index at startup) and run it.
# Run from repo root.
#
# Usage:
#   ALTASTATA_ACCOUNT_DIR=$HOME/.altastata/accounts/amazon.rsa.bob123 ./openshift/rag/build-and-run-rag-mac.sh
#   # or: ALTASTATA_ACCOUNT_DIR=/full/path/to/.altastata/accounts/amazon.rsa.bob123 ./openshift/rag/build-and-run-rag-mac.sh
#
# Default password for bob123-style accounts is 123. Override: ALTASTATA_PASSWORD=yourpass
# Faster answers: run Ollama on Mac (ollama run smollm2:360m) then use LLM_PROVIDER=ollama.
# Optional: HF_LLM_MODEL=... LLM_PROVIDER=transformers (default). For HPCS set ALTASTATA_USE_HPCS=1.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-altastata/rag-open-llm:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-rag-mac}"
WEB_PORT="${WEB_PORT:-8000}"

ALTASTATA_ACCOUNT_DIR="${ALTASTATA_ACCOUNT_DIR:-}"
if [ -z "$ALTASTATA_ACCOUNT_DIR" ] || [ ! -d "$ALTASTATA_ACCOUNT_DIR" ]; then
  echo "Set ALTASTATA_ACCOUNT_DIR to your AltaStata account directory (e.g. \$HOME/.altastata/accounts/amazon.rsa.bob123)"
  exit 1
fi

echo "Repo root: $REPO_ROOT"
echo "Account dir: $ALTASTATA_ACCOUNT_DIR"
echo "Building image: $IMAGE_NAME"

docker build -f "$SCRIPT_DIR/Dockerfile.open_llm_mac" -t "$IMAGE_NAME" "$REPO_ROOT"

# Stop/remove existing container if present
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

ACCOUNT_PARENT="$(dirname "$ALTASTATA_ACCOUNT_DIR")"
ACCOUNT_BASENAME="$(basename "$ALTASTATA_ACCOUNT_DIR")"
CONTAINER_ACCOUNT_DIR="/altastata_account/$ACCOUNT_BASENAME"
# Named volume so the index persists across container restarts (no "No index found" after rebuild)
INDEX_VOLUME="${INDEX_VOLUME:-rag-mac-index}"
RUN_OPTS=(
  -d
  -p "$WEB_PORT:8000"
  --name "$CONTAINER_NAME"
  -e "ALTASTATA_ACCOUNT_DIR=$CONTAINER_ACCOUNT_DIR"
  -e "ALTASTATA_ACCOUNT_ID=${ALTASTATA_ACCOUNT_ID:-bob123}"
  -v "$ACCOUNT_PARENT:/altastata_account:ro"
  -v "$INDEX_VOLUME:/app/open_llm/local_index"
)
# Default password 123 for bob123; skip for HPCS
if [ -n "${ALTASTATA_USE_HPCS:-}" ]; then
  RUN_OPTS+=(-e "ALTASTATA_USE_HPCS=$ALTASTATA_USE_HPCS")
else
  RUN_OPTS+=(-e "ALTASTATA_PASSWORD=${ALTASTATA_PASSWORD:-123}")
fi
[ -n "${LLM_PROVIDER:-}" ] && RUN_OPTS+=(-e "LLM_PROVIDER=$LLM_PROVIDER")
# Ollama on host = much faster (Metal). Run: ollama run smollm2:360m  then use LLM_PROVIDER=ollama
if [ "${LLM_PROVIDER:-}" = "ollama" ]; then
  RUN_OPTS+=(-e "OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://host.docker.internal:11434}")
fi
[ -n "${HF_LLM_MODEL:-}" ] && RUN_OPTS+=(-e "HF_LLM_MODEL=$HF_LLM_MODEL")
[ -n "${RAG_INDEX_PATH:-}" ] && RUN_OPTS+=(-e "RAG_INDEX_PATH=$RAG_INDEX_PATH")

echo "Container env (account): ALTASTATA_ACCOUNT_DIR=$CONTAINER_ACCOUNT_DIR, ALTASTATA_ACCOUNT_ID=${ALTASTATA_ACCOUNT_ID:-bob123}"
echo "Container mounts: $ACCOUNT_PARENT -> /altastata_account:ro, $INDEX_VOLUME -> /app/open_llm/local_index"
echo "RAG_INDEX_PATH (in container): ${RAG_INDEX_PATH:-RAGDocs/policies}"
echo "Running container: $CONTAINER_NAME (port $WEB_PORT)..."
docker run "${RUN_OPTS[@]}" "$IMAGE_NAME"
echo "Container started. To follow logs: docker logs -f $CONTAINER_NAME"

echo "Done. Open http://localhost:$WEB_PORT/"
echo "Index is in volume $INDEX_VOLUME (persists across restarts). Wait 1–2 min on first run for indexer + server startup, then query."
echo "For much faster answers, run Ollama on the host (e.g. ollama run smollm2:360m) and restart with: LLM_PROVIDER=ollama $0 (with same ALTASTATA_ACCOUNT_DIR)."
echo ""
echo "If you see 'No index found' in the UI: check container logs for AltaStata/indexer errors:"
echo "  docker logs $CONTAINER_NAME"
echo "Ensure ALTASTATA_ACCOUNT_ID matches your account (default bob123), and that documents exist at RAG_INDEX_PATH (default RAGDocs/policies) in AltaStata."
