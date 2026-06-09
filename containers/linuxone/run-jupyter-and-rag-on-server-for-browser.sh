#!/usr/bin/env bash
# Start Jupyter + RAG s390x containers ON LinuxONE so you can test in a browser.
# Run from repo root on your Mac *after* builds on the server (`build-s390x-jupyter-and-rag-on-server.sh`).
# Containers keep running until you stop them (no curl smoke, no auto-remove).
#
# Default images (from version.sh):
#   altastata/jupyter-datascience-s390x:${JUPYTER_VERSION}
#   altastata/rag-open-llm-s390x:${RAG_VERSION}
#
# Optional env:
#   SSH_HOST SSH_KEY ACCOUNT_NAME (RAG) RUN_JUP=0 or RUN_RAG=0 to skip one service
#   JUPYTER_HOST_PORT (default 8888) RAG_HOST_PORT (default 8000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../../version.sh
source "${REPO_ROOT}/version.sh"

SSH_KEY="${SSH_KEY:-/Users/sergevilvovsky/Downloads/torontolinuxonesshkey_rsa.prv}"
SSH_HOST="${SSH_HOST:-root@163.66.89.80}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o GSSAPIAuthentication=no -o PreferredAuthentications=publickey"
PUBLIC_HOST="${PUBLIC_HOST:-${SSH_HOST#*@}}"

REMOTE_ALTASTATA_ACCOUNTS="${REMOTE_ALTASTATA_ACCOUNTS:-/root/.altastata/accounts}"
ACCOUNT_NAME="${ACCOUNT_NAME:-amazon.rsa.bob123}"
REMOTE_GREP11_YAML="${REMOTE_GREP11_YAML:-/etc/ep11client/grep11client.yaml}"
REMOTE_HPCS_DIR="${REMOTE_HPCS_DIR:-/home/jovyan/hpcs}"
REMOTE_HPCS_BLOB="${REMOTE_HPCS_BLOB:-$REMOTE_HPCS_DIR/hpcs-privkey.blob}"
REMOTE_PROPERTIES_FILE="${REMOTE_PROPERTIES_FILE:-$REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME/altastata-myorgrsa444-serge678.user.properties}"
REMOTE_MODELS_DIR="${REMOTE_MODELS_DIR:-/root/llama_models}"

JUP_IMG="altastata/jupyter-datascience-s390x:${JUPYTER_VERSION}"
RAG_IMG="altastata/rag-open-llm-s390x:${RAG_VERSION}"

JUP_CTR="${JUP_CTR:-altastata-jupyter-s390x-web}"
RAG_CTR="${RAG_CTR:-rag-open-llm-s390x-web}"

JUPYTER_HOST_PORT="${JUPYTER_HOST_PORT:-8888}"
RAG_HOST_PORT="${RAG_HOST_PORT:-8000}"

REMOTE_JUP_WORK="${REMOTE_JUP_WORK:-/root/jupyter-web-work}"

LLM_PROVIDER="${LLM_PROVIDER:-llama-cpp}"
QUERY_TIMEOUT="${QUERY_TIMEOUT:-400}"
# RAG_TIMING_LOG=1 turns on per-query phase timings in the RAG container logs
# (vector store / similarity / AltaStata chunk reads / LLM TTFT / llama.cpp perf print).
# Default ON for now so investigations always have data; set RAG_TIMING_LOG=0 to silence.
RAG_TIMING_LOG="${RAG_TIMING_LOG:-1}"

RUN_JUP="${RUN_JUP:-1}"
RUN_RAG="${RUN_RAG:-1}"

echo "SSH: $SSH_HOST  (browser: http://$PUBLIC_HOST:... )"
echo "Jupyter image: $JUP_IMG  container: $JUP_CTR  port: $JUPYTER_HOST_PORT"
echo "RAG image:     $RAG_IMG  container: $RAG_CTR  port: $RAG_HOST_PORT"
echo "RAG account:   $REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME"
echo ""

ssh $SSH_OPTS "$SSH_HOST" "mkdir -p '$REMOTE_JUP_WORK' '$REMOTE_MODELS_DIR'" || true

# Console UI (gRPC + SPA) defaults to enabled inside the Jupyter image but is
# only reachable through the port we publish here. Bind 9877 the same way 8888
# is bound; override with JUPYTER_CONSOLE_UI_HOST_PORT.
JUPYTER_CONSOLE_UI_HOST_PORT="${JUPYTER_CONSOLE_UI_HOST_PORT:-9877}"

# HPCS auto-mount for Jupyter: when the canonical HPCS files exist on the VM,
# mount them at the paths the example notebooks reference. This matches what
# the RAG block below already does. If you don't use HPCS the mounts are
# simply skipped (no failure). The mount destinations match the
# user_properties shipped in the example notebooks:
#   hpcs-yaml-path=/etc/ep11client/grep11client.yaml
#   hpcs-priv-key-blob-path=/home/jovyan/hpcs-privkey.blob
JUP_HPCS_MOUNTS=""
if ssh $SSH_OPTS "$SSH_HOST" "test -f '$REMOTE_GREP11_YAML'" 2>/dev/null; then
  JUP_HPCS_MOUNTS="$JUP_HPCS_MOUNTS -v $REMOTE_GREP11_YAML:/etc/ep11client/grep11client.yaml:ro"
fi
if ssh $SSH_OPTS "$SSH_HOST" "test -f '$REMOTE_HPCS_BLOB'" 2>/dev/null; then
  JUP_HPCS_MOUNTS="$JUP_HPCS_MOUNTS -v $REMOTE_HPCS_BLOB:/home/jovyan/hpcs-privkey.blob:ro"
fi
[ -n "$JUP_HPCS_MOUNTS" ] && echo "  HPCS mounts for Jupyter:$JUP_HPCS_MOUNTS"

if [ "$RUN_JUP" = "1" ]; then
  echo "--- Starting Jupyter (detached)"
  ssh $SSH_OPTS "$SSH_HOST" "docker rm -f '$JUP_CTR' 2>/dev/null || true"
  ssh $SSH_OPTS "$SSH_HOST" "docker run -d --name '$JUP_CTR' \
    -p '${JUPYTER_HOST_PORT}:8888' \
    -p '${JUPYTER_CONSOLE_UI_HOST_PORT}:9877' \
    -e 'ENABLE_ALTASTATA_CONSOLE_UI=${ENABLE_ALTASTATA_CONSOLE_UI:-1}' \
    -v '${REMOTE_JUP_WORK}:/home/jovyan/work' \
    -v '/root/.altastata:/opt/app-root/src/.altastata:rw' \
    $JUP_HPCS_MOUNTS \
    '$JUP_IMG'"
else
  echo "--- Skipping Jupyter (RUN_JUP=$RUN_JUP)"
fi

if [ "$RUN_RAG" = "1" ]; then
  HPCS_ENV=""
  HPCS_MOUNTS=""
  case "$ACCOUNT_NAME" in *hpcs*)
    ssh $SSH_OPTS "$SSH_HOST" "for f in '$REMOTE_GREP11_YAML' '$REMOTE_HPCS_BLOB' '$REMOTE_PROPERTIES_FILE'; do
      [ -f \"\$f\" ] || { echo \"Missing: \$f\"; exit 1; }; echo \"  OK \$f\"; done"
    HPCS_ENV="-e ALTASTATA_USE_HPCS=1 -e GREP11_YAML=/etc/ep11client/grep11client.yaml -e HPCS_PRIV_KEY_BLOB_PATH=/home/jovyan/hpcs/hpcs-privkey.blob"
    HPCS_MOUNTS="-v $REMOTE_GREP11_YAML:/etc/ep11client/grep11client.yaml:ro -v $REMOTE_HPCS_DIR:/home/jovyan/hpcs:ro"
    ;;
  esac

  LLAMA_OPTS=""
  [ -n "${LLAMA_CPP_MODEL_REPO+x}" ] && LLAMA_OPTS="$LLAMA_OPTS -e LLAMA_CPP_MODEL_REPO=$LLAMA_CPP_MODEL_REPO"
  [ -n "${LLAMA_CPP_MODEL_FILE:-}" ] && LLAMA_OPTS="$LLAMA_OPTS -e LLAMA_CPP_MODEL_FILE=$LLAMA_CPP_MODEL_FILE"
  [ -n "${HF_LLM_MODEL:-}" ]      && LLAMA_OPTS="$LLAMA_OPTS -e HF_LLM_MODEL=$HF_LLM_MODEL"

  echo "--- Starting RAG (detached)"
  ssh $SSH_OPTS "$SSH_HOST" "docker rm -f '$RAG_CTR' 2>/dev/null || true"
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "$SSH_HOST" "docker run -d --name '$RAG_CTR' -p '${RAG_HOST_PORT}:8000' \
    -e ALTASTATA_ACCOUNT_DIR=$REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME \
    $HPCS_ENV \
    -e LLM_PROVIDER=$LLM_PROVIDER \
    -e QUERY_TIMEOUT=$QUERY_TIMEOUT \
    -e RAG_TIMING_LOG=$RAG_TIMING_LOG \
    $LLAMA_OPTS \
    -v $REMOTE_ALTASTATA_ACCOUNTS:$REMOTE_ALTASTATA_ACCOUNTS:ro \
    -v $REMOTE_MODELS_DIR:/models \
    $HPCS_MOUNTS \
    '$RAG_IMG'"
else
  echo "--- Skipping RAG (RUN_RAG=$RUN_RAG)"
fi

echo ""
echo "===== Browser ====="
[ "$RUN_JUP" = "1" ] && {
  echo "Jupyter Lab:    http://$PUBLIC_HOST:$JUPYTER_HOST_PORT/"
  echo "Console UI:     http://$PUBLIC_HOST:$JUPYTER_CONSOLE_UI_HOST_PORT/   (set gRPC base URL to the same host:port on first load)"
  echo "Get token:      ssh $SSH_OPTS $SSH_HOST \"docker exec '$JUP_CTR' jupyter server list\""
  echo "Logs:           ssh $SSH_OPTS $SSH_HOST \"docker logs -f '$JUP_CTR'\""
}
[ "$RUN_RAG" = "1" ] && {
  echo "RAG web app:   http://$PUBLIC_HOST:$RAG_HOST_PORT/"
  echo "Logs:           ssh $SSH_OPTS $SSH_HOST \"docker logs -f '$RAG_CTR'\""
}
echo ""
echo "Allow inbound 8888 / 8000 (or your ports) on the VPC security group if needed."
echo "Stop: ssh $SSH_OPTS $SSH_HOST \"docker stop '$JUP_CTR' '$RAG_CTR'; docker rm '$JUP_CTR' '$RAG_CTR'\""
echo " (ignore errors if only one was started)"
