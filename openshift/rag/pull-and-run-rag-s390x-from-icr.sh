#!/usr/bin/env bash
# Pull the RAG s390x image from ICR and run it on the server (no local build).
# Run from repo root on your Mac. Uses same VERSION as Jupyter (version.sh).
#
# Memory: We recommend 16 GB RAM for the default model (TinyLlama). On 8 GB, set
#   HF_LLM_MODEL=gpt2 to avoid OOM (weaker answers).
# Hardware: Inference runs on CPU; NNPA/zDNN acceleration on IBM Z (e.g. z16/z17) can
#   speed up inference when available in the runtime environment.
#
# ICR only (no Docker Hub). Set ICR_TOKEN on your Mac (e.g. in .zshrc or for this run:
#   export ICR_TOKEN="your_icr_api_key"). The script uses it to log in to icr.io on the
# server before pull—so you never get a Docker Hub "Username?" prompt.
#
# Usage: ./openshift/rag/pull-and-run-rag-s390x-from-icr.sh
# Optional: ACCOUNT_NAME=amazon.rsa.hpcs.serge678 ICR_TOKEN=... SSH_HOST=... SSH_KEY=...
#           REMOTE_GREP11_YAML=... REMOTE_HPCS_BLOB=... REMOTE_PROPERTIES_FILE=... (paths on server)
#           For 8 GB VM set HF_LLM_MODEL=gpt2 to avoid OOM.
# Account dir must exist on server at $REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME (default /root/.altastata/accounts/...).
# For HPCS we use three files on the server:
#   1. grep11client.yaml   — REMOTE_GREP11_YAML (default /etc/ep11client/grep11client.yaml)
#   2. hpcs-privkey.blob   — REMOTE_HPCS_BLOB (default: inside account dir, .../amazon.rsa.hpcs.serge678/hpcs-privkey.blob)
#   3. *.user.properties   — REMOTE_PROPERTIES_FILE (default .../amazon.rsa.hpcs.serge678/altastata-myorgrsa444-serge678.user.properties)
# Yaml is mounted separately; blob and properties live in the account dir (per-user). Omit hpcs-yaml-path and hpcs-priv-key-blob-path from the properties file.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/version.sh"
SSH_KEY="${SSH_KEY:-/Users/sergevilvovsky/Downloads/torontolinuxonesshkey_rsa.prv}"
SSH_HOST="${SSH_HOST:-root@163.66.89.80}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o GSSAPIAuthentication=no -o PreferredAuthentications=publickey"
REMOTE_ALTASTATA_ACCOUNTS="${REMOTE_ALTASTATA_ACCOUNTS:-/root/.altastata/accounts}"
ACCOUNT_NAME="${ACCOUNT_NAME:-amazon.rsa.hpcs.serge678}"
# All three files on the server (override with env if your paths differ)
REMOTE_GREP11_YAML="${REMOTE_GREP11_YAML:-/etc/ep11client/grep11client.yaml}"
REMOTE_HPCS_BLOB="${REMOTE_HPCS_BLOB:-$REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME/hpcs-privkey.blob}"
REMOTE_PROPERTIES_FILE="${REMOTE_PROPERTIES_FILE:-$REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME/altastata-myorgrsa444-serge678.user.properties}"
CONTAINER_NAME="rag-s390x-test"
MAX_WAIT="${MAX_WAIT:-300}"
HF_LLM_MODEL="${HF_LLM_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
# Transformers on CPU (s390x) is slow; allow longer for first query
QUERY_TIMEOUT="${QUERY_TIMEOUT:-400}"
RAG_IMAGE="icr.io/altastata/rag-open-llm-s390x:$VERSION"

echo "SSH: $SSH_HOST"
echo "Pull and run: $RAG_IMAGE"
echo "Account: $REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME"
echo "Three files on server:"
echo "  yaml:      $REMOTE_GREP11_YAML"
echo "  blob:      $REMOTE_HPCS_BLOB"
echo "  properties: $REMOTE_PROPERTIES_FILE"
echo "HF_LLM_MODEL: $HF_LLM_MODEL"
if [ -n "$ICR_TOKEN" ]; then
  echo "Logging in to icr.io on server (ICR only; no Docker Hub)..."
  echo "$ICR_TOKEN" | ssh $SSH_OPTS "$SSH_HOST" "docker login -u iamapikey --password-stdin icr.io"
fi
echo "Stopping and removing existing container (if any)..."
ssh $SSH_OPTS "$SSH_HOST" "docker stop $CONTAINER_NAME 2>/dev/null || true; docker rm $CONTAINER_NAME 2>/dev/null || true"
echo "Pulling image on server..."
ssh $SSH_OPTS "$SSH_HOST" "docker pull $RAG_IMAGE"

# HPCS: ensure all three files exist on server. Mount yaml; blob and properties are in the account dir (already mounted).
HPCS_ENV=""
HPCS_MOUNTS=""
case "$ACCOUNT_NAME" in *hpcs*)
  echo "Checking HPCS files on server..."
  ssh $SSH_OPTS "$SSH_HOST" "for f in '$REMOTE_GREP11_YAML' '$REMOTE_HPCS_BLOB' '$REMOTE_PROPERTIES_FILE'; do if [ ! -f \"\$f\" ]; then echo \"Missing: \$f\"; exit 1; fi; echo \"  OK \$f\"; done"
  HPCS_ENV="-e ALTASTATA_USE_HPCS=1 -e GREP11_YAML=/etc/ep11client/grep11client.yaml -e HPCS_PRIV_KEY_BLOB_PATH=$REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME/hpcs-privkey.blob"
  HPCS_MOUNTS="-v $REMOTE_GREP11_YAML:/etc/ep11client/grep11client.yaml:ro"
  ;;
esac
echo "Starting container $CONTAINER_NAME..."
ssh $SSH_OPTS "$SSH_HOST" "docker run -d --name $CONTAINER_NAME -p 8000:8000 \
  -e ALTASTATA_ACCOUNT_DIR=$REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME \
  $HPCS_ENV \
  -e HF_LLM_MODEL=$HF_LLM_MODEL \
  -e QUERY_TIMEOUT=$QUERY_TIMEOUT \
  -v $REMOTE_ALTASTATA_ACCOUNTS:$REMOTE_ALTASTATA_ACCOUNTS:ro \
  $HPCS_MOUNTS \
  $RAG_IMAGE"

echo "Waiting for app (up to ${MAX_WAIT}s)..."
waited=0
while [ "$waited" -lt "$MAX_WAIT" ]; do
  if ssh $SSH_OPTS "$SSH_HOST" "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/" 2>/dev/null | grep -q 200; then
    echo "App is up."
    break
  fi
  sleep 10
  waited=$((waited + 10))
  echo "  ${waited}s..."
done
if [ "$waited" -ge "$MAX_WAIT" ]; then
  echo "Timeout waiting for app. Logs:"
  ssh $SSH_OPTS "$SSH_HOST" "docker logs $CONTAINER_NAME 2>&1 | tail -50"
  exit 1
fi

echo "Sending POST /query (first run may load the model; timeout ${QUERY_TIMEOUT}s)..."
ssh $SSH_OPTS "$SSH_HOST" "curl -s --max-time $((QUERY_TIMEOUT + 60)) -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{\"query\": \"What are the password requirements?\"}'"

echo ""
echo "Done. Container $CONTAINER_NAME is running on the server (port 8000)."
echo "Open http://<server-ip>:8000/ or stop: ssh $SSH_OPTS $SSH_HOST 'docker stop $CONTAINER_NAME; docker rm $CONTAINER_NAME'"
