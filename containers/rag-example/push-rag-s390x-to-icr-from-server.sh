#!/usr/bin/env bash
# Push the RAG s390x image from the LinuxONE server to ICR.
# Prereq: image built on server (./containers/rag-example/build-rag-s390x-on-server.sh). ICR_TOKEN on your Mac.
# Run from repo root: ICR_TOKEN=... ./containers/rag-example/push-rag-s390x-to-icr-from-server.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/version.sh"
SSH_KEY="${SSH_KEY:-/Users/sergevilvovsky/Downloads/torontolinuxonesshkey_rsa.prv}"
SSH_HOST="${SSH_HOST:-root@163.66.89.80}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o GSSAPIAuthentication=no -o PreferredAuthentications=publickey"
REMOTE_DIR="${REMOTE_DIR:-/tmp/altastata-python-package}"

if [ -z "$ICR_TOKEN" ]; then
  echo "Set ICR_TOKEN (your ICR API key) and run again, e.g.:"
  echo "  export ICR_TOKEN=\"your_icr_api_key\""
  echo "  $0"
  exit 1
fi

# Picks tag based on ENABLE_ZDNN (matches build-rag-s390x-on-server.sh):
#   ENABLE_ZDNN=0 (default) -> push :latest -> :$RAG_VERSION (end-user image)
#   ENABLE_ZDNN=1           -> push :${RAG_VERSION}_zdnn -> same (research image)
if [ "${ENABLE_ZDNN:-0}" = "1" ]; then
  SRC_TAG="altastata/rag-open-llm-s390x:${RAG_VERSION}_zdnn"
  DST_TAG="icr.io/altastata/rag-open-llm-s390x:${RAG_VERSION}_zdnn"
else
  SRC_TAG="altastata/rag-open-llm-s390x:latest"
  DST_TAG="icr.io/altastata/rag-open-llm-s390x:$RAG_VERSION"
fi

echo "Pushing from server $SSH_HOST to $DST_TAG (source: $SRC_TAG)"
echo "Logging in to icr.io on server..."
echo "$ICR_TOKEN" | ssh $SSH_OPTS "$SSH_HOST" "docker login -u iamapikey --password-stdin icr.io"
echo "Tag and push..."
ssh $SSH_OPTS "$SSH_HOST" "docker tag $SRC_TAG $DST_TAG && docker push $DST_TAG"
echo "Done. Pull with: docker pull $DST_TAG"