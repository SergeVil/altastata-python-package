#!/usr/bin/env bash
# Push the RAG s390x image from the LinuxONE server to ICR.
# Prereq: image built on server (./openshift/rag/build-rag-s390x-on-server.sh). ICR_TOKEN on your Mac.
# Run from repo root: ICR_TOKEN=... ./openshift/rag/push-rag-s390x-to-icr-from-server.sh

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

echo "Pushing from server $SSH_HOST to icr.io/altastata/rag-open-llm-s390x:$VERSION"
echo "Logging in to icr.io on server..."
echo "$ICR_TOKEN" | ssh $SSH_OPTS "$SSH_HOST" "docker login -u iamapikey --password-stdin icr.io"
echo "Tag and push..."
ssh $SSH_OPTS "$SSH_HOST" "docker tag altastata/rag-open-llm-s390x:latest icr.io/altastata/rag-open-llm-s390x:$VERSION && docker push icr.io/altastata/rag-open-llm-s390x:$VERSION"
echo "Done. Pull with: docker pull icr.io/altastata/rag-open-llm-s390x:$VERSION"