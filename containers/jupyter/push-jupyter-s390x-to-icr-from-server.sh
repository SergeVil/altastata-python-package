#!/usr/bin/env bash
# Push the Jupyter DataScience s390x image from the LinuxONE server to ICR.
# Mirror of containers/rag-example/push-rag-s390x-to-icr-from-server.sh.
# Prereq: image built on server (./containers/jupyter/build-jupyter-s390x-on-server.sh). ICR_TOKEN on your Mac.
# Run from repo root: ICR_TOKEN=... ./containers/jupyter/push-jupyter-s390x-to-icr-from-server.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/version.sh"
SSH_KEY="${SSH_KEY:-/Users/sergevilvovsky/Downloads/torontolinuxonesshkey_rsa.prv}"
SSH_HOST="${SSH_HOST:-root@163.66.89.80}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o GSSAPIAuthentication=no -o PreferredAuthentications=publickey"

if [ -z "$ICR_TOKEN" ]; then
  echo "Set ICR_TOKEN (your ICR API key) and run again, e.g.:"
  echo "  export ICR_TOKEN=\"your_icr_api_key\""
  echo "  $0"
  exit 1
fi

echo "Pushing from server $SSH_HOST to icr.io/altastata/jupyter-datascience-s390x:$JUPYTER_VERSION"
echo "Logging in to icr.io on server..."
echo "$ICR_TOKEN" | ssh $SSH_OPTS "$SSH_HOST" "docker login -u iamapikey --password-stdin icr.io"
echo "Tag and push..."
ssh $SSH_OPTS "$SSH_HOST" "docker tag altastata/jupyter-datascience-s390x:latest icr.io/altastata/jupyter-datascience-s390x:$JUPYTER_VERSION && docker push icr.io/altastata/jupyter-datascience-s390x:$JUPYTER_VERSION"
echo "Done. Pull with: docker pull icr.io/altastata/jupyter-datascience-s390x:$JUPYTER_VERSION"
