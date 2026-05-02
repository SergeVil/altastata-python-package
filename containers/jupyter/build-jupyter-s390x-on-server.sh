#!/usr/bin/env bash
# Build the Jupyter DataScience s390x image ON the LinuxONE server.
# Mirror of containers/rag-example/build-rag-s390x-on-server.sh — keeps the
# Jupyter and RAG s390x build flows symmetric (same SSH plumbing, same
# version.sh single-source-of-truth, same tag scheme).
#
# Run from repo root on your Mac (with the SSH key and server reachable).
#
# Usage:
#   ./containers/jupyter/build-jupyter-s390x-on-server.sh
# Or with custom host/key:
#   SSH_KEY=~/.ssh/my.key SSH_HOST=user@10.0.0.1 ./containers/jupyter/build-jupyter-s390x-on-server.sh
#
# Set ALTASTATA_ACCOUNTS="" to skip syncing accounts (build-only, no test data).
#
# If SSH does not connect: check (1) you're on a network that can reach the server (VPN, firewall),
# (2) SSH_HOST and SSH_KEY are correct, (3) server is up and sshd is running. Test with:
#   ssh -i "$SSH_KEY" $SSH_HOST "echo ok"

set -e
# Repo root = directory that contains containers/ and examples/ (altastata-python-package)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SSH_KEY="${SSH_KEY:-/Users/sergevilvovsky/Downloads/torontolinuxonesshkey_rsa.prv}"
SSH_HOST="${SSH_HOST:-root@163.66.89.80}"
# Speed up SSH (avoid GSSAPI/reverse-DNS delays)
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o GSSAPIAuthentication=no -o PreferredAuthentications=publickey"
REMOTE_DIR="${REMOTE_DIR:-/tmp/altastata-python-package}"
REMOTE_ALTASTATA_ACCOUNTS="${REMOTE_ALTASTATA_ACCOUNTS:-/root/.altastata/accounts}"
# Local AltaStata accounts to copy for testing (set ALTASTATA_ACCOUNTS="" to skip).
# Same default as the RAG build script so both share /root/.altastata/accounts on the server.
ALTASTATA_ACCOUNTS="${ALTASTATA_ACCOUNTS:-$HOME/.altastata/accounts/amazon.rsa.bob123 $HOME/.altastata/accounts/amazon.rsa.hpcs.serge678}"

echo "Repo root: $REPO_ROOT"
echo "SSH: $SSH_HOST (key: $SSH_KEY)"
echo "Remote dir: $REMOTE_DIR"

if [ -n "$ALTASTATA_ACCOUNTS" ]; then
  echo "Syncing AltaStata accounts to server ($REMOTE_ALTASTATA_ACCOUNTS)..."
  ssh $SSH_OPTS "$SSH_HOST" "mkdir -p $REMOTE_ALTASTATA_ACCOUNTS"
  for account in $ALTASTATA_ACCOUNTS; do
    if [ -d "$account" ]; then
      name="$(basename "$account")"
      rsync -avz -e "ssh $SSH_OPTS" "$account/" "$SSH_HOST:$REMOTE_ALTASTATA_ACCOUNTS/$name/"
      echo "  -> $name"
    else
      echo "  Skip (not found): $account"
    fi
  done
fi

echo "Syncing repo to server..."
rsync -avz --delete \
  -e "ssh $SSH_OPTS" \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude 'local_index' \
  "$REPO_ROOT/" "$SSH_HOST:$REMOTE_DIR/"

# Single source of truth: version.sh extracts ALTASTATA_PYPI_VERSION from setup.py
# and forwards it to the s390x Dockerfile via --build-arg, so the s390x image
# always pins the same altastata version as the Mac/AMD64 ones (no silent drift
# to "latest from PyPI" when we bump setup.py).
source "$REPO_ROOT/version.sh"
echo "Building image on server (tag: $VERSION, altastata==$ALTASTATA_PYPI_VERSION)..."
ssh $SSH_OPTS "$SSH_HOST" "cd $REMOTE_DIR && source version.sh && docker build --build-arg ALTASTATA_VERSION=\$ALTASTATA_PYPI_VERSION -f containers/jupyter/Dockerfile.s390x -t altastata/jupyter-datascience-s390x:latest -t altastata/jupyter-datascience-s390x:\$VERSION ."

echo ""
echo "Done. Image altastata/jupyter-datascience-s390x:$VERSION (and :latest) is on the server."
echo "Accounts (if synced) are under $REMOTE_ALTASTATA_ACCOUNTS. Run with:"
echo "  ssh -i $SSH_KEY $SSH_HOST 'docker run -d --name altastata-jupyter-s390x \\"
echo "    -p 8888:8888 \\"
echo "    -v $REMOTE_ALTASTATA_ACCOUNTS:/home/jovyan/.altastata/accounts:ro \\"
echo "    altastata/jupyter-datascience-s390x:$VERSION'"
echo ""
echo "Then get the token:"
echo "  ssh -i $SSH_KEY $SSH_HOST 'docker exec altastata-jupyter-s390x jupyter server list'"
echo ""
echo "Push to ICR for end users: ICR_TOKEN=... ./containers/jupyter/push-jupyter-s390x-to-icr-from-server.sh"
