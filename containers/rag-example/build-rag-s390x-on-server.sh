#!/usr/bin/env bash
# Build the RAG open_llm s390x image ON the LinuxONE server.
# Run from repo root on your Mac (with the SSH key and server reachable).
#
# Usage:
#   ./containers/rag-example/build-rag-s390x-on-server.sh
# Or with custom host/key:
#   SSH_KEY=~/.ssh/my.key SSH_HOST=user@10.0.0.1 ./containers/rag-example/build-rag-s390x-on-server.sh
#
# If SSH does not connect: check (1) you're on a network that can reach the server (VPN, firewall),
# (2) SSH_HOST and SSH_KEY are correct, (3) server is up and sshd is running. Test with:
#   ssh -i "$SSH_KEY" $SSH_HOST "echo ok"

set -e
# Repo root = directory that contains containers/ and examples/rag-example/ (altastata-python-package)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SSH_KEY="${SSH_KEY:-/Users/sergevilvovsky/Downloads/torontolinuxonesshkey_rsa.prv}"
SSH_HOST="${SSH_HOST:-root@163.66.89.80}"
# Speed up SSH (avoid GSSAPI/reverse-DNS delays)
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o GSSAPIAuthentication=no -o PreferredAuthentications=publickey"
REMOTE_DIR="${REMOTE_DIR:-/tmp/altastata-python-package}"
REMOTE_ALTASTATA_ACCOUNTS="${REMOTE_ALTASTATA_ACCOUNTS:-/root/.altastata/accounts}"
# Local AltaStata accounts to copy for testing (set ALTASTATA_ACCOUNTS="" to skip)
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
  # On server: strip hpcs-yaml-path and hpcs-priv-key-blob-path from *.user.properties so container uses GREP11_YAML / HPCS_PRIV_KEY_BLOB_PATH from env
  echo "Stripping HPCS path lines from account *.user.properties on server..."
  ssh $SSH_OPTS "$SSH_HOST" "for f in $REMOTE_ALTASTATA_ACCOUNTS/*/*.user.properties; do [ -f \"\$f\" ] && sed -i '/^hpcs-yaml-path=/d' \"\$f\" && sed -i '/^hpcs-priv-key-blob-path=/d' \"\$f\" && echo \"  \$(basename \"\$f\")\"; done"
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

# Copy HPCS user.properties from repo to server only if present (file is not in repo by default; put it in containers/rag-example/ if you want it synced)
HPCS_ACCOUNT="amazon.rsa.hpcs.serge678"
HPCS_PROPERTIES_FILE="altastata-myorgrsa444-serge678.user.properties"
HPCS_PROPERTIES_SRC="$REPO_ROOT/containers/rag-example/$HPCS_PROPERTIES_FILE"
if [ -f "$HPCS_PROPERTIES_SRC" ]; then
  echo "Copying $HPCS_PROPERTIES_FILE to server..."
  ssh $SSH_OPTS "$SSH_HOST" "mkdir -p $REMOTE_ALTASTATA_ACCOUNTS/$HPCS_ACCOUNT && cp $REMOTE_DIR/containers/rag-example/$HPCS_PROPERTIES_FILE $REMOTE_ALTASTATA_ACCOUNTS/$HPCS_ACCOUNT/$HPCS_PROPERTIES_FILE && echo '  Done'"
else
  echo "Skipping $HPCS_PROPERTIES_FILE (not in repo). Ensure it exists on server at $REMOTE_ALTASTATA_ACCOUNTS/$HPCS_ACCOUNT/ if using HPCS."
fi

# Use same VERSION as Jupyter (version.sh)
source "$REPO_ROOT/version.sh"
echo "Building image on server (tag: $VERSION)..."
ssh $SSH_OPTS "$SSH_HOST" "cd $REMOTE_DIR && source version.sh && docker build -f containers/rag-example/Dockerfile.open_llm_s390x -t altastata/rag-open-llm-s390x:latest -t altastata/rag-open-llm-s390x:\$VERSION ."

echo "Done. Image altastata/rag-open-llm-s390x:$VERSION (and :latest) is on the server."
echo "Accounts (if synced) are under $REMOTE_ALTASTATA_ACCOUNTS. Run with:"
echo "  docker run -p 8000:8000 -e ALTASTATA_ACCOUNT_DIR=$REMOTE_ALTASTATA_ACCOUNTS/amazon.rsa.bob123 -v $REMOTE_ALTASTATA_ACCOUNTS:$REMOTE_ALTASTATA_ACCOUNTS:ro ... altastata/rag-open-llm-s390x:$VERSION"
