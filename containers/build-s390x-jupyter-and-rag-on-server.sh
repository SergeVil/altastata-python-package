#!/usr/bin/env bash
# Workstation/Mac only — rsync + SSH orchestration. Do NOT run this script ON LinuxONE.
# On IBM Z/LinuxONE VM (after your tree is in place): only
#   ./containers/linuxone/build-jupyter-and-rag-on-linuxone.sh
#
# Sync repo (+ accounts blob) from your Mac to LinuxONE, then start Jupyter+RAG s390x
# Docker builds *on LinuxONE*.
#
# **Default DETACHED=1:** starts builds under nohup and returns immediately so your Mac is
# not tied to a hours-long SSH session — work on the Mac in parallel with builds on IBM Z.
#
# Blocking (stream logs until done): DETACHED=0 ./containers/build-s390x-jupyter-and-rag-on-server.sh
#
# Pure LinuxONE (no rsync): after git pull, run
#   ./containers/linuxone/build-jupyter-and-rag-on-linuxone.sh
#
# Run from repo root:
#   ./containers/build-s390x-jupyter-and-rag-on-server.sh
# Optional:
#   ENABLE_ZDNN=1 SSH_HOST=... REMOTE_BUILD_LOG=/tmp/foo.log DETACHED=0 ...
#
set -euo pipefail

if [ "${SKIP_S390X_WORKSTATION_CHECK:-}" != "1" ] && command -v uname >/dev/null 2>&1 && [ "$(uname -m)" = "s390x" ]; then
    echo "$(basename "$0"): this script orchestrates SSH/rsync FROM a workstation; you are ON s390x." >&2
    echo "On LinuxONE, from repo root run only:" >&2
    echo "  ./containers/linuxone/build-jupyter-and-rag-on-linuxone.sh" >&2
    echo "(Override: SKIP_S390X_WORKSTATION_CHECK=1)" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_KEY="${SSH_KEY:-/Users/sergevilvovsky/Downloads/torontolinuxonesshkey_rsa.prv}"
SSH_HOST="${SSH_HOST:-root@163.66.89.80}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o GSSAPIAuthentication=no -o PreferredAuthentications=publickey"
REMOTE_DIR="${REMOTE_DIR:-/tmp/altastata-python-package}"
REMOTE_ALTASTATA_ACCOUNTS="${REMOTE_ALTASTATA_ACCOUNTS:-/root/.altastata/accounts}"
REMOTE_HPCS_DIR="${REMOTE_HPCS_DIR:-/home/jovyan/hpcs}"
ALTASTATA_ACCOUNTS="${ALTASTATA_ACCOUNTS:-$HOME/.altastata/accounts/amazon.rsa.bob123 $HOME/.altastata/accounts/amazon.rsa.hpcs.serge678}"
ENABLE_ZDNN="${ENABLE_ZDNN:-0}"
DETACHED="${DETACHED:-1}"
REMOTE_BUILD_LOG="${REMOTE_BUILD_LOG:-/tmp/altastata-s390x-jupyter-rag-build.log}"

echo "Repo root: $REPO_ROOT"
echo "SSH: $SSH_HOST (key: $SSH_KEY)"
echo "REMOTE_DIR=$REMOTE_DIR ENABLE_ZDNN=$ENABLE_ZDNN DETACHED=$DETACHED log=$REMOTE_BUILD_LOG"

if [ -n "$ALTASTATA_ACCOUNTS" ]; then
  echo "Syncing AltaStata accounts..."
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
  echo "Stripping HPCS path lines from account *.user.properties on server..."
  ssh $SSH_OPTS "$SSH_HOST" "for f in $REMOTE_ALTASTATA_ACCOUNTS/*/*.user.properties; do [ -f \"\$f\" ] && sed -i '/^hpcs-yaml-path=/d' \"\$f\" && sed -i '/^hpcs-priv-key-blob-path=/d' \"\$f\" && echo \"  \$(basename \"\$f\")\"; done"
  HPCS_BLOB_SRC="$HOME/.altastata/accounts/amazon.rsa.hpcs.serge678/hpcs-privkey.blob"
  if [ -f "$HPCS_BLOB_SRC" ]; then
    echo "Copying HPCS key blob to server..."
    ssh $SSH_OPTS "$SSH_HOST" "mkdir -p $REMOTE_HPCS_DIR"
    rsync -avz -e "ssh $SSH_OPTS" "$HPCS_BLOB_SRC" "$SSH_HOST:$REMOTE_HPCS_DIR/hpcs-privkey.blob"
  fi
fi

echo "Syncing repo to LinuxONE ..."
rsync -avz --delete \
  -e "ssh $SSH_OPTS" \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude 'local_index' \
  "$REPO_ROOT/" "$SSH_HOST:$REMOTE_DIR/"

HPCS_ACCOUNT="amazon.rsa.hpcs.serge678"
HPCS_PROPERTIES_FILE="altastata-myorgrsa444-serge678.user.properties"
HPCS_PROPERTIES_SRC="$REPO_ROOT/containers/rag-example/$HPCS_PROPERTIES_FILE"
if [ -f "$HPCS_PROPERTIES_SRC" ]; then
  echo "Copying $HPCS_PROPERTIES_FILE to server..."
  ssh $SSH_OPTS "$SSH_HOST" "mkdir -p $REMOTE_ALTASTATA_ACCOUNTS/$HPCS_ACCOUNT && cp $REMOTE_DIR/containers/rag-example/$HPCS_PROPERTIES_FILE $REMOTE_ALTASTATA_ACCOUNTS/$HPCS_ACCOUNT/$HPCS_PROPERTIES_FILE && echo '  Done'"
fi

F16_FILE="llama-3.2-1b-instruct-be.f16.gguf"
if [ "$ENABLE_ZDNN" = "1" ]; then
  echo "ENABLE_ZDNN=1: staging real $F16_FILE into build context on server..."
  ssh $SSH_OPTS "$SSH_HOST" "
    set -e
    if [ ! -f /root/llama_models/$F16_FILE ]; then
      echo 'ERROR: /root/llama_models/$F16_FILE missing on server.'
      exit 2
    fi
    ln -f /root/llama_models/$F16_FILE $REMOTE_DIR/containers/rag-example/$F16_FILE 2>/dev/null \\
      || cp -f /root/llama_models/$F16_FILE $REMOTE_DIR/containers/rag-example/$F16_FILE
    ls -la $REMOTE_DIR/containers/rag-example/$F16_FILE
  "
else
  ssh $SSH_OPTS "$SSH_HOST" "
    set -e
    : > $REMOTE_DIR/containers/rag-example/$F16_FILE
    ls -la $REMOTE_DIR/containers/rag-example/$F16_FILE
  "
fi

if [ "$DETACHED" = "1" ]; then
  echo "Starting builds on LinuxONE under nohup (this SSH exits now; Jupyter then RAG sequentially on the server)."
  ssh $SSH_OPTS "$SSH_HOST" bash -s -- "$REMOTE_DIR" "$REMOTE_BUILD_LOG" "$ENABLE_ZDNN" <<'EOS'
set -euo pipefail
REMOTE_DIR="$1"
BUILD_LOG="$2"
ENABLE_ZDNN="$3"
RUNNER="$REMOTE_DIR/containers/linuxone/build-jupyter-and-rag-on-linuxone.sh"
chmod +x "$RUNNER" 2>/dev/null || true
touch "$BUILD_LOG"
nohup env ENABLE_ZDNN="$ENABLE_ZDNN" bash -eo pipefail "$RUNNER" >>"$BUILD_LOG" 2>&1 </dev/null &
echo "Detached PID=$!  log=$BUILD_LOG"
EOS
  echo ""
  echo "Tail on LinuxONE (from any machine): ssh $SSH_HOST tail -f $REMOTE_BUILD_LOG"
else
  echo "Blocking mode: streaming build output from LinuxONE ..."
  ssh $SSH_OPTS "$SSH_HOST" bash -s -- "$REMOTE_DIR" "$ENABLE_ZDNN" <<'EOS'
set -euo pipefail
REMOTE_DIR="$1"
ENABLE_ZDNN="$2"
RUNNER="$REMOTE_DIR/containers/linuxone/build-jupyter-and-rag-on-linuxone.sh"
chmod +x "$RUNNER" 2>/dev/null || true
exec env ENABLE_ZDNN="$ENABLE_ZDNN" bash -eo pipefail "$RUNNER"
EOS
fi

echo ""
echo "When builds finish (see log), run Jupyter + RAG for browser testing before ICR push:"
echo "  ./containers/linuxone/run-jupyter-and-rag-on-server-for-browser.sh"
echo "Push Jupyter: ./containers/jupyter/push-jupyter-s390x-to-icr-from-server.sh"
echo "Push RAG:     ICR_TOKEN=... ./containers/rag-example/push-rag-s390x-to-icr-from-server.sh"
