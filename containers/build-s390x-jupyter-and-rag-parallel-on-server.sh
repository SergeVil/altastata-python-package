#!/usr/bin/env bash
# Optional: run from your Mac to rsync the repo to LinuxONE then start parallel
# Docker builds *on the server* (still no Docker on Mac — only rsync + ssh).
#
# Prefer running everything on LinuxONE (no Mac workflows):
#   ssh … "cd /path/to/altastata-python-package && git pull && ./containers/linuxone/build-jupyter-and-rag-in-parallel.sh"
#
# This script: one rsync prep, then Jupyter + RAG s390x builds IN PARALLEL on LinuxONE.
# Avoids invoking build-jupyter + build-rag scripts together (their concurrent
# rsync --delete races the same REMOTE_DIR and can corrupt context).
#
# Run from repo root:
#   ./containers/build-s390x-jupyter-and-rag-parallel-on-server.sh
# Optional same as rag script:
#   ENABLE_ZDNN=1 SSH_HOST=...
#
# Prerequisites: SSH to server, Docker on server, same REMOTE_DIR default as sibling scripts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_KEY="${SSH_KEY:-/Users/sergevilvovsky/Downloads/torontolinuxonesshkey_rsa.prv}"
SSH_HOST="${SSH_HOST:-root@163.66.89.80}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o GSSAPIAuthentication=no -o PreferredAuthentications=publickey"
REMOTE_DIR="${REMOTE_DIR:-/tmp/altastata-python-package}"
REMOTE_ALTASTATA_ACCOUNTS="${REMOTE_ALTASTATA_ACCOUNTS:-/root/.altastata/accounts}"
REMOTE_HPCS_DIR="${REMOTE_HPCS_DIR:-/home/jovyan/hpcs}"
ALTASTATA_ACCOUNTS="${ALTASTATA_ACCOUNTS:-$HOME/.altastata/accounts/amazon.rsa.bob123 $HOME/.altastata/accounts/amazon.rsa.hpcs.serge678}"
ENABLE_ZDNN="${ENABLE_ZDNN:-0}"

echo "Repo root: $REPO_ROOT"
echo "SSH: $SSH_HOST (key: $SSH_KEY)"
echo "REMOTE_DIR=$REMOTE_DIR  ENABLE_ZDNN=$ENABLE_ZDNN"

# --- Same account + repo prep as build-rag-s390x-on-server.sh (covers Jupyter needs too) ---
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

echo "Syncing repo to server (single rsync — safe for parallel docker builds afterward)..."
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
    ln -f /root/llama_models/$F16_FILE $REMOTE_DIR/containers/rag-example/$F16_FILE 2>/dev/null \
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

echo "Building Jupyter + RAG on server concurrently (same host — heavy CPU/mem; consider larger VM)."
# stdin script runs on LinuxONE — vars via env(1) passed by ssh.

# shellcheck disable=SC2086
ssh $SSH_OPTS "$SSH_HOST" env "REMOTE_DIR=$REMOTE_DIR" "ENABLE_ZDNN=$ENABLE_ZDNN" bash -s <<'REMOTE'
set -eo pipefail
cd "$REMOTE_DIR"
# shellcheck source=/dev/null
source version.sh
echo "parallel: Jupyter jupyter-datascience-s390x ..."
docker build --build-arg ALTASTATA_VERSION="${ALTASTATA_PYPI_VERSION}" \
  -f containers/jupyter/Dockerfile.s390x \
  -t altastata/jupyter-datascience-s390x:latest \
  -t "altastata/jupyter-datascience-s390x:${JUPYTER_VERSION}" . &
PID_J="$!"

echo "parallel: RAG rag-open-llm-s390x ..."
if [ "$ENABLE_ZDNN" = "1" ]; then
  RAG_TAGS="-t altastata/rag-open-llm-s390x:${RAG_VERSION}_zdnn"
else
  RAG_TAGS="-t altastata/rag-open-llm-s390x:latest -t altastata/rag-open-llm-s390x:${RAG_VERSION}"
fi
# shellcheck disable=SC2086
docker build --build-arg ENABLE_ZDNN="$ENABLE_ZDNN" \
  -f containers/rag-example/Dockerfile.open_llm_s390x \
  ${RAG_TAGS} . &
PID_R="$!"

ej=0
er=0
wait "$PID_J" || ej=$?
wait "$PID_R" || er=$?
if [ "$ej" -ne 0 ]; then echo "Jupyter docker build exited $ej"; fi
if [ "$er" -ne 0 ]; then echo "RAG docker build exited $er"; fi
exit $(( ej ? ej : er ? er : 0 ))
REMOTE

echo ""
echo "Done. Tagged images on server:"
echo "  altastata/jupyter-datascience-s390x:latest and :\$JUPYTER_VERSION"
echo "  RAG default: altastata/rag-open-llm-s390x:latest and :\$RAG_VERSION ; zDNN: :\${RAG_VERSION}_zdnn"
echo "(Use ../version.sh for exact strings.)"
echo "Push Jupyter: ./containers/jupyter/push-jupyter-s390x-to-icr-from-server.sh"
echo "Push RAG:     ICR_TOKEN=... ./containers/rag-example/push-rag-s390x-to-icr-from-server.sh"
