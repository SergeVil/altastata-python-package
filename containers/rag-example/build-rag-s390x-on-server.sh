#!/usr/bin/env bash
# Build the RAG open_llm s390x image on LinuxONE by rsync + ssh from your **workstation** (Mac).
# If you are already **on** the LinuxONE VM with the repo tree on disk, do NOT use this script —
# run from repo root:
#   ENABLE_ZDNN=1 ./containers/linuxone/build-jupyter-and-rag-on-linuxone.sh
# (Optionally SKIP_JUPYTER=1 to build only RAG.)
#
# Usage (from Mac):
#   ./containers/rag-example/build-rag-s390x-on-server.sh
# Or with custom host/key:
#   SSH_KEY=~/.ssh/my.key SSH_HOST=user@10.0.0.1 ./containers/rag-example/build-rag-s390x-on-server.sh
#
# If SSH does not connect: check (1) you're on a network that can reach the server (VPN, firewall),
# (2) SSH_HOST and SSH_KEY are correct, (3) server is up and sshd is running. Test with:
#   ssh -i "$SSH_KEY" $SSH_HOST "echo ok"
#
# If your Mac reports exit 137 / "Killed: 9" on the ssh line: the *local* SSH client was
# SIGKILL'd—often memory pressure on the Mac when another heavy Docker build runs in parallel,
# or the IDE closing the session. Fix: run only this build (pause other local Docker work), or
# SSH to LinuxONE and run the same `docker build ...` inside tmux so losing the Mac session
# does not cancel the server-side build.

set -e

if [ "${SKIP_S390X_WORKSTATION_CHECK:-}" != "1" ] && command -v uname >/dev/null 2>&1 && [ "$(uname -m)" = "s390x" ]; then
  echo "$(basename "$0"): this script rsync + SSH FROM a workstation; you are ON s390x." >&2
  echo "On LinuxONE, from repo root run:" >&2
  echo "  ENABLE_ZDNN=1 ./containers/linuxone/build-jupyter-and-rag-on-linuxone.sh" >&2
  echo "(Optional: SKIP_JUPYTER=1 for RAG only.) Override: SKIP_S390X_WORKSTATION_CHECK=1" >&2
  exit 2
fi

# Repo root = directory that contains containers/ and examples/rag-example/ (altastata-python-package)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SSH_KEY="${SSH_KEY:-/Users/sergevilvovsky/Downloads/torontolinuxonesshkey_rsa.prv}"
SSH_HOST="${SSH_HOST:-root@163.66.89.80}"
# Speed up SSH (avoid GSSAPI/reverse-DNS delays)
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o GSSAPIAuthentication=no -o PreferredAuthentications=publickey"
REMOTE_DIR="${REMOTE_DIR:-/tmp/altastata-python-package}"
REMOTE_ALTASTATA_ACCOUNTS="${REMOTE_ALTASTATA_ACCOUNTS:-/root/.altastata/accounts}"
REMOTE_HPCS_DIR="${REMOTE_HPCS_DIR:-/home/jovyan/hpcs}"
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
  HPCS_BLOB_SRC="$HOME/.altastata/accounts/amazon.rsa.hpcs.serge678/hpcs-privkey.blob"
  if [ -f "$HPCS_BLOB_SRC" ]; then
    echo "Copying HPCS key blob to server ($REMOTE_HPCS_DIR/hpcs-privkey.blob)..."
    ssh $SSH_OPTS "$SSH_HOST" "mkdir -p $REMOTE_HPCS_DIR"
    rsync -avz -e "ssh $SSH_OPTS" "$HPCS_BLOB_SRC" "$SSH_HOST:$REMOTE_HPCS_DIR/hpcs-privkey.blob"
  fi
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

# zDNN / NNPA hardware acceleration is OFF by default (ENABLE_ZDNN=0); see
# Dockerfile.open_llm_s390x for the why. The Dockerfile still has a `COPY
# containers/rag-example/llama-3.2-1b-instruct-be.f16.gguf ...` line that runs
# unconditionally, so we always need a file at that path in the build context.
# When ENABLE_ZDNN=1 we stage the real ~2.5 GB F16 BE GGUF (Dockerfile keeps it,
# entrypoint can pick it). When ENABLE_ZDNN=0 we stage a 1-byte placeholder
# and the Dockerfile's next RUN deletes it from /opt/models.
ENABLE_ZDNN="${ENABLE_ZDNN:-0}"
F16_FILE="llama-3.2-1b-instruct-be.f16.gguf"
if [ "$ENABLE_ZDNN" = "1" ]; then
  echo "ENABLE_ZDNN=1: staging real $F16_FILE from /root/llama_models/ into build context..."
  ssh $SSH_OPTS "$SSH_HOST" "
    if [ ! -f /root/llama_models/$F16_FILE ]; then
      echo 'ERROR: /root/llama_models/$F16_FILE missing on server.'
      echo 'Build (once) on Mac and scp:'
      echo '  ./containers/rag-example/build-llama32-1b-f16-be-gguf.sh'
      echo '  scp ~/llama_models/$F16_FILE $SSH_HOST:/root/llama_models/'
      exit 2
    fi
    # Hard-link if same filesystem (fast, no extra disk); else copy. The file
    # is 2.5 GB so we want to avoid a full copy on every rebuild.
    ln -f /root/llama_models/$F16_FILE $REMOTE_DIR/containers/rag-example/$F16_FILE 2>/dev/null \
      || cp -f /root/llama_models/$F16_FILE $REMOTE_DIR/containers/rag-example/$F16_FILE
    ls -la $REMOTE_DIR/containers/rag-example/$F16_FILE
  "
else
  echo "ENABLE_ZDNN=0: staging tiny placeholder for $F16_FILE (Dockerfile will delete it from /opt/models)."
  ssh $SSH_OPTS "$SSH_HOST" "
    : > $REMOTE_DIR/containers/rag-example/$F16_FILE
    ls -la $REMOTE_DIR/containers/rag-example/$F16_FILE
  "
fi

# Tagging strategy (RAG_VERSION comes from version.sh, currently 2026g_latest):
#   ENABLE_ZDNN=0 (default, end-user image) -> :latest AND :$RAG_VERSION
#   ENABLE_ZDNN=1 (research-only, F16 + zDNN baked in) -> :${RAG_VERSION}_zdnn
# The zDNN variant must NEVER claim :latest, otherwise an end user pulling the
# default tag would get the heavier image with the known z17 quality regression.
source "$REPO_ROOT/version.sh"
if [ "$ENABLE_ZDNN" = "1" ]; then
  # Remote expands RAG_VERSION after sourcing version.sh; ${_zdnn} suffix is appended on the remote shell only.
  # Use \${VAR} below so TAG_ARGS is built locally with a literal dollar for SSH to expand.
  TAG_ARGS="-t altastata/rag-open-llm-s390x:\${RAG_VERSION}_zdnn"
  TAG_REPORT="altastata/rag-open-llm-s390x:${RAG_VERSION}_zdnn"
else
  TAG_ARGS="-t altastata/rag-open-llm-s390x:latest -t altastata/rag-open-llm-s390x:\${RAG_VERSION}"
  TAG_REPORT="altastata/rag-open-llm-s390x:${RAG_VERSION} and :latest"
fi
echo "Building image on server — tags=${TAG_REPORT} ENABLE_ZDNN=${ENABLE_ZDNN}"
ssh $SSH_OPTS "$SSH_HOST" "cd $REMOTE_DIR && source version.sh && docker build --build-arg ENABLE_ZDNN=$ENABLE_ZDNN -f containers/rag-example/Dockerfile.open_llm_s390x $TAG_ARGS ."

echo "Done. Image $TAG_REPORT is on the server."
echo "Accounts — if synced — are under $REMOTE_ALTASTATA_ACCOUNTS. Run with:"
echo "  docker run -p 8000:8000 -e ALTASTATA_ACCOUNT_DIR=$REMOTE_ALTASTATA_ACCOUNTS/amazon.rsa.bob123 -v $REMOTE_ALTASTATA_ACCOUNTS:$REMOTE_ALTASTATA_ACCOUNTS:ro ... $TAG_REPORT"
