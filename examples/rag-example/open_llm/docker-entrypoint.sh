#!/bin/sh
set -e
cd /app/open_llm

echo "[entrypoint] WORKDIR=$(pwd), LOCAL_INDEX_PATH=${LOCAL_INDEX_PATH:-/app/open_llm/local_index}"
echo "[entrypoint] Index volume: embeddings.npy exists=$([ -f local_index/embeddings.npy ] && echo yes || echo no), docs.json exists=$([ -f local_index/docs.json ] && echo yes || echo no)"

# Log env (mask password)
if [ -n "$ALTASTATA_ACCOUNT_DIR" ]; then
  echo "[entrypoint] ALTASTATA_ACCOUNT_DIR=$ALTASTATA_ACCOUNT_DIR"
  echo "[entrypoint] ALTASTATA_ACCOUNT_ID=${ALTASTATA_ACCOUNT_ID:-bob123}, ALTASTATA_PASSWORD=${ALTASTATA_PASSWORD:+<set>}"
  echo "[entrypoint] RAG_INDEX_PATH=${RAG_INDEX_PATH:-RAGDocs/policies}, ALTASTATA_USE_HPCS=${ALTASTATA_USE_HPCS:-}"
fi

# Verify AltaStata account directory when set (helps debug "No index found" / connection issues)
if [ -n "$ALTASTATA_ACCOUNT_DIR" ]; then
  if [ ! -d "$ALTASTATA_ACCOUNT_DIR" ]; then
    echo "[entrypoint] WARNING: ALTASTATA_ACCOUNT_DIR is set to '$ALTASTATA_ACCOUNT_DIR' but it is not a directory in this container."
    echo "[entrypoint] Check that the host account path is mounted correctly (e.g. -v \$(dirname \$ALTASTATA_ACCOUNT_DIR):/altastata_account:ro and ALTASTATA_ACCOUNT_DIR=/altastata_account/\$(basename ...))."
  else
    echo "[entrypoint] AltaStata account dir OK: $ALTASTATA_ACCOUNT_DIR"
    echo "[entrypoint] Account dir contents:"
    ls -la "$ALTASTATA_ACCOUNT_DIR" 2>/dev/null || echo "[entrypoint] (list failed)"
  fi
fi

# Application initialization: build index from AltaStata if no index exists yet
if [ -n "$ALTASTATA_ACCOUNT_DIR" ] && [ -d "$ALTASTATA_ACCOUNT_DIR" ] && [ ! -f local_index/embeddings.npy ]; then
  RAG_PATH="${RAG_INDEX_PATH:-RAGDocs/policies}"
  if [ -n "$RAG_PATH" ]; then
    echo "[entrypoint] No index found. Building index from AltaStata path: $RAG_PATH"
    if python -m indexer --once "$RAG_PATH"; then
      echo "[entrypoint] Index built successfully."
      echo "[entrypoint] Index volume after build: embeddings.npy exists=$([ -f local_index/embeddings.npy ] && echo yes || echo no)"
    else
      echo "[entrypoint] Indexer failed (exit $?). Common causes: AltaStata connection (password, Java gateway, or no documents at $RAG_PATH)."
      echo "[entrypoint] To retry later: docker exec <container> python -m indexer --once $RAG_PATH"
    fi
  else
    echo "[entrypoint] RAG_INDEX_PATH empty; skipping index build."
  fi
else
  if [ -f local_index/embeddings.npy ]; then
    echo "[entrypoint] Using existing index (local_index/embeddings.npy present)."
  elif [ -z "$ALTASTATA_ACCOUNT_DIR" ]; then
    echo "[entrypoint] ALTASTATA_ACCOUNT_DIR not set; skipping index build."
  elif [ ! -d "$ALTASTATA_ACCOUNT_DIR" ]; then
    echo "[entrypoint] AltaStata account dir missing; skipping index build."
  fi
fi

# Start indexer in background: uses existing index or bootstraps if none, then listens for SHARE events
if [ -n "$ALTASTATA_ACCOUNT_DIR" ] && [ -d "$ALTASTATA_ACCOUNT_DIR" ]; then
  echo "[entrypoint] Starting indexer in background (index existing + listen for new shared docs)."
  python -m indexer &
fi

# Optional Console UI on :9877. See mycloud/altastata-grpc/TLS_DESIGN.md.
if [ "${ENABLE_ALTASTATA_CONSOLE_UI:-0}" = "1" ]; then
  if [ -n "$ALTASTATA_ACCOUNT_DIR" ] && [ -d "$ALTASTATA_ACCOUNT_DIR" ]; then
    echo "[entrypoint] Starting altastata-grpc-server on :9877."
    altastata-grpc-server > /tmp/altastata-grpc-server.log 2>&1 &
  else
    echo "[entrypoint] ENABLE_ALTASTATA_CONSOLE_UI=1 but no ALTASTATA_ACCOUNT_DIR; skipping."
  fi
fi

# Auto-select GGUF for llama-cpp based on s390x hardware capability:
#   * z16 / z17 (Telum I/II) advertise the 'nnpa' facility in /proc/cpuinfo and
#     can run llama.cpp's zDNN backend, which only accelerates F32/F16/BF16
#     tensors. We prefer a self-converted F16 BE GGUF so zDNN actually
#     accelerates work at runtime.
#   * z15 and earlier (no NNPA) — or any host where no F16 file is reachable —
#     fall back to the smaller Q5_K_S quantization, which runs faster on plain
#     CPU/VXE2 than F16 would.
# Two F16 sources are checked, in priority order:
#   1. /models/<F16_FILE>  — user-supplied via -v <host_dir>:/models
#   2. /opt/models/<F16_FILE> — baked into the image at build time (default;
#      see Dockerfile.open_llm_s390x), so a plain `docker run` works in
#      confidential / air-gapped envs with no host-side setup.
# User overrides win: any explicit LLAMA_CPP_MODEL_FILE or LLAMA_CPP_MODEL_REPO
# from `docker run -e ...` is left untouched.
#
# When the image is built with ENABLE_ZDNN=0 (current default — see
# Dockerfile.open_llm_s390x for why), llama-cpp-python has no zDNN backend, the
# F16 GGUF is not baked in, and even if a user mounts one over /models it would
# run slower than Q5_K_S on plain CPU. So we short-circuit: always pick Q5_K_S.
if [ "${LLM_PROVIDER:-llama-cpp}" = "llama-cpp" ] && [ -z "${LLAMA_CPP_MODEL_FILE:-}" ]; then
  _F16_FILE="llama-3.2-1b-instruct-be.f16.gguf"
  _Q5_FILE="llama-3.2-1b-instruct-be.Q5_K_S.gguf"
  _USER_MODELS_DIR="${LLAMA_CPP_MODEL_DIR:-/models}"
  _BAKED_MODELS_DIR="/opt/models"

  if [ "${ENABLE_ZDNN:-0}" != "1" ]; then
    export LLAMA_CPP_MODEL_FILE="$_Q5_FILE"
    echo "[entrypoint] ENABLE_ZDNN=0 (image built without NNPA acceleration); using $_Q5_FILE on CPU/VXE2 regardless of /proc/cpuinfo."
  else
    _F16_PATH=""
    if [ -f "$_USER_MODELS_DIR/$_F16_FILE" ]; then
      _F16_PATH="$_USER_MODELS_DIR/$_F16_FILE"
      _F16_SRC="user-supplied $_USER_MODELS_DIR"
    elif [ -f "$_BAKED_MODELS_DIR/$_F16_FILE" ]; then
      _F16_PATH="$_BAKED_MODELS_DIR/$_F16_FILE"
      _F16_SRC="baked-in $_BAKED_MODELS_DIR"
    fi

    if grep -qE '^features[[:space:]]*:.*\bnnpa\b' /proc/cpuinfo 2>/dev/null \
       && [ -n "$_F16_PATH" ]; then
      export LLAMA_CPP_MODEL_FILE="$_F16_FILE"
      export LLAMA_CPP_MODEL_REPO=""
      export LLAMA_CPP_MODEL_DIR="$(dirname "$_F16_PATH")"
      echo "[entrypoint] ENABLE_ZDNN=1 + NNPA hardware + F16 BE GGUF ($_F16_SRC) -> selecting F16 BE for zDNN acceleration"
    else
      export LLAMA_CPP_MODEL_FILE="$_Q5_FILE"
      if grep -qE '^features[[:space:]]*:.*\bnnpa\b' /proc/cpuinfo 2>/dev/null; then
        echo "[entrypoint] ENABLE_ZDNN=1 + NNPA hardware detected but no $_F16_FILE in $_USER_MODELS_DIR or $_BAKED_MODELS_DIR; using $_Q5_FILE (no zDNN acceleration)."
      else
        echo "[entrypoint] ENABLE_ZDNN=1 but no NNPA in /proc/cpuinfo; using $_Q5_FILE (CPU path)."
      fi
    fi
  fi
fi

echo "[entrypoint] Starting web server on port ${WEB_PORT:-8000}..."
exec "$@"
