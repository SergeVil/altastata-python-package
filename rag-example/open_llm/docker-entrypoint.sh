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

echo "[entrypoint] Starting web server on port ${WEB_PORT:-8000}..."
exec "$@"
