#!/bin/sh
set -e
cd /app/open_llm

# Application initialization: build index from AltaStata if no index exists yet
if [ -n "$ALTASTATA_ACCOUNT_DIR" ] && [ ! -f local_index/embeddings.npy ]; then
  RAG_PATH="${RAG_INDEX_PATH:-RAGDocs/policies}"
  if [ -n "$RAG_PATH" ]; then
    echo "Building index from AltaStata path: $RAG_PATH"
    python -m indexer --once "$RAG_PATH" || true
  fi
fi

exec "$@"
