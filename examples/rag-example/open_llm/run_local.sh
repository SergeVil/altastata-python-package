#!/usr/bin/env bash
# Run the RAG web app locally (no Docker). Ollama must be running on the host (e.g. ollama serve).
# From repo root:  pip install -e . && cd examples/rag-example/open_llm && pip install -r requirements.txt
# Then:  cd examples/rag-example/open_llm && ./run_local.sh

set -e
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "No .env found. Copy .env.example to .env and set ALTASTATA_ACCOUNT_DIR."
  exit 1
fi

# Load .env for this shell (optional; web_app reads os.environ which may already have .env from the IDE)
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

echo "Starting RAG web app at http://localhost:8000 (no Docker)"
echo "Ensure Ollama is running: ollama serve && ollama pull llama3.2:1b"
echo ""
exec python web_app.py
