"""Configuration for open-source RAG (simple vector store + Ollama + AltaStata)."""
import os
from pathlib import Path

# Load .env from this package directory when running locally (no Docker)
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

# AltaStata
ALTASTATA_ACCOUNT_DIR = os.getenv("ALTASTATA_ACCOUNT_DIR", "")
_raw_password = os.getenv("ALTASTATA_PASSWORD", "123")
# AltaStata Java setPassword does not accept null/empty; use default if missing or blank
ALTASTATA_PASSWORD = _raw_password if (_raw_password and str(_raw_password).strip()) else "123"
# HPCS (IBM Hyper Protect) accounts use key blobs/GREP11, not a password; skip set_password when set
ALTASTATA_USE_HPCS = os.getenv("ALTASTATA_USE_HPCS", "").strip().lower() in ("1", "true", "yes")
ALTASTATA_ACCOUNT_ID = os.getenv("ALTASTATA_ACCOUNT_ID", "bob123")
ALTASTATA_CALLBACK_PORT = int(os.getenv("ALTASTATA_CALLBACK_PORT", "25334"))

# Vector store: simple (pure Python + numpy, s390x-safe)
LOCAL_INDEX_PATH = os.getenv("LOCAL_INDEX_PATH", os.path.join(os.path.dirname(__file__), "local_index"))

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM: "mlx" (fastest on M1/Mac), "ollama" (fast on Mac with Metal), "transformers" (Mac + IBM 390, slow on CPU), or "watsonx" (IBM Cloud / Granite API).
# If unset, use ollama when Ollama is reachable, else transformers.
def _default_llm_provider():
    env_val = os.getenv("LLM_PROVIDER", "").strip().lower()
    if env_val:
        return env_val
    try:
        import urllib.request
        u = urllib.request.urlopen(
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/tags", timeout=2
        )
        u.close()
        return "ollama"
    except Exception:
        return "transformers"


LLM_PROVIDER = _default_llm_provider()

# Ollama (when LLM_PROVIDER=ollama). Use small models for fast response: smollm2:360m, qwen2.5:0.5b, llama3.2:1b
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smollm2:360m")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "128"))

# MLX (when LLM_PROVIDER=mlx). Fastest on Apple Silicon; Mac-only. Small 4-bit models: Llama-3.2-1B-Instruct-4bit, etc.
MLX_MODEL = os.getenv("MLX_MODEL", "mlx-community/Llama-3.2-1B-Instruct-4bit")
MLX_MAX_TOKENS = int(os.getenv("MLX_MAX_TOKENS", "128"))

# Hugging Face Transformers (when LLM_PROVIDER=transformers). Same on Mac, Docker, and IBM 390. TinyLlama is instruction-tuned; gpt2 for smaller/faster.
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
HF_LLM_MAX_NEW_TOKENS = int(os.getenv("HF_LLM_MAX_NEW_TOKENS", "80"))
# If HF_LLM_MODEL fails to load (gated, missing, etc.), use this fallback. Should be small and ungated.
HF_LLM_FALLBACK_MODEL = os.getenv("HF_LLM_FALLBACK_MODEL", "gpt2")

# IBM watsonx (when LLM_PROVIDER=watsonx). Granite and other models via IBM Cloud API. Fast from Mac, Docker, 390.
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY", os.getenv("WATSONX_API_KEY", ""))
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2")
WATSONX_MAX_TOKENS = int(os.getenv("WATSONX_MAX_TOKENS", "128"))

# Chunking (same as Vertex example)
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "4000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "800"))

# Indexer: path to index on startup before listening for events (e.g. RAGDocs/policies). Empty = only listen.
RAG_INDEX_PATH = os.getenv("RAG_INDEX_PATH", "RAGDocs/policies")

# Web app
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", "")
SSL_KEY_FILE = os.getenv("SSL_KEY_FILE", "")
# Max seconds for a single RAG query (Transformers on CPU needs more; Ollama is faster)
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "180"))
