# AltaStata RAG – Open LLM stack (simple vector store + Ollama + AltaStata)
# Build from repo root: docker build -f rag-example/open_llm/Dockerfile .
# Used by docker-compose (app + indexer). Includes sample_documents so you can run index_local.py for a quick demo without AltaStata.
# For AltaStata-only Mac (same layout as s390x): use openshift/rag/Dockerfile.open_llm_mac and build-and-run-rag-mac.sh.

FROM python:3.11-slim

WORKDIR /app

# System deps: build-essential for sentence-transformers, default-jre for AltaStata (Java gateway)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Open RAG deps first (cached unless requirements.txt changes; avoids rebuilding scikit-learn etc. on every code change)
COPY rag-example/open_llm/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install AltaStata from repo copy only (do not also pip install altastata from PyPI – would conflict)
COPY . /src/altastata-python-package
RUN pip install --no-cache-dir /src/altastata-python-package

# App code + sample documents (for indexing in container)
COPY rag-example/open_llm /app/open_llm
COPY rag-example/sample_documents /app/sample_documents
ENV PYTHONPATH=/app/open_llm:/src/altastata-python-package

VOLUME /app/open_llm/local_index
RUN chmod +x /app/open_llm/docker-entrypoint.sh

WORKDIR /app/open_llm
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8000
ENV PYTHONPATH=/app/open_llm
ENV VECTOR_STORE=simple
ENV LOCAL_INDEX_PATH=/app/open_llm/local_index
EXPOSE 8000

ENTRYPOINT ["/app/open_llm/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
