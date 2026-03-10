#!/usr/bin/env python3
"""
RAG web interface: HTTPS-capable FastAPI app.

- GET /  -> HTML form to ask questions
- POST /query -> JSON { "query": "..." } -> { "answer": "...", "sources": [...] }
"""

import asyncio
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import (
    ALTASTATA_ACCOUNT_DIR,
    ALTASTATA_USE_HPCS,
    WEB_HOST,
    WEB_PORT,
    SSL_CERT_FILE,
    SSL_KEY_FILE,
    QUERY_TIMEOUT,
)
from query_rag import query_rag

app = FastAPI(title="AltaStata RAG (Open LLM)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Optional: reuse AltaStata connection across requests (set at startup)
_altastata_fs = None
_altastata_af = None


@app.on_event("startup")
def startup():
    global _altastata_fs, _altastata_af
    if ALTASTATA_ACCOUNT_DIR and os.path.isdir(ALTASTATA_ACCOUNT_DIR):
        try:
            from altastata.altastata_functions import AltaStataFunctions
            from altastata.fsspec import create_filesystem
            from config import ALTASTATA_PASSWORD, ALTASTATA_ACCOUNT_ID
            _altastata_af = AltaStataFunctions.from_account_dir(
                ALTASTATA_ACCOUNT_DIR,
                callback_server_port=0,
                enable_callback_server=False,
            )
            if not ALTASTATA_USE_HPCS:
                _altastata_af.set_password(ALTASTATA_PASSWORD)
            _altastata_fs = create_filesystem(_altastata_af, ALTASTATA_ACCOUNT_ID)
        except Exception as e:
            print(f"⚠️  AltaStata startup: {e}")

    # Warmup in background: load embeddings, vector store, and LLM. Skip a full LLM invoke for
    # Transformers (CPU inference can take minutes); first user query will do the first generation.
    def _warmup():
        try:
            from query_rag import _get_vector_store, _get_llm
            from config import LLM_PROVIDER
            print("Warming up (embeddings + vector store + LLM)...")
            _get_vector_store()
            llm = _get_llm()
            if LLM_PROVIDER != "transformers":
                # Ollama/MLX are fast; one short invoke warms the pipe
                llm.invoke("Reply with one word: OK")
            print("✅ RAG warmup done")
        except Exception as e:
            print(f"⚠️  Warmup skipped: {e}")
    import threading
    threading.Thread(target=_warmup, daemon=True).start()


class QueryRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/debug")
async def debug():
    """Check if RAG caches are warm (embeddings, vector store, Ollama). Rebuild app image to get caching."""
    try:
        from query_rag import _cached_embeddings, _cached_vector_store, _cached_llm
        return {
            "cache": {
                "embeddings": _cached_embeddings is not None,
                "vector_store": _cached_vector_store is not None,
                "llm": _cached_llm is not None,
            },
            "all_warm": _cached_embeddings is not None and _cached_vector_store is not None and _cached_llm is not None,
        }
    except Exception as e:
        return {"error": str(e), "cache": {}}


@app.post("/query")
async def api_query(body: QueryRequest):
    if not body.query or not body.query.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "query is required"},
        )
    try:
        loop = asyncio.get_event_loop()
        answer, sources = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: query_rag(
                    body.query.strip(),
                    altastata_fs=_altastata_fs,
                    altastata_af=_altastata_af,
                ),
            ),
            timeout=float(QUERY_TIMEOUT),
        )
        return {
            "answer": answer,
            "sources": [
                {"filename": s.get("filename", "?"), "preview": s.get("text", "")[:200]}
                for s in sources
            ],
        }
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "error": f"Query timed out after {QUERY_TIMEOUT}s. Try a shorter question or increase QUERY_TIMEOUT in .env (Transformers on CPU can be slow).",
                "answer": "",
                "sources": [],
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "answer": "", "sources": []},
        )


def main():
    import uvicorn
    use_ssl = SSL_CERT_FILE and SSL_KEY_FILE and os.path.isfile(SSL_CERT_FILE) and os.path.isfile(SSL_KEY_FILE)
    uvicorn.run(
        "web_app:app",
        host=WEB_HOST,
        port=WEB_PORT,
        reload=False,
        ssl_keyfile=SSL_KEY_FILE if use_ssl else None,
        ssl_certfile=SSL_CERT_FILE if use_ssl else None,
    )


if __name__ == "__main__":
    main()
