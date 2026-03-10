#!/usr/bin/env python3
"""
Open-source RAG query: vector search + AltaStata (chunk content) + LLM (MLX, Ollama, or Hugging Face Transformers).

Use LLM_PROVIDER=mlx for fastest on M1/Mac; LLM_PROVIDER=transformers for same stack on Mac and IBM 390.
Returns answer and source chunks for citations.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import (
    ALTASTATA_ACCOUNT_DIR,
    ALTASTATA_PASSWORD,
    ALTASTATA_ACCOUNT_ID,
    EMBEDDING_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT,
    MLX_MODEL,
    MLX_MAX_TOKENS,
    HF_LLM_MODEL,
    HF_LLM_MAX_NEW_TOKENS,
    HF_LLM_FALLBACK_MODEL,
    WATSONX_URL,
    WATSONX_APIKEY,
    WATSONX_PROJECT_ID,
    WATSONX_MODEL_ID,
    WATSONX_MAX_TOKENS,
)
from indexer import get_vector_store as _get_vector_store_impl

# Cache so we don't reload embedding model / vector store / LLM on every request
_cached_embeddings = None
_cached_vector_store = None
_cached_llm = None


def _get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _cached_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _cached_embeddings


def _get_vector_store():
    global _cached_vector_store
    if _cached_vector_store is None:
        _cached_vector_store = _get_vector_store_impl(_get_embeddings())
    return _cached_vector_store


def _get_llm():
    global _cached_llm
    if _cached_llm is None:
        if LLM_PROVIDER == "mlx":
            try:
                from mlx_lm import load, generate
            except ImportError as e:
                raise ImportError(
                    "LLM_PROVIDER=mlx requires mlx and mlx-lm (Mac only). "
                    "Install: pip install mlx mlx-lm. In Docker use LLM_PROVIDER=transformers or ollama."
                ) from e
            model, tokenizer = load(MLX_MODEL)

            class _MLXChat:
                def __init__(self, model, tokenizer, max_tokens):
                    self._model = model
                    self._tokenizer = tokenizer
                    self._max_tokens = max_tokens

                def invoke(self, prompt):
                    messages = [{"role": "user", "content": prompt}]
                    try:
                        prompt_str = self._tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False
                        )
                    except Exception:
                        prompt_str = prompt
                    full = generate(
                        self._model,
                        self._tokenizer,
                        prompt=prompt_str,
                        max_tokens=self._max_tokens,
                    )
                    content = (
                        full[len(prompt_str) :].strip()
                        if full.startswith(prompt_str)
                        else full
                    )
                    return type("R", (), {"content": content})()

            _cached_llm = _MLXChat(model, tokenizer, MLX_MAX_TOKENS)
        elif LLM_PROVIDER == "transformers":
            from transformers import pipeline
            from langchain_community.llms import HuggingFacePipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1  # CPU on Mac/390, GPU if present
            model_to_use = HF_LLM_MODEL
            for attempt in range(2):
                try:
                    pipe = pipeline(
                        "text-generation",
                        model=model_to_use,
                        max_new_tokens=HF_LLM_MAX_NEW_TOKENS,
                        temperature=0.1,
                        do_sample=True,
                        device=device,
                    )
                    _cached_llm = HuggingFacePipeline(pipeline=pipe)
                    if attempt > 0:
                        print(f"✅ Using fallback model: {model_to_use}")
                    break
                except Exception as e:  # pylint: disable=broad-except
                    if attempt == 0 and model_to_use != HF_LLM_FALLBACK_MODEL:
                        print(f"⚠️  {HF_LLM_MODEL} failed ({e}), trying fallback: {HF_LLM_FALLBACK_MODEL}")
                        model_to_use = HF_LLM_FALLBACK_MODEL
                    else:
                        raise
        elif LLM_PROVIDER == "watsonx":
            if not WATSONX_APIKEY or not WATSONX_PROJECT_ID:
                raise ValueError(
                    "LLM_PROVIDER=watsonx requires WATSONX_APIKEY and WATSONX_PROJECT_ID. "
                    "Set them in .env or see https://cloud.ibm.com/docs/account?topic=account-userapikey"
                )
            try:
                from langchain_ibm import ChatWatsonx
            except ImportError as e:
                raise ImportError(
                    "LLM_PROVIDER=watsonx requires langchain-ibm. Install: pip install langchain-ibm"
                ) from e
            _cached_llm = ChatWatsonx(
                model_id=WATSONX_MODEL_ID,
                url=WATSONX_URL,
                project_id=WATSONX_PROJECT_ID,
                params={"temperature": 0.1, "max_tokens": WATSONX_MAX_TOKENS},
            )
            # ChatWatsonx.invoke expects messages; we pass a single user message
            _original_invoke = _cached_llm.invoke
            def _invoke_prompt(prompt):
                return _original_invoke([("human", prompt)])
            _cached_llm.invoke = _invoke_prompt
        else:
            from langchain_ollama import ChatOllama
            _cached_llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0.1,
                num_predict=OLLAMA_NUM_PREDICT,
            )
    return _cached_llm


def _connect_altastata():
    """AltaStata + fsspec for reading chunks (no callback server needed for query-only)."""
    from altastata.altastata_functions import AltaStataFunctions
    from altastata.fsspec import create_filesystem
    from config import ALTASTATA_USE_HPCS
    af = AltaStataFunctions.from_account_dir(
        ALTASTATA_ACCOUNT_DIR,
        callback_server_port=0,
        enable_callback_server=False,
    )
    if not ALTASTATA_USE_HPCS:
        af.set_password(ALTASTATA_PASSWORD)
    return create_filesystem(af, ALTASTATA_ACCOUNT_ID), af


def query_rag(
    query_text: str,
    k: int = 2,
    altastata_fs=None,
    altastata_af=None,
):
    """
    Run RAG: embed query -> similarity search -> read chunks from AltaStata -> Ollama.

    Returns (answer: str, sources: list[dict]).
    If altastata_fs/altastata_af are None and ALTASTATA_ACCOUNT_DIR is set, they are created.
    """
    vector_store = _get_vector_store()
    if vector_store is None:
        return "No index found. Run the indexer to index documents from AltaStata (or index_local.py for a local directory).", []

    llm = _get_llm()

    # Similarity search
    results = vector_store.similarity_search_with_score(query_text, k=k)
    if not results:
        return "No relevant documents found.", []

    # Prefer lower distance (we use cosine; sort for consistency)
    results.sort(key=lambda x: x[1])
    top = results[: min(k, len(results))]

    fs = altastata_fs
    _af = altastata_af  # keep reference so connection is not gc'd
    if fs is None and ALTASTATA_ACCOUNT_DIR and os.path.isdir(ALTASTATA_ACCOUNT_DIR):
        fs, _af = _connect_altastata()

    sources = []
    chunk_texts = []
    for doc, _score in top:
        meta = doc.metadata
        chunk_path = meta.get("chunk_path") or meta.get("source")
        if not chunk_path:
            chunk_texts.append(doc.page_content)
            sources.append({"filename": meta.get("filename", "unknown"), "text": doc.page_content[:200]})
            continue
        if fs:
            try:
                with fs.open(chunk_path, "r") as f:
                    text = f.read()
            except Exception as e:  # pylint: disable=broad-except
                text = doc.page_content
                print(f"⚠️  Could not read {chunk_path}: {e}")
        else:
            text = doc.page_content
        chunk_texts.append(text)
        sources.append({
            "filename": meta.get("filename", os.path.basename(chunk_path)),
            "chunk_path": chunk_path,
            "text": text[:300],
        })

    context = "\n\n".join([f"[Excerpt {i+1}]:\n{t}" for i, t in enumerate(chunk_texts)])
    prompt = f"""Answer in 2-4 short sentences. Use only the context below. If the context doesn't have the answer, say "Not in the documents."

Context:
{context}

Question: {query_text}

Answer:"""

    try:
        response = llm.invoke(prompt)
        # Handle different return types: ChatMessage .content, string, or raw pipeline list/dict
        if hasattr(response, "content"):
            answer = response.content
        elif isinstance(response, list) and response and isinstance(response[0], dict):
            answer = response[0].get("generated_text", str(response))
        elif isinstance(response, dict):
            answer = response.get("generated_text", str(response))
        else:
            answer = str(response)
        # Pipeline often returns prompt + generated; keep only the generated part after "Answer:"
        if "Context:" in answer and "Question:" in answer:
            answer = answer.split("Answer:", 1)[-1].strip()
    except Exception as e:  # pylint: disable=broad-except
        answer = f"LLM error: {e}"
    return answer.strip(), sources


def main():
    """CLI: interactive or single query."""
    if not ALTASTATA_ACCOUNT_DIR or not os.path.isdir(ALTASTATA_ACCOUNT_DIR):
        print("Set ALTASTATA_ACCOUNT_DIR to your AltaStata account directory.")
        sys.exit(1)

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="*", help="Question (or run without args for interactive)")
    args = p.parse_args()

    if args.query:
        q = " ".join(args.query)
        answer, sources = query_rag(q)
        print("\n🤖 Answer:", answer)
        print("\n📚 Sources:", len(sources))
        for i, s in enumerate(sources, 1):
            print(f"  {i}. {s.get('filename', '?')}")
    else:
        print("💡 Ask questions (quit to exit)\n")
        while True:
            q = input("❓ Question: ").strip()
            if not q or q.lower() in ("quit", "exit", "q"):
                break
            answer, sources = query_rag(q)
            print("\n🤖", answer)
            print("📚 Sources:", [s.get("filename", "?") for s in sources])
            print()


if __name__ == "__main__":
    main()
