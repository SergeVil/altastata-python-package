#!/usr/bin/env python3
"""
Open-source RAG query: vector search + AltaStata (chunk content) + LLM (MLX, Ollama, llama.cpp, Transformers, or watsonx).

Use LLM_PROVIDER=mlx for fastest on M1/Mac, LLM_PROVIDER=llama-cpp for fastest on s390x (CPU + GGUF),
LLM_PROVIDER=transformers for the slowest-but-most-portable PyTorch path.
Returns answer and source chunks for citations.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import (
    ALTASTATA_ACCOUNT_DIR,
    ALTASTATA_PASSWORD,
    ALTASTATA_ACCOUNT_ID,
    EMBEDDING_MODEL,
    LLM_PROVIDER,
    LOCAL_INDEX_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT,
    MLX_MODEL,
    MLX_MAX_TOKENS,
    LLAMA_CPP_MODEL_REPO,
    LLAMA_CPP_MODEL_FILE,
    LLAMA_CPP_MODEL_DIR,
    LLAMA_CPP_N_CTX,
    LLAMA_CPP_N_THREADS,
    LLAMA_CPP_MAX_TOKENS,
    HF_LLM_MODEL,
    HF_LLM_MAX_NEW_TOKENS,
    HF_LLM_FALLBACK_MODEL,
    WATSONX_URL,
    WATSONX_APIKEY,
    WATSONX_PROJECT_ID,
    WATSONX_MODEL_ID,
    WATSONX_MAX_TOKENS,
    RAG_DEBUG_RESPONSE,
    RAG_TIMING_LOG,
)
from indexer import get_vector_store as _get_vector_store_impl

# Cache so we don't reload embedding model / vector store / LLM on every request
_cached_embeddings = None
_cached_vector_store = None
_cached_vector_store_mtime = None  # invalidate when index is updated by background indexer
_cached_llm = None


class _TransformersChatWrapper:
    """Wraps HF pipeline to use chat template when available (fixes empty output for instruction models like SmolLM2)."""

    def __init__(self, pipe):
        self._pipe = pipe

    def invoke(self, prompt: str):
        tokenizer = getattr(self._pipe, "tokenizer", None)
        formatted = prompt
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if not isinstance(formatted, str):
                    formatted = prompt
            except Exception:  # pylint: disable=broad-except
                pass
        out = self._pipe(formatted)
        generated = out[0].get("generated_text", "") if out else ""
        if isinstance(formatted, str) and generated.startswith(formatted):
            content = generated[len(formatted) :].strip()
        else:
            content = generated.strip()
        return type("R", (), {"content": content})()


def _get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _cached_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _cached_embeddings


def _get_vector_store():
    global _cached_vector_store, _cached_vector_store_mtime
    index_file = os.path.join(LOCAL_INDEX_PATH, "embeddings.npy")
    if _cached_vector_store is not None and os.path.isfile(index_file):
        mtime = os.path.getmtime(index_file)
        if _cached_vector_store_mtime is not None and mtime > _cached_vector_store_mtime:
            _cached_vector_store = None
    if _cached_vector_store is None:
        _cached_vector_store = _get_vector_store_impl(_get_embeddings())
        _cached_vector_store_mtime = os.path.getmtime(index_file) if os.path.isfile(index_file) else None
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
            import torch
            device = 0 if torch.cuda.is_available() else -1  # CPU on Mac/390, GPU if present
            model_to_use = HF_LLM_MODEL
            # Avoid meta tensors: low_cpu_mem_usage can leave placeholders that cause "Tensor.item() cannot be called on meta tensors" during generation
            model_kwargs = {"low_cpu_mem_usage": False}
            for attempt in range(2):
                try:
                    pipe = pipeline(
                        "text-generation",
                        model=model_to_use,
                        max_new_tokens=HF_LLM_MAX_NEW_TOKENS,
                        temperature=0.1,
                        do_sample=True,
                        device=device,
                        model_kwargs=model_kwargs,
                    )
                    # Use chat template if available (SmolLM2 and other instruction models generate better)
                    _cached_llm = _TransformersChatWrapper(pipe)
                    if attempt > 0:
                        print(f"✅ Using fallback model: {model_to_use}")
                    break
                except Exception as e:  # pylint: disable=broad-except
                    if attempt == 0 and model_to_use != HF_LLM_FALLBACK_MODEL:
                        print(f"⚠️  {HF_LLM_MODEL} failed ({e}), trying fallback: {HF_LLM_FALLBACK_MODEL}")
                        model_to_use = HF_LLM_FALLBACK_MODEL
                    else:
                        raise
        elif LLM_PROVIDER in ("llama-cpp", "llama_cpp", "llamacpp"):
            try:
                import llama_cpp
            except ImportError as e:
                raise ImportError(
                    "LLM_PROVIDER=llama-cpp requires llama-cpp-python. "
                    "On s390x install with: "
                    "CMAKE_ARGS=\"-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS\" "
                    "pip install llama-cpp-python huggingface_hub"
                ) from e
            # Two ways to point at a GGUF:
            #   1. LLAMA_CPP_MODEL_REPO + LLAMA_CPP_MODEL_FILE -> hf_hub_download into LLAMA_CPP_MODEL_DIR.
            #   2. LLAMA_CPP_MODEL_REPO empty -> load LLAMA_CPP_MODEL_DIR/LLAMA_CPP_MODEL_FILE directly.
            #      Used by the s390x entrypoint when it auto-selects a self-converted F16 BE GGUF
            #      (not published on Hugging Face) for the zDNN/NNPA path on z16/z17.
            if LLAMA_CPP_MODEL_REPO and LLAMA_CPP_MODEL_REPO.strip():
                try:
                    from huggingface_hub import hf_hub_download
                except ImportError as e:
                    raise ImportError(
                        "LLAMA_CPP_MODEL_REPO is set but huggingface_hub is not installed. "
                        "Install: pip install huggingface_hub  (or unset LLAMA_CPP_MODEL_REPO and "
                        "mount the GGUF at LLAMA_CPP_MODEL_DIR/LLAMA_CPP_MODEL_FILE)."
                    ) from e
                print(
                    f"[llama-cpp] Resolving GGUF {LLAMA_CPP_MODEL_REPO}/{LLAMA_CPP_MODEL_FILE} "
                    f"(cache_dir={LLAMA_CPP_MODEL_DIR})..."
                )
                os.makedirs(LLAMA_CPP_MODEL_DIR, exist_ok=True)
                gguf_path = hf_hub_download(
                    repo_id=LLAMA_CPP_MODEL_REPO,
                    filename=LLAMA_CPP_MODEL_FILE,
                    cache_dir=LLAMA_CPP_MODEL_DIR,
                )
            else:
                gguf_path = os.path.join(LLAMA_CPP_MODEL_DIR, LLAMA_CPP_MODEL_FILE)
                if not os.path.isfile(gguf_path):
                    raise FileNotFoundError(
                        f"LLAMA_CPP_MODEL_REPO is empty so the GGUF must already exist at {gguf_path}, "
                        f"but no such file was found. Either set LLAMA_CPP_MODEL_REPO=<hf repo> or "
                        f"mount a directory containing {LLAMA_CPP_MODEL_FILE!r} at {LLAMA_CPP_MODEL_DIR}."
                    )
                print(f"[llama-cpp] Using local GGUF {gguf_path} (LLAMA_CPP_MODEL_REPO is empty)...")
            print(f"[llama-cpp] Loading {gguf_path} (n_ctx={LLAMA_CPP_N_CTX})...")
            # When RAG_TIMING_LOG=1, enable llama.cpp's own verbose perf output so
            # per-call `llama_perf_context_print` lines appear in the log
            # (prompt eval time / eval time / total time — i.e. prefill vs decode,
            # the canonical split for measuring NNPA / zDNN benefit).
            _llm_obj = llama_cpp.Llama(
                model_path=gguf_path,
                n_ctx=LLAMA_CPP_N_CTX,
                n_threads=(LLAMA_CPP_N_THREADS or None),
                verbose=bool(RAG_TIMING_LOG),
            )

            class _LlamaCppChat:
                def __init__(self, llm, max_tokens):
                    self._llm = llm
                    self._max_tokens = max_tokens
                    self.last_ttft_ms = None
                    self.last_decode_ms = None

                def invoke(self, prompt):
                    # Note: we deliberately send only a `user` message (no `system` role).
                    # The big-endian Llama-3.2 GGUF chat template segfaults on s390x when
                    # given a separate system message, so any guidance must live inside
                    # the user prompt itself (see _query_rag in this file).
                    messages = [{"role": "user", "content": prompt}]
                    self.last_ttft_ms = None
                    self.last_decode_ms = None
                    # Streaming path when timing is requested: measures TTFT (~= prefill cost)
                    # and decode throughput (decode tokens / decode time) at the Python layer,
                    # complementing llama.cpp's own llama_perf_context_print output.
                    if RAG_TIMING_LOG:
                        t_start = time.perf_counter()
                        t_first = None
                        chunks_text = []
                        n_chunks = 0
                        for chunk in self._llm.create_chat_completion(
                            messages=messages,
                            max_tokens=self._max_tokens,
                            temperature=0.1,
                            stream=True,
                        ):
                            delta = chunk["choices"][0].get("delta", {}) or {}
                            piece = delta.get("content")
                            if piece is None:
                                continue
                            if t_first is None:
                                t_first = time.perf_counter()
                            chunks_text.append(piece)
                            n_chunks += 1
                        t_end = time.perf_counter()
                        ttft_ms = ((t_first - t_start) * 1000) if t_first is not None else (t_end - t_start) * 1000
                        decode_ms = ((t_end - t_first) * 1000) if t_first is not None else 0.0
                        decode_chunks_after_first = max(n_chunks - 1, 0)
                        decode_cps = (
                            decode_chunks_after_first / ((t_end - t_first))
                            if (t_first is not None and decode_chunks_after_first > 0)
                            else 0.0
                        )
                        print(
                            "[RAG timing] llm_detail provider=llama-cpp"
                            f" ttft_ms={round(ttft_ms, 2)}"
                            f" decode_ms={round(decode_ms, 2)}"
                            f" output_chunks={n_chunks}"
                            f" decode_chunks_per_s={round(decode_cps, 2)}"
                        )
                        self.last_ttft_ms = round(ttft_ms, 2)
                        self.last_decode_ms = round(decode_ms, 2)
                        content = "".join(chunks_text)
                    else:
                        out = self._llm.create_chat_completion(
                            messages=messages,
                            max_tokens=self._max_tokens,
                            temperature=0.1,
                        )
                        content = out["choices"][0]["message"]["content"]
                    return type("R", (), {"content": content})()

            _cached_llm = _LlamaCppChat(_llm_obj, LLAMA_CPP_MAX_TOKENS)
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
    )
    if not ALTASTATA_USE_HPCS:
        af.set_password(ALTASTATA_PASSWORD)
    return create_filesystem(af, ALTASTATA_ACCOUNT_ID), af


def _print_rag_phase_table(total_wall_ms, main_rows, llm_cpp_subrows=None, info_rows=None):
    """Human-readable phase breakdown when RAG_TIMING_LOG=1 (after each timed query)."""
    if not RAG_TIMING_LOG:
        return
    denom = float(total_wall_ms) if total_wall_ms and total_wall_ms > 0 else 1.0
    w = 46
    print("[RAG timing] phase_table_ms")
    print(f"  {'phase':<{w}} {'ms':>11} {'pct':>7}")
    print(f"  {'-' * w} {'-' * 11} {'-' * 7}")
    summed = 0.0
    for label, ms in main_rows:
        ms = float(ms)
        summed += ms
        pct = 100.0 * ms / denom
        print(f"  {label:<{w}} {ms:>11.2f} {pct:>6.1f}%")
    if llm_cpp_subrows:
        for label, ms in llm_cpp_subrows:
            ms = float(ms)
            pct = 100.0 * ms / denom
            print(f"  {label:<{w}} {ms:>11.2f} {pct:>6.1f}%")
    if info_rows:
        note = "(reference - fetch_chunks sum may exceed parallel wall)"
        print(f"  {note:<{w}} {'':>11} {'':>7}")
        for label, ms in info_rows:
            ms = float(ms)
            pct = 100.0 * ms / denom
            print(f"  {label:<{w}} {ms:>11.2f} {pct:>6.1f}%")
    ov = float(total_wall_ms) - summed
    if ov > 0.5 or ov < -0.5:
        pct_ov = 100.0 * ov / denom
        print(f"  {'timing_residual_vs_total_ms':<{w}} {ov:>11.2f} {pct_ov:>6.1f}%")
    print(f"  {'END_TO_END_total_ms':<{w}} {float(total_wall_ms):>11.2f} {'100.0':>6}%")


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
    t_req = time.perf_counter()

    def _emit_timing(parts):
        """parts: list of (key, value) — stable column order in logs."""
        if not RAG_TIMING_LOG:
            return
        total_ms = round((time.perf_counter() - t_req) * 1000, 2)
        extra = " ".join(f"{name}={value}" for name, value in parts)
        print(f"[RAG timing] total_ms={total_ms} {extra}")

    t0 = time.perf_counter()
    vector_store = _get_vector_store()
    dt_vector_store = time.perf_counter() - t0
    if vector_store is None:
        _emit_timing(
            [
                ("vector_store_ms", round(dt_vector_store * 1000, 2)),
                ("note", "no_index"),
            ]
        )
        _tw = round((time.perf_counter() - t_req) * 1000, 2)
        _print_rag_phase_table(
            _tw,
            [("load_vector_store_index", round(dt_vector_store * 1000, 2))],
        )
        return "No index found. Run the indexer to index documents from AltaStata (or index_local.py for a local directory).", []

    t0 = time.perf_counter()
    llm = _get_llm()
    dt_llm_resolve = time.perf_counter() - t0

    # Similarity search (includes embed_query + numpy scores in SimpleVectorStore)
    t0 = time.perf_counter()
    results = vector_store.similarity_search_with_score(query_text, k=k)
    dt_similarity = time.perf_counter() - t0
    if not results:
        _emit_timing(
            [
                ("vector_store_ms", round(dt_vector_store * 1000, 2)),
                ("llm_resolve_ms", round(dt_llm_resolve * 1000, 2)),
                ("similarity_ms", round(dt_similarity * 1000, 2)),
                ("note", "no_hits"),
            ]
        )
        _tw = round((time.perf_counter() - t_req) * 1000, 2)
        _print_rag_phase_table(
            _tw,
            [
                ("load_vector_store_index", round(dt_vector_store * 1000, 2)),
                ("llm_resolve (cached handle)", round(dt_llm_resolve * 1000, 2)),
                ("vector_db_search (embed + similarity)", round(dt_similarity * 1000, 2)),
            ],
        )
        return "No relevant documents found.", []

    # Prefer lower distance (we use cosine; sort for consistency)
    results.sort(key=lambda x: x[1])
    top = results[: min(k, len(results))]

    fs = altastata_fs
    _af = altastata_af  # keep reference so connection is not gc'd
    t0 = time.perf_counter()
    if fs is None and ALTASTATA_ACCOUNT_DIR and os.path.isdir(ALTASTATA_ACCOUNT_DIR):
        fs, _af = _connect_altastata()
    dt_altastata_connect = time.perf_counter() - t0

    def fetch_one(idx, doc, _score):
        t_fetch = time.perf_counter()
        meta = doc.metadata
        chunk_path = meta.get("chunk_path") or meta.get("source")
        label = chunk_path or "(inline_doc)"
        if not chunk_path:
            text = doc.page_content
            src = {"filename": meta.get("filename", "unknown"), "text": doc.page_content[:200]}
        elif fs:
            try:
                with fs.open(chunk_path, "r") as f:
                    text = f.read()
            except Exception as e:  # pylint: disable=broad-except
                text = doc.page_content
                print(f"⚠️  Could not read {chunk_path}: {e}")
            src = {
                "filename": meta.get("filename", os.path.basename(chunk_path)),
                "chunk_path": chunk_path,
                "text": text[:300],
            }
        else:
            text = doc.page_content
            src = {
                "filename": meta.get("filename", os.path.basename(chunk_path)),
                "chunk_path": chunk_path,
                "text": text[:300],
            }
        fetch_ms = (time.perf_counter() - t_fetch) * 1000
        return idx, text, src, label, fetch_ms

    sources = []
    chunk_texts = []
    chunk_read_detail = []

    t_chunks = time.perf_counter()
    if len(top) <= 1:
        for i, (doc, sc) in enumerate(top):
            _, text, src, label, fetch_ms = fetch_one(i, doc, sc)
            chunk_texts.append(text)
            sources.append(src)
            chunk_read_detail.append((label, fetch_ms))
    else:
        results_by_idx = {}
        with ThreadPoolExecutor(max_workers=min(len(top), 8)) as executor:
            futures = {executor.submit(fetch_one, i, doc, sc): i for i, (doc, sc) in enumerate(top)}
            for fut in as_completed(futures):
                idx, text, src, label, fetch_ms = fut.result()
                results_by_idx[idx] = (text, src)
                chunk_read_detail.append((label, fetch_ms))
        for i in range(len(top)):
            text, src = results_by_idx[i]
            chunk_texts.append(text)
            sources.append(src)
    dt_chunk_phase = time.perf_counter() - t_chunks
    sum_chunk_ms = sum(ms for _lbl, ms in chunk_read_detail)

    t0 = time.perf_counter()
    context = "\n\n".join([f"[Excerpt {i+1}]:\n{t}" for i, t in enumerate(chunk_texts)])
    # Earlier prompt said "Answer in 2-4 short sentences" which fought list-style
    # questions ("what are the password requirements?"): small models resolved the
    # conflict by emitting just the lead-in ("The password requirements are as
    # follows:") and stopping. Allowing bullets when the answer is naturally a
    # list fixes the intermittent truncation; cap length so prose answers stay
    # short.
    prompt = f"""Answer the question using only the context below. Be concise (under 6 sentences). If the answer is naturally a list of items, return it as a bulleted list.

Context:
{context}

Question: {query_text}

Answer:"""

    dt_prompt_build = time.perf_counter() - t0

    t_llm = time.perf_counter()
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
        raw_answer = answer
        # Pipeline often returns prompt + generated; keep only the generated part after "Answer:"
        if "Context:" in answer and "Question:" in answer:
            answer = answer.split("Answer:", 1)[-1].strip()
        # If that left nothing, use whatever comes after the prompt (model may not repeat "Answer:")
        if not answer or not answer.strip():
            if raw_answer.startswith(prompt):
                answer = raw_answer[len(prompt):].strip()
            elif len(prompt) < len(raw_answer):
                answer = raw_answer[len(prompt):].strip()
        if not answer or not answer.strip():
            if RAG_DEBUG_RESPONSE:
                print(f"[RAG debug] prompt len={len(prompt)}, raw_answer len={len(raw_answer)}")
                print(f"[RAG debug] raw_answer start: {repr(raw_answer[:400])}")
                print(f"[RAG debug] raw_answer end: {repr(raw_answer[-300]) if len(raw_answer) > 300 else ''}")
            answer = "No text was generated. Try a shorter question, or increase HF_LLM_MAX_NEW_TOKENS (default 400) if you need longer answers."
    except Exception as e:  # pylint: disable=broad-except
        answer = f"LLM error: {e}"
    dt_llm_invoke = time.perf_counter() - t_llm

    reads_bits = []
    for lbl, ms in chunk_read_detail:
        short = lbl if lbl == "(inline_doc)" else os.path.basename(lbl)
        reads_bits.append(f"{short}:{round(ms, 2)}ms")
    reads_joined = ",".join(reads_bits)

    _emit_timing(
        [
            ("vector_store_ms", round(dt_vector_store * 1000, 2)),
            ("llm_resolve_ms", round(dt_llm_resolve * 1000, 2)),
            ("similarity_ms", round(dt_similarity * 1000, 2)),
            ("altastata_connect_ms", round(dt_altastata_connect * 1000, 2)),
            ("chunk_phase_wall_ms", round(dt_chunk_phase * 1000, 2)),
            ("chunk_fetch_sum_ms", round(sum_chunk_ms, 2)),
            ("chunk_reads", reads_joined),
            ("prompt_build_ms", round(dt_prompt_build * 1000, 2)),
            ("llm_invoke_ms", round(dt_llm_invoke * 1000, 2)),
            ("prompt_chars", len(prompt)),
        ]
    )

    total_wall_ms = round((time.perf_counter() - t_req) * 1000, 2)
    llm_sub = []
    if getattr(llm, "last_ttft_ms", None) is not None:
        llm_sub.append(("  llama-cpp TTFT (~prefill+1st token)", llm.last_ttft_ms))
    if getattr(llm, "last_decode_ms", None) is not None:
        llm_sub.append(("  llama-cpp decode (streaming)", llm.last_decode_ms))
    _print_rag_phase_table(
        total_wall_ms,
        [
            ("load_vector_store_index", round(dt_vector_store * 1000, 2)),
            ("llm_resolve (cached handle)", round(dt_llm_resolve * 1000, 2)),
            ("vector_db_search (embed + similarity)", round(dt_similarity * 1000, 2)),
            ("altastata_connect", round(dt_altastata_connect * 1000, 2)),
            ("fetch_chunks (parallel wall)", round(dt_chunk_phase * 1000, 2)),
            ("build_prompt", round(dt_prompt_build * 1000, 2)),
            ("llm_inference", round(dt_llm_invoke * 1000, 2)),
        ],
        llm_cpp_subrows=llm_sub or None,
        info_rows=[("fetch_chunks_sum_threads (reference)", round(sum_chunk_ms, 2))],
    )

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
