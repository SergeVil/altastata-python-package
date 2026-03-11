"""
Pure-Python vector store (numpy only). No Chroma, no FAISS.
Runs on any architecture (x86, ARM, s390x). Use for the demo.
"""
from pathlib import Path
import json
import numpy as np
from langchain_core.documents import Document


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    if n <= 0:
        return x
    return x / n


class SimpleVectorStore:
    """In-memory vector store with save/load to disk. Cosine similarity via numpy."""

    def __init__(self, embeddings: np.ndarray, docs: list[dict], embedding_fn):
        # embeddings: (n_docs, dim), normalized
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        if self._embeddings.ndim == 1:
            self._embeddings = self._embeddings.reshape(1, -1)
        self._docs = list(docs)  # [{"content": str, "metadata": dict}, ...]
        self._embedding_fn = embedding_fn
        assert len(self._docs) == len(self._embeddings), "docs and embeddings length mismatch"

    def similarity_search_with_score(self, query: str, k: int = 4):
        from langchain_core.embeddings import Embeddings
        if not self._docs:
            return []
        q = self._embedding_fn.embed_query(query)
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        q = _normalize(q)
        # Cosine similarity: (1, dim) @ (dim, n) -> (1, n)
        sims = np.dot(q, self._embeddings.T).ravel()
        # Convert to distance: lower is better (1 - sim)
        distances = 1.0 - sims
        idx = np.argsort(distances)[: min(k, len(self._docs))]
        out = []
        for i in idx:
            d = self._docs[i]
            doc = Document(page_content=d["content"], metadata=d.get("metadata") or {})
            out.append((doc, float(distances[i])))
        return out

    def save_local(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self._embeddings)
        with open(path / "docs.json", "w", encoding="utf-8") as f:
            json.dump(self._docs, f, ensure_ascii=False, indent=0)

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        save_path: str | None = None,
    ) -> list[str]:
        """Append texts (and optionally save to path). Returns ids (or generated)."""
        metadatas = metadatas or [{}] * len(texts)
        ids = ids or [f"id_{len(self._docs) + i}" for i in range(len(texts))]
        vecs = self._embedding_fn.embed_documents(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        for i in range(len(vecs)):
            vecs[i] = _normalize(vecs[i])
        safe_meta = []
        for m in metadatas:
            safe_meta.append({k: v for k, v in (m or {}).items() if isinstance(k, str) and (v is None or isinstance(v, (str, int, float, bool)))})
        if len(self._docs) == 0:
            self._embeddings = vecs
        else:
            self._embeddings = np.vstack([self._embeddings, vecs])
        self._docs.extend([{"content": t, "metadata": m} for t, m in zip(texts, safe_meta)])
        if save_path:
            self.save_local(save_path)
        return ids

    @classmethod
    def load_local(cls, path: str, embedding_fn) -> "SimpleVectorStore":
        path = Path(path)
        embeddings = np.load(path / "embeddings.npy")
        with open(path / "docs.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        return cls(embeddings, docs, embedding_fn)

    @classmethod
    def from_documents(cls, documents: list, embedding_fn) -> "SimpleVectorStore":
        from langchain_core.documents import Document
        texts = [d.page_content for d in documents]
        # Keep only JSON-serializable metadata
        meta = []
        for d in documents:
            m = getattr(d, "metadata", {}) or {}
            safe = {k: v for k, v in m.items() if isinstance(k, str) and (v is None or isinstance(v, (str, int, float, bool)))}
            meta.append(safe)
        vecs = embedding_fn.embed_documents(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        for i in range(len(vecs)):
            vecs[i] = _normalize(vecs[i])
        docs = [{"content": t, "metadata": m} for t, m in zip(texts, meta)]
        return cls(vecs, docs, embedding_fn)


def simple_store_exists(path: str) -> bool:
    p = Path(path)
    return (p / "embeddings.npy").exists() and (p / "docs.json").exists()


def create_or_load_simple_store(path: str, embedding_fn) -> "SimpleVectorStore":
    """Load existing simple store or create an empty one (for indexer add_texts)."""
    path = Path(path)
    if simple_store_exists(str(path)):
        return SimpleVectorStore.load_local(str(path), embedding_fn)
    return SimpleVectorStore(np.zeros((0, 0), dtype=np.float32), [], embedding_fn)
