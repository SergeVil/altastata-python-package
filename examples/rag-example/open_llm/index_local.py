#!/usr/bin/env python3
"""
Index a local directory into the simple vector store (no AltaStata). Use this to test RAG with
sample documents before uploading to the cloud. Run once, then use the web app or query_rag.
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LOCAL_INDEX_PATH,
)
from indexer import get_vector_store

# Default: examples/rag-example/sample_documents (sibling of open_llm)
_DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent.parent / "sample_documents"


def index_local_dir(local_dir: str | Path) -> None:
    local_path = Path(local_dir)
    if not local_path.is_dir():
        print(f"❌ Not a directory: {local_path}")
        sys.exit(1)

    print(f"📂 Indexing local directory: {local_path}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for f in sorted(local_path.iterdir()):
        if f.is_dir():
            continue
        if f.suffix.lower() not in (".txt", ".md", ".rst"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"   ⚠️  Skip {f.name}: {e}")
            continue
        doc = Document(
            page_content=text,
            metadata={"source": str(f), "filename": f.name},
        )
        chunks = text_splitter.split_documents([doc])
        all_chunks.extend(chunks)
        print(f"   ✅ {f.name} -> {len(chunks)} chunks")

    if not all_chunks:
        print("❌ No documents to index.")
        return

    from simple_store import SimpleVectorStore
    vector_store = SimpleVectorStore.from_documents(all_chunks, embeddings)
    vector_store.save_local(LOCAL_INDEX_PATH)
    print(f"✅ Done. Indexed {len(all_chunks)} chunks at {LOCAL_INDEX_PATH}. Run the web app to query.")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Index local files into simple vector store (no AltaStata)")
    p.add_argument(
        "dir",
        nargs="?",
        default=str(_DEFAULT_LOCAL_DIR),
        help=f"Directory to index (default: {_DEFAULT_LOCAL_DIR})",
    )
    args = p.parse_args()
    index_local_dir(args.dir)


if __name__ == "__main__":
    main()
