#!/usr/bin/env python3
"""
Open-source RAG indexer: AltaStata + simple vector store + HuggingFace embeddings.

Listens for SHARE events (like bob_indexer.py) or indexes a path once.
Chunks are stored in AltaStata; vectors and metadata in the simple (numpy) store.
"""

import os
import sys
import threading
import queue
import signal
import atexit

# Parent repo root for altastata
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    ALTASTATA_ACCOUNT_DIR,
    ALTASTATA_PASSWORD,
    ALTASTATA_ACCOUNT_ID,
    ALTASTATA_CALLBACK_PORT,
    ALTASTATA_USE_HPCS,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAG_INDEX_PATH,
    LOCAL_INDEX_PATH,
)


def get_chunk_path(base_path: str, chunk_index: int) -> str:
    """Chunk storage path in AltaStata (same convention as Vertex example)."""
    return f"chunks/{base_path}_{chunk_index}.txt"


def get_vector_store(embeddings):
    """Return simple vector store if index exists."""
    from simple_store import SimpleVectorStore, simple_store_exists
    if simple_store_exists(LOCAL_INDEX_PATH):
        return SimpleVectorStore.load_local(LOCAL_INDEX_PATH, embeddings)
    return None


class OpenRAGIndexer:
    """Event-driven indexer: AltaStata -> chunk in AltaStata -> embed -> simple store."""

    def __init__(self):
        self.fs = None
        self.altastata = None
        self.embeddings = None
        self.vector_store = None
        self.event_queue = queue.Queue()
        self.processing_thread = None
        self.stop = False
        self._lock = threading.Lock()
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    def _on_signal(self, signum, _frame):
        print(f"\n🛑 Signal {signum}, shutting down...")
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        self.stop = True
        if self.altastata:
            try:
                self.altastata.shutdown()
            except Exception:
                pass
        print("✅ Indexer cleanup done")

    def _event_handler(self, event_name, data):
        print(f"\n🔔 Event queued: {event_name}")
        self.event_queue.put((event_name, data))

    def _process_queue(self):
        while not self.stop:
            try:
                event_name, data = self.event_queue.get(timeout=1.0)
                if event_name == "SHARE":
                    self._index_file(str(data))
                elif event_name == "DELETE":
                    print(f"🗑️  Delete (not yet implemented): {data}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Process error: {e}")
                import traceback
                traceback.print_exc()

    def _index_file(self, file_path: str):
        """Read file from AltaStata, chunk, store chunks in AltaStata, embed, add to simple vector store."""
        base_path = file_path.split("✹")[0] if "✹" in file_path else file_path
        print(f"📄 Indexing: {base_path}")

        with self._lock:
            try:
                with self.fs.open(file_path, "r") as f:
                    content = f.read()
            except Exception as e:
                print(f"   ❌ Read failed: {e}")
                return

            doc = Document(
                page_content=content,
                metadata={"source": file_path, "filename": os.path.basename(base_path)},
            )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = text_splitter.split_documents([doc])

            chunk_paths = []
            for i, chunk in enumerate(chunks):
                chunk_path = get_chunk_path(base_path, i)
                chunk_paths.append(chunk_path)
                self.altastata.create_file(chunk_path, chunk.page_content.encode("utf-8"))
                print(f"   ✅ Chunk {i + 1}/{len(chunks)} -> {chunk_path}")

            texts = [c.page_content for c in chunks]
            # Store will embed via self.embeddings when we add_texts
            metadatas = [
                {
                    "chunk_path": chunk_paths[i],
                    "source_file": base_path,
                    "source": file_path,
                    "chunk_index": str(i),
                }
                for i in range(len(chunks))
            ]
            ids = [f"{base_path.replace('/', '_')}_{i}" for i in range(len(chunks))]
            self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids, save_path=LOCAL_INDEX_PATH)
            print(f"   ✅ Indexed {len(chunks)} chunks")

    def index_path_once(self, path_prefix: str):
        """One-off: list files under path_prefix in AltaStata and index each."""
        if not self.fs or not self.vector_store:
            raise RuntimeError("Initialize first with initialize()")
        print(f"[indexer] Listing AltaStata path: {path_prefix!r}")
        try:
            entries = self.fs.find(path_prefix)
        except Exception as e:
            print(f"❌ List failed: {e}")
            import traceback
            traceback.print_exc()
            return
        files = [e for e in entries if getattr(e, "type", None) != 1 and not e.endswith("/")]
        print(f"[indexer] Found {len(entries)} entries, {len(files)} file(s) to index")
        if not files:
            print(f"[indexer] No files at {path_prefix!r}. Upload documents to AltaStata and share with this account, or use a path that already has files.")
        for f in files:
            self._index_file(f)

    def initialize(self):
        """Connect AltaStata, load embeddings, create or connect simple vector store."""
        print(f"[indexer] initialize: ALTASTATA_ACCOUNT_DIR={ALTASTATA_ACCOUNT_DIR!r}, isdir={os.path.isdir(ALTASTATA_ACCOUNT_DIR) if ALTASTATA_ACCOUNT_DIR else False}")
        if not ALTASTATA_ACCOUNT_DIR or not os.path.isdir(ALTASTATA_ACCOUNT_DIR):
            print("❌ Set ALTASTATA_ACCOUNT_DIR to your AltaStata account directory (e.g. .altastata/accounts/azure.rsa.bob123)")
            return False

        from altastata.altastata_functions import AltaStataFunctions
        from altastata.fsspec import create_filesystem

        print("[indexer] Connecting to AltaStata (from_account_dir + set_password)...")
        self.altastata = AltaStataFunctions.from_account_dir(
            ALTASTATA_ACCOUNT_DIR,
            callback_server_port=ALTASTATA_CALLBACK_PORT,
        )
        if not ALTASTATA_USE_HPCS:
            self.altastata.set_password(ALTASTATA_PASSWORD)
        self.fs = create_filesystem(self.altastata, ALTASTATA_ACCOUNT_ID)
        print("✅ AltaStata connected")

        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"✅ Embeddings: {EMBEDDING_MODEL}")

        from simple_store import create_or_load_simple_store
        self.vector_store = create_or_load_simple_store(LOCAL_INDEX_PATH, self.embeddings)
        print("✅ Simple vector store ready")

        return True

    def run_listener(self):
        """Start event listener and process queue (like bob_indexer)."""
        if not self.altastata or not self.vector_store:
            if not self.initialize():
                return
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        self.altastata.add_event_listener(self._event_handler)
        print("🎧 Listening for SHARE events. Upload and share docs to trigger indexing.")
        print("⏳ Ctrl+C to stop.\n")
        try:
            while not self.stop:
                threading.Event().wait(1)
        except KeyboardInterrupt:
            pass
        self.stop = True
        if self.altastata:
            try:
                self.altastata.remove_event_listener(self._event_handler)
            except Exception:
                pass
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        if self.altastata:
            self.altastata.shutdown()


def main():
    import argparse
    from simple_store import simple_store_exists
    p = argparse.ArgumentParser(description="Open RAG indexer (simple vector store + AltaStata)")
    p.add_argument("--once", type=str, default="", help="Index once from this AltaStata path (e.g. RAGDocs/policies) and exit")
    args = p.parse_args()

    indexer = OpenRAGIndexer()
    if not indexer.initialize():
        sys.exit(1)
    if args.once:
        indexer.index_path_once(args.once)
        if indexer.altastata:
            indexer.altastata.shutdown()
        return
    # Default: index what we have (if no index yet), then listen for SHARE events
    if not simple_store_exists(LOCAL_INDEX_PATH) and RAG_INDEX_PATH and RAG_INDEX_PATH.strip():
        print(f"📂 Indexing existing path: {RAG_INDEX_PATH}")
        indexer.index_path_once(RAG_INDEX_PATH.strip())
        print("✅ Bootstrap index done.")
    print("🎧 Starting event listener for new shared files.\n")
    indexer.run_listener()


if __name__ == "__main__":
    main()
