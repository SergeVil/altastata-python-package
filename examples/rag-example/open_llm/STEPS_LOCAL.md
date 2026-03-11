# Step-by-step: Run RAG demo locally (simple mode, no Docker)

Uses the **simple vector store** (pure Python + numpy). Works on **x86, ARM, s390x**.

---

## Step 1 – Install

From the **repo root**:

```bash
pip install -e .
cd examples/rag-example/open_llm
pip install -r requirements.txt
```

---

## Step 2 – Configure

In **`examples/rag-example/open_llm/`**:

```bash
cp .env.example .env
```

Edit `.env` and set **`ALTASTATA_ACCOUNT_DIR`** to your account path (e.g. `$HOME/.altastata/accounts/amazon.rsa.bob123`). Leave **`OLLAMA_BASE_URL=http://localhost:11434`** as is.

---

## Step 3 – Start Ollama

In a terminal:

```bash
ollama serve
ollama pull llama3.2:1b
```

Leave it running.

---

## Step 4 – Index sample documents

In **`examples/rag-example/open_llm/`**:

```bash
python index_local.py
```

Wait for **"Done. Indexed … chunks at …"**

---

## Step 5 – Start the web app

```bash
python web_app.py
```

Open **http://localhost:8000** and try e.g. **What are the password requirements?**

---

## Quick reference

| Step | Command |
|------|---------|
| 1 | Repo root: `pip install -e .` then `cd examples/rag-example/open_llm` and `pip install -r requirements.txt` |
| 2 | In `open_llm/`: `cp .env.example .env` and set `ALTASTATA_ACCOUNT_DIR` |
| 3 | `ollama serve` and `ollama pull llama3.2:1b` |
| 4 | In `open_llm/`: `python index_local.py` |
| 5 | In `open_llm/`: `python web_app.py` → http://localhost:8000 |

No Docker required. For **s390x**, use the same steps; the simple store uses only Python and numpy.
