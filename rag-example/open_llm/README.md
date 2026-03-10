# RAG with Open LLM + AltaStata

This variant of the RAG example uses **no GCP**: open-source LLM (Ollama), a vector store (default: simple), and local embeddings (sentence-transformers), with **LangChain** for orchestration. Data can live in **AltaStata** (encrypted); the stack runs locally or in Docker.

**Recent changes:** Default Transformers model is **SmolLM2-360M-Instruct** (faster than TinyLlama on CPU). Chunk reads from AltaStata are **parallelized** to reduce query latency. Mac build script (`openshift/rag/build-and-run-rag-mac.sh`) supports `LLM_PROVIDER=ollama` with Ollama on the host for much faster answers. Entrypoint and indexer emit clearer logs for AltaStata connection debugging.

## Simplest demo (ARM, x86 – no Docker)

Uses a **simple vector store** (pure Python + numpy). Works on **ARM and x86** out of the box; for **IBM Z (s390x)** see [IBM Z (s390x) – will it work?](#ibm-z-s390x--will-it-work) below.

1. `pip install -e .` (repo root) then `cd rag-example/open_llm` and `pip install -r requirements.txt`
2. `cp .env.example .env` and set **ALTASTATA_ACCOUNT_DIR**
3. Start Ollama: `ollama serve` and `ollama pull llama3.2:1b`
4. `python index_local.py` then `python web_app.py`
5. Open http://localhost:8766 (Docker maps host 8766 → app 8000)

→ Full steps: [STEPS_LOCAL.md](STEPS_LOCAL.md)

## Architecture

```
AltaStata (encrypted docs + chunks)
    ↓
LangChain (load, chunk, embed)
    ↓
Simple vector store (numpy) + HuggingFace (embeddings)
    ↓
Ollama (open LLM)
    ↓
FastAPI web app (HTTP or HTTPS)
```

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, no API key).
- **Vector store**: Simple (numpy only, s390x-safe).
- **LLM**: [Ollama](https://ollama.com/) (e.g. `llama3.2`, `mistral`, `phi`).
- **Orchestration**: LangChain (document loaders, text splitters, retriever, chain).

## Running without Docker

Use this to run everything on your Mac (no containers). Ollama runs natively and can use **Metal (GPU)** for much faster answers.

**Fast local Mac:** Run Ollama (`ollama serve`, `ollama pull llama3.2:1b`) and **do not set** `LLM_PROVIDER` in `.env`. The app will auto-detect Ollama and use it (fast). If Ollama isn’t running, it falls back to Transformers (slower; default timeout 180s).

**Simplest everywhere (Mac, Docker, 390):** Use an **IBM watsonx project** and set `LLM_PROVIDER=watsonx` with `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` in `.env`. No local model to run; fast answers from the Granite API; free tier is enough for testing. See [.env.example](.env.example) for the watsonx variables. **Note:** With watsonx, retrieved chunks are sent to the cloud model; for data that must stay inside confidential containers, use a local LLM (transformers, Ollama, or MLX) instead.

### Step-by-step (local, no Docker)

**→ See [STEPS_LOCAL.md](STEPS_LOCAL.md) for a numbered checklist you can follow.**

Do these in order. You need **Ollama** installed.

**Step 1 – Install dependencies**

From the **repo root**:

```bash
pip install -e .
cd rag-example/open_llm
pip install -r requirements.txt
```

**Step 2 – Create `.env`**

In **`rag-example/open_llm/`**, copy `.env.example` to `.env` and set at least:

- **`ALTASTATA_ACCOUNT_DIR`** – path to your AltaStata account dir (e.g. `$HOME/.altastata/accounts/amazon.rsa.bob123`).
- **`OLLAMA_BASE_URL`** can stay `http://localhost:11434` (Ollama on the host).


### 3. Start Ollama (native, uses Metal on Apple Silicon)

In a terminal:

```bash
ollama serve
ollama pull smollm2:360m
```

Leave this running. Default is **smollm2:360m** (fast, 360M params). Alternatives: `ollama pull qwen2.5:0.5b` or `ollama pull llama3.2:1b`.

### 4. Start the web app

In another terminal, from **`rag-example/open_llm/`**:

```bash
python web_app.py
# or
./run_local.sh
```

Then open **http://localhost:8000**.

### 5. Index documents (optional)

**Quick test with sample docs (no AltaStata upload):**

```bash
python index_local.py
```

This indexes `rag-example/sample_documents/` . **Restart the web app** after running so it picks up the new index, then ask e.g. “What are the password requirements?”.

**Index from AltaStata** (after uploading/sharing docs there), in a **third** terminal from `rag-example/open_llm/`:

```bash
python indexer.py --once RAGDocs/policies   # one-off
# or
python indexer.py                           # then listen for SHARE events
```

Index data is stored under `rag-example/open_llm/local_index/`.



---

## Quick start (local) – reference

### 1. Install

```bash
cd rag-example/open_llm
pip install -r requirements.txt
```

From the repo root, install AltaStata in dev mode if needed:

```bash
pip install -e .
```

### 2. Set environment

**`.env` lives in** `rag-example/open_llm/.env`. Copy `.env.example` to `.env` there and set at least:

- `ALTASTATA_ACCOUNT_DIR` – path to your AltaStata account dir (e.g. Bob’s for RAG).

### 3. Run Ollama (if not already)

```bash
ollama serve
ollama pull llama3.2:1b
```

### 4. Index documents (one-off or listener)

**One-off** – index everything under an AltaStata path:

```bash
export ALTASTATA_ACCOUNT_DIR=/path/to/.altastata/accounts/azure.rsa.bob123
python indexer.py --once RAGDocs/policies
```

**Event-driven** – listen for SHARE events (like the GCP Bob indexer):

```bash
python indexer.py
# Then upload and share docs from another account (e.g. alice_upload_docs.py from parent)
```

### 5. Query (CLI or web)

**CLI:**

```bash
python query_rag.py "What are the password requirements?"
# or interactive:
python query_rag.py
```

**Web:**

```bash
python web_app.py
# Open http://localhost:8000
```

**HTTPS:** set `SSL_CERT_FILE` and `SSL_KEY_FILE` in `.env`, then run `python web_app.py`. The app will serve over HTTPS.

## How the app reads RAG data from AltaStata

- **How it reads**: The container uses the **AltaStata Python API** with the account whose directory you mount. At startup the app calls `AltaStataFunctions.from_account_dir(ALTASTATA_ACCOUNT_DIR)` and builds an **fsspec** filesystem. Queries use that filesystem to **read chunk files** from AltaStata (paths stored in the vector store metadata). So the app does **not** read from the host filesystem directly—it talks to AltaStata’s backend using the credentials in the mounted account directory.
- **What user (account) it uses**: The app runs as the **recipient** AltaStata account that was **shared** the documents. That is controlled by:
  - **`ALTASTATA_ACCOUNT_DIR`** – path to that account’s directory on the host (e.g. `~/.altastata/accounts/azure.rsa.bob123`). The directory must contain the account’s config and keys (e.g. `user.properties`, private key).
  - **`ALTASTATA_ACCOUNT_ID`** – the account id (default `bob123`), used as the “user” name when creating the fsspec filesystem. It must match the account in `ALTASTATA_ACCOUNT_DIR`.
  - **`ALTASTATA_USE_HPCS`** – set to `1` (or `true`/`yes`) for **HPCS (IBM Hyper Protect)** accounts, which use key blobs (GREP11) instead of a password; the app will skip `set_password`. For password-based accounts (e.g. bob123), leave unset.
- **What directory should the data provider create**: The **data provider** (e.g. Alice) does **not** create a directory on the RAG server or in the container. They:
  1. Use their **own** AltaStata account to **upload** documents to a **path in AltaStata** (e.g. `RAGDocs/policies/`).
  2. **Share** those files with the **RAG account** (e.g. `bob123`) via AltaStata’s share API (e.g. `share_files(..., users=["bob123"])`).

So the “directory” the data provider creates is a **path prefix in AltaStata** (e.g. `RAGDocs/policies`). The RAG operator mounts the **Bob** account directory so the app can read everything that was shared with Bob. To index that data, run the indexer (once or event-driven) with the same Bob account; it will list and read from paths like `RAGDocs/policies` and store chunks in AltaStata under `chunks/...`.

### Cloud (AWS vs Azure), files, and directory

- **Which cloud does bob123 use? / Which cloud is the Docker container using?**  
  **The container does not choose a cloud.** It uses whatever AltaStata account directory you **mount** into the container. That account’s directory name indicates the backend:
  - **`azure.rsa.bob123`** → Azure
  - **`amazon.rsa.bob123`** → AWS  
  Set `ALTASTATA_ACCOUNT_DIR` in `.env` to your **host** path (e.g. `$HOME/.altastata/accounts/amazon.rsa.bob123` for AWS, or `$HOME/.altastata/accounts/azure.rsa.bob123` for Azure). The same image works for any cloud; the mounted account dir decides which backend is used.

- **What files to upload?**  
  Any documents you want the RAG to search (policies, docs, text). The repo includes sample files in **`rag-example/sample_documents/`**: `company_policy.txt`, `security_guidelines.txt`, `remote_work_policy.txt`, `ai_usage_policy.txt`. Use those for testing or upload your own.

- **What directory (path) to use?**  
  This is a **path inside AltaStata**, not on disk. The examples use the path prefix **`RAGDocs/policies`**. So the data provider uploads files to paths like `RAGDocs/policies/company_policy.txt`, then shares them with **bob123**. You can use another prefix (e.g. `MyOrg/docs`) as long as you share with the same account the RAG app uses. When indexing, use that path (e.g. `indexer.py --once RAGDocs/policies`).

## Docker: full stack + HTTPS

Goal: one Docker setup that takes data from AltaStata and serves requests via an HTTPS web interface.

### 1. Prepare

- Have an AltaStata **recipient** account directory on the host (e.g. Bob: `~/.altastata/accounts/azure.rsa.bob123`). The data provider must have **shared** the documents with this account (e.g. `bob123`).
- In **`rag-example/open_llm/`**, copy `.env.example` to `.env` and set:
  - **`ALTASTATA_ACCOUNT_DIR`** to the **host path** to that directory (e.g. `$HOME/.altastata/accounts/azure.rsa.bob123`).

### 2. Run with Docker Compose

From **repo root** (so the Dockerfile build context can see the AltaStata package):

```bash
cd rag-example/open_llm
docker compose up -d
```

This starts:

- **Simple vector store** – index in container volume.
- **app** – FastAPI RAG web app (host port **8766** → container 8000). Uses **Transformers** with the **same model as the Mac MLX default** (Llama 3.2 1B Instruct) so answers are consistent. The first query may need to download the model; if the model is gated, log in with `huggingface-cli login` or set `HF_TOKEN` in the app environment.
- **Optional – Ollama**: To use Ollama in Docker instead, add the `ollama` profile and pull a model:  
  `docker compose --profile ollama up -d` then `docker compose exec ollama ollama pull llama3.2`.

Open **http://localhost:8766** to use the web UI.

**Mac image (same layout as s390x – AltaStata only):** To run the same setup as on IBM 390 (no sample_documents, index from AltaStata at startup) on your Mac:

```bash
# From repo root. Set your account dir (password-based or HPCS).
ALTASTATA_ACCOUNT_DIR=$HOME/.altastata/accounts/amazon.rsa.bob123 ./openshift/rag/build-and-run-rag-mac.sh
# Optional: ALTASTATA_PASSWORD=yourpass HF_LLM_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct
# Much faster: run Ollama on the host (ollama run smollm2:360m), then:
# LLM_PROVIDER=ollama ALTASTATA_ACCOUNT_DIR=... ./openshift/rag/build-and-run-rag-mac.sh
```

Image: `openshift/rag/Dockerfile.open_llm_mac`. Build only: `docker build -f openshift/rag/Dockerfile.open_llm_mac -t altastata/rag-open-llm:latest .`

### 3. HTTPS in front of the app

The app supports TLS if you set `SSL_CERT_FILE` and `SSL_KEY_FILE` (e.g. in `docker-compose` `environment` and mount the certs). Alternatively, put a reverse proxy in front:

- **Option A – Caddy**: Uncomment the `caddy` service in `docker-compose.yml`, add a `Caddyfile` that proxies to `app:8000`, and expose 443.
- **Option B – External proxy**: Run Caddy, nginx, or your load balancer on the host and proxy to `localhost:8766` with TLS termination.

### 4. Indexing inside Docker

**Automatic at startup (s390x / Mac production images):** When using the s390x image (`openshift/rag/Dockerfile.open_llm_s390x`) or the Mac image (`openshift/rag/Dockerfile.open_llm_mac`), the entrypoint builds the index from AltaStata at container start if no index exists: it runs `indexer --once` on `RAG_INDEX_PATH` (default `RAGDocs/policies`). So the first run indexes automatically; no manual indexer step needed.

**Manual (docker-compose or one-off):** To run the indexer inside the same app image (e.g. one-off index after startup):

```bash
docker compose exec app python indexer.py --once RAGDocs/policies
```

For an event-driven indexer, run the indexer as a long-lived process (e.g. a second container or a background job) that shares the same `ALTASTATA_ACCOUNT_DIR`.

## Why is it slow? / Performance

Each RAG query does: **embed query** (sentence-transformers) → **vector search** → **read chunks from AltaStata** → **Ollama LLM**. Most of the time is **Ollama (LLM inference)**.

### Is my Mac too slow?

Probably not “too slow” — it’s that **running the LLM in Docker on a Mac is CPU-only and heavy**. Ollama in a container usually doesn’t use GPU/Metal; it runs on CPU, so each reply can take **5–30+ seconds** even for a small model like `llama3.2:1b`. That’s normal for local CPU inference.

### What actually takes the time

- **Ollama (LLM)** – most of the delay (several seconds per answer).
- **First request** – loading the embedding model and first Ollama call (warmup at startup reduces this).
- **Embedding + vector store + AltaStata** – usually under a couple of seconds total.

### Ways to make it feel faster

1. **Run Ollama on your Mac (not in Docker)** so it can use **Metal (GPU)** on Apple Silicon:
   - Install Ollama on the host: `brew install ollama` (or from ollama.com), run `ollama serve`, pull `llama3.2:1b`.
   - In `.env` set `OLLAMA_BASE_URL=http://host.docker.internal:11434` and **remove or stop** the `ollama` service in `docker-compose.yml` so only the app and indexer use Docker. The app will call Ollama on your Mac; Ollama can use Metal and will be faster.

2. **Keep the small model** – `llama3.2:1b` is already the default. Larger models will be slower on CPU.

3. **Short answers** – the app limits reply length (`OLLAMA_NUM_PREDICT=256`); you can set `OLLAMA_NUM_PREDICT=128` in `.env` for even shorter, quicker replies.

4. **More CPU for Ollama in Docker** – in `docker-compose.yml` you can give the `ollama` service more CPUs (e.g. `deploy.resources.limits` / `cpus`), but CPU inference will still be slower than Ollama on the host with Metal.

**Summary:** The slowness is mostly the LLM running on CPU. For a faster experience on a Mac, run Ollama on the host with Metal and point the app at `http://host.docker.internal:11434`.

### Ollama still slow on Mac?

If **Ollama is slow even when running natively** on your Mac, check:

1. **Ollama must run on the host, not in Docker** – If Ollama is in a container, it won’t use Metal (GPU). Quit Docker’s Ollama and run `ollama serve` in a terminal on your Mac (install with `brew install ollama` or from ollama.com).
2. **Use the smallest model** – Default is `smollm2:360m`. For maximum speed try `ollama pull smollm2:360m` and set `OLLAMA_MODEL=smollm2:360m` in `.env`. Smaller = faster.
3. **Let the app use Ollama** – If `.env` has `LLM_PROVIDER=transformers`, the app uses Transformers (CPU), not Ollama. Remove `LLM_PROVIDER` or set `LLM_PROVIDER=ollama` so the app talks to Ollama. Leave `OLLAMA_BASE_URL=http://localhost:11434`.
4. **First query is always slower** – Model loads into memory; the next queries are faster. Wait for one answer to finish before judging.
5. **Faster option: remote LLM** – For the fastest answers, use a **remote** LLM (e.g. a cloud API or another machine with a GPU). Set `OLLAMA_BASE_URL` to that server’s URL; the app on your Mac only does retrieval and sends the prompt to the remote model.

### Is there any LLM that works fast on M1?

**Yes.** On Apple Silicon (M1/M2/M3/M4), the fastest local option is **Apple’s MLX** (Metal-optimized). This app supports it.

| Option | Speed on M1 | Notes |
|--------|-------------|--------|
| **MLX** (`LLM_PROVIDER=mlx`) | Fastest (typically 1.5–2× Ollama) | Uses Metal; Mac-only. Set `LLM_PROVIDER=mlx` and `MLX_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit` (or another [MLX model](https://huggingface.co/mlx-community)). |
| **Ollama** (native on Mac) | Fast | Run `ollama serve` on the host (not in Docker) so it uses Metal. Use a small model (e.g. `smollm2:360m`). |
| **Transformers** | Slow on CPU | Same stack on Mac and IBM Z; use when you need portability, not speed. |

**To use MLX (fastest on M1):**

1. Install: `pip install mlx mlx-lm`
2. In `.env`: `LLM_PROVIDER=mlx` and e.g. `MLX_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit`
3. Run the app; the first query will download the model, then inference uses Metal and is fast.

MLX is **Mac-only** (Metal). On Linux/IBM Z use `ollama` (if available) or `transformers`.

### "Error getting collection: Collection [uuid] does not exist"

That message comes from **ChromaDB**. This demo **does not use Chroma**; it uses only the **simple vector store** (numpy). So if you see it:

1. **Rebuild the app image** so no old Chroma code is in the container:  
   `docker compose build --no-cache app` then `docker compose up -d`
2. **Unset Chroma env vars** if you have any: `unset CHROMA_SERVER_HOST CHROMA_SERVER_HTTP_PORT CHROMA_COLLECTION` (or remove them from `.env`)
3. **Confirm you're running this app** – the open_llm stack uses only `simple_store.py` and `LOCAL_INDEX_PATH`; another script or project might still be calling Chroma.

After that, run `python index_local.py` (or let the Docker entrypoint index on first run) so the simple index exists, then start the web app again.

### IBM Z (s390x) – will it work?

**Partially, with extra setup.** The stack has three parts; here’s how they behave on s390x:

| Component | s390x status |
|-----------|----------------|
| **Vector store** | ✅ Works. Simple store is pure Python + numpy; numpy is available on s390x. |
| **Embeddings** | ✅ Works when PyTorch is installed (e.g. via [IBM AI Toolkit for IBM Z and LinuxONE](https://www.ibm.com/support/pages/ai-toolkit-ibm-z-and-linuxone) or [IBM pytorch-on-z](https://github.com/IBM/pytorch-on-z)); then `sentence-transformers` runs as usual. See [openshift/IBM-LINUXONE-S390X-SETUP.md](../../openshift/IBM-LINUXONE-S390X-SETUP.md). |
| **LLM (Ollama)** | ❌ No official s390x build. Ollama does not ship binaries for s390x. You can try building Ollama from source on s390x, or run the RAG app against a **remote** LLM (e.g. Ollama or an OpenAI-compatible API on another host) by setting `OLLAMA_BASE_URL` to that server. |

**Summary:** On s390x, the vector store and embeddings (with PyTorch installed) work. The only extra requirement is an LLM: either build Ollama from source on s390x or point `OLLAMA_BASE_URL` at a remote LLM server.

**Models that work on s390x:**

- **Ollama (build from source):** Ollama can be built on s390x (`make` works; the install script doesn’t yet support s390x). Once built, any model Ollama supports (e.g. Llama, Mistral, Phi, Gemma) works.
- **llama.cpp:** You can **build llama.cpp on s390x**; see [Build llama.cpp on s390x](https://fossies.org/linux/llama.cpp/docs/build-s390x.md#:~:text=%22Fossies%22%20%2D%20the%20Free%20Open,cmake%20%2DS%20.%20%2D). Upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) has s390x CI; use GGUF models and the `--bigendian` conversion option for s390x. The [llama.cpp-s390x](https://github.com/taronaeo/llama.cpp-s390x) fork adds IBM Z/LinuxONE optimizations. Run a llama.cpp server and point `OLLAMA_BASE_URL` at it if it exposes an Ollama-compatible API, or add a small adapter.
- **Remote LLM:** Point `OLLAMA_BASE_URL` at an Ollama or OpenAI-compatible server on another host; then any model that server runs works from the RAG app on s390x.

### Running TinyLlama on LinuxONE

Two main ways to run TinyLlama (or similar models) on s390x:

1. **Via Ollama:** Build [Ollama from source](https://github.com/ollama/ollama) on your s390x instance, then run `ollama run tinyllama`. Ollama will pull the model and run it using its internal inference engine (llama.cpp under the hood).

2. **Via llama.cpp:** Build [llama.cpp on s390x](https://fossies.org/linux/llama.cpp/docs/build-s390x.md#:~:text=%22Fossies%22%20%2D%20the%20Free%20Open,cmake%20%2DS%20.%20%2D). The community has added big-endian fixes and the `--bigendian` flag for model conversion. Run a llama.cpp server and point `OLLAMA_BASE_URL` at it if it exposes an Ollama-compatible API, or use our app with `LLM_PROVIDER=transformers` and TinyLlama via PyTorch.

**Performance and optimization:**

- **Hardware acceleration (z17 / LinuxONE 5):** Enable IBM NNPA (Neural Network Processing Assist) by building llama.cpp with `-DGGML_ZDNN=ON`. **Use `-DGGML_ZDNN=ON` even if your current host is not z17/LinuxONE 5:** the binary will use NNPA when available and fall back to CPU elsewhere, so one build is ready for future hardware.
- **CPU (older LinuxONE):** The model runs on IFL (Integrated Facility for Linux) cores. For best results use at least 8 shared IFLs.
- **Quantization:** Use quantized GGUF variants (e.g. Q4_K_M or Q2_K) to reduce memory and latency; they are supported on s390x.

**Building llama.cpp on s390x (optional):** To use a llama.cpp server instead of Transformers, build [llama.cpp on s390x](https://fossies.org/linux/llama.cpp/docs/build-s390x.md) on your host (upstream has s390x CI and big-endian GGUF notes). You can then point `OLLAMA_BASE_URL` at that server or compare latency with the in-container Transformers (TinyLlama) backend.

### Simplest run on IBM Z (s390x) – no Docker build

Run the RAG stack on 390 **without building any Docker images**: use Python on the s390x host and a **remote** LLM so you don’t need Ollama on 390.

**Prerequisites on the s390x host:** Python 3.10+, pip, PyTorch (for embeddings, e.g. from IBM AI Toolkit or pytorch-on-z). No Docker required.

**1. Get the code** (clone or copy the repo onto the s390x machine.)

**2. Install dependencies** (from repo root, then open_llm):

```bash
pip install -e .
cd rag-example/open_llm
pip install -r requirements.txt
```

**3. Configure** – in `rag-example/open_llm/` copy `.env.example` to `.env` and set:

- **`OLLAMA_BASE_URL`** – URL of a **remote** LLM server (e.g. `http://your-ollama-host:11434` or an OpenAI-compatible API). The RAG app on 390 will call this for answers; nothing runs locally on 390.
- **`ALTASTATA_ACCOUNT_DIR`** – path to your AltaStata account directory on the s390x host (if you use AltaStata); can leave empty for local-only indexing.

**4. Index sample documents** (optional, for a quick test):

```bash
cd rag-example/open_llm
python index_local.py
```

**5. Start the web app:**

```bash
python web_app.py
```

Then open **http://\<s390x-host-ip\>:8765** (or the port in your `.env`, default 8000 if `WEB_PORT` is unset).

**Summary:** No Docker, no image build on 390. Python + PyTorch on 390 for indexing and the web app; LLM runs elsewhere and is reached via `OLLAMA_BASE_URL`.

### IBM Z (s390x) containers – no pre-built LLM images

Pre-built “model-in-a-container” images (e.g. Red Hat Granite on Docker Hub) are **amd64/arm64 only**; they do **not** ship for **IBM Z (s390x)**.

**What exists for IBM 390:**

- **IBM Z runtime containers** (s390x) – frameworks only, no bundled LLM:
  - [IBM Z Open Source Hub – Containers](https://ibm.github.io/ibm-z-oss-hub/containers/index.html): **ibmz-accelerated-for-pytorch**, **ibmz-accelerated-for-tensorflow**, **ibmz-accelerated-for-nvidia-triton-inference-server**, etc. (e.g. from `icr.io/ibmz/...`).
  - [IBM ai-on-z-containers](https://github.com/IBM/ai-on-z-containers): example Dockerfiles/Containerfiles for AI stacks on s390x.
  - **LLM-optimized (compiled path):** [**zDLC**](https://ibm.github.io/zDLC/) (`icr.io/ibmz/zdlc`) compiles ONNX models into shared libraries optimized for Z (SIMD, Integrated Accelerator for AI on z16/z17). Use it when you export your LLM to ONNX and run inference via the compiled C/C++/Java/Python libs—different workflow from our Python Transformers stack.

- **No IBM-published “LLM pre-built in a container” for s390x** – you use a runtime image and either load a model at run time (e.g. Hugging Face + PyTorch) or point at a remote LLM. For **Python + Transformers + TinyLlama** (this RAG app), **ibmz-accelerated-for-pytorch** is the right base; there is no separate “LLM-optimized” Python image that replaces it.

**Practical options on 390:**

1. **Transformers in a PyTorch container** – Base your app image on IBM’s **ibmz-accelerated-for-pytorch** (or our Dockerfile built on s390x). Set `LLM_PROVIDER=transformers`; the app downloads the default model (TinyLlama) from Hugging Face on first run. No separate “model container.”
2. **Remote LLM** – Run the RAG app on 390 and set `OLLAMA_BASE_URL` (or equivalent) to a server that runs the LLM (e.g. on x86). No LLM container on 390.
3. **Build llama.cpp or Ollama on s390x** – llama.cpp can be built on 390 ([build guide](https://fossies.org/linux/llama.cpp/docs/build-s390x.md#:~:text=%22Fossies%22%20%2D%20the%20Free%20Open,cmake%20%2DS%20.%20%2D)); Ollama can also be built from source. Run the LLM on 390 yourself; no pre-built s390x LLM image to pull.

**RAG container image for s390x:** Build on the server, push to ICR, then pull and run. The container builds the index from AltaStata at startup if no index exists (see [Indexing inside Docker](#4-indexing-inside-docker)).

**Workflow (from your Mac):**
1. **Build on LinuxONE server:** `./openshift/rag/build-rag-s390x-on-server.sh` (syncs repo, builds image on the server).
2. **Push to ICR:** `./openshift/rag/push-rag-s390x-to-icr-from-server.sh` (requires `ICR_TOKEN`).
3. **Pull and run:** `./openshift/rag/pull-and-run-rag-s390x-from-icr.sh` (pulls image on the server, runs container, runs a test query).

**Server setup:** The server needs three files: `grep11client.yaml` (e.g. in `/etc/ep11client/`), and in the account directory (e.g. `/root/.altastata/accounts/amazon.rsa.hpcs.serge678/`) the files `hpcs-privkey.blob` and `*.user.properties` (container-ready: no `hpcs-yaml-path` or `hpcs-priv-key-blob-path` in the properties file). See [openshift/README-ICR-BUILD-AND-PUSH.md](../../openshift/README-ICR-BUILD-AND-PUSH.md) for details.

**Build (on s390x host or from Mac via script):**
```bash
# From Mac – sync repo and build on the LinuxONE server (recommended)
./openshift/rag/build-rag-s390x-on-server.sh

# Or on the s390x server directly
source ../../version.sh 2>/dev/null || true
docker build -f openshift/rag/Dockerfile.open_llm_s390x --platform linux/s390x \
  -t altastata/rag-open-llm-s390x:latest -t altastata/rag-open-llm-s390x:${VERSION:-2026b_latest} .
```

**Run (after build or pull from ICR):** Use the same version tag as the Jupyter image (from `version.sh`).
```bash
source ../../version.sh 2>/dev/null || true
# Replace with your AltaStata account dir (must contain the account subdir, e.g. amazon.rsa.bob123)
export ALTASTATA_ACCOUNT_DIR="$HOME/.altastata/accounts/amazon.rsa.bob123"

# 8 GB VM: use gpt2. For faster answers (still decent quality): HF_LLM_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct. 16+ GB: TinyLlama for best quality.
docker run -d -p 8000:8000 --name rag \
  -e ALTASTATA_ACCOUNT_DIR=$ALTASTATA_ACCOUNT_DIR \
  -e HF_LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  -v "$(dirname $ALTASTATA_ACCOUNT_DIR)":/root/.altastata/accounts:ro \
  altastata/rag-open-llm-s390x:${VERSION:-2026b_latest}
# Open http://<host>:8000/
```

**Pull from IBM Container Registry (ICR):** See [openshift/README-ICR-BUILD-AND-PUSH.md](../../openshift/README-ICR-BUILD-AND-PUSH.md) for push; to run from a pre-pushed image (tag matches Jupyter image, e.g. 2026b_latest from `version.sh`):
```bash
source ../../version.sh 2>/dev/null || true
docker pull icr.io/altastata/rag-open-llm-s390x:${VERSION:-2026b_latest}
# Then run as above, using image icr.io/altastata/rag-open-llm-s390x:${VERSION}
```

**Pull and run from your Mac (script):** From repo root run `./openshift/rag/pull-and-run-rag-s390x-from-icr.sh`. It SSHs to the server, stops/removes any existing RAG container, pulls the image, runs the container, runs a test query, and leaves the container running. Set `ICR_TOKEN` on your Mac. **Accounts:** Default is **HPCS** (`amazon.rsa.hpcs.serge678`; no password; script passes `ALTASTATA_USE_HPCS=1`). For **bob123** (password-based): `ACCOUNT_NAME=amazon.rsa.bob123 ./openshift/rag/pull-and-run-rag-s390x-from-icr.sh`. Optional env: `SSH_HOST`, `SSH_KEY`, `HF_LLM_MODEL=gpt2` (8 GB VMs). Account dir must exist on the server at `$REMOTE_ALTASTATA_ACCOUNTS/$ACCOUNT_NAME` (default `/root/.altastata/accounts/...`).

Use an NNPA-capable host for faster inference. **Faster models:** set **`HF_LLM_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct`** for roughly 2–3× faster inference than TinyLlama with still usable RAG answers (360M params, instruction-tuned). **`HF_LLM_MODEL=gpt2`** is fastest but gives weak quality. On **small VMs (8 GB RAM)** use **gpt2** to avoid OOM, or **watsonx** for better answers without a local model.

**scikit-learn on IBM Z:** The base image **icr.io/ibmz/ibmz-accelerated-for-pytorch** is PyTorch and zDNN only; it does **not** include scikit-learn, so our `pip install -r requirements.txt` builds it from source (slow on s390x). The Dockerfile caches that layer until `requirements.txt` changes. To confirm the base has no scikit-learn: `docker run --rm icr.io/ibmz/ibmz-accelerated-for-pytorch:1.3.0 pip list | grep -i scikit`. For pre-built ML stacks on Z, see [IBM Cloud Pak for Data](https://www.ibm.com/docs/en/solution-assurance?topic=solutions-set-up-cloud-pak-data-z-linuxone) (OpenShift on Z + entitlement required).

### Same stack on Mac and IBM 390 (Transformers)

To run **the same RAG stack** on your Mac, in Docker, and on IBM 390, use the **Hugging Face Transformers** LLM backend. One codebase, one config, one default model that works on all platforms.

**Set in `.env` (Mac, Docker, and 390):**

```bash
LLM_PROVIDER=transformers
# Default is SmolLM2-360M-Instruct (faster than TinyLlama on CPU); works on Mac, Docker, and IBM 390
# HF_LLM_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct
# HF_LLM_MAX_NEW_TOKENS=80
```

**On your Mac:** Install deps (including `transformers`), run `python index_local.py` then `python web_app.py`. The app loads the model and runs it locally.

**In Docker:** `docker compose up -d app` uses the same default (SmolLM2-360M). No extra env needed.

**On IBM 390:** Use a **Docker image that has PyTorch for s390x** (e.g. IBM’s **ibmz-accelerated-for-pytorch** from [IBM Z Open Source Hub](https://ibm.github.io/ibm-z-oss-hub/containers/ibmz-accelerated-for-pytorch.html), or build our Dockerfile on an s390x host). Install the repo and open_llm requirements, set the same env vars, then run `index_local.py` and `web_app.py` inside the container. The model is pulled from Hugging Face at first run; no separate “model container” for 390.

**Summary:** `LLM_PROVIDER=transformers` + default **SmolLM2-360M-Instruct** gives you one stack for Mac, Docker, and IBM 390 (faster than TinyLlama on CPU). For **fastest** (weak quality): `HF_LLM_MODEL=gpt2`. For larger/better quality: `HF_LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

## Files

| File | Purpose |
|------|--------|
| `config.py` | Env-based config (AltaStata, vector store, Ollama, web, chunking). |
| `indexer.py` | Index AltaStata docs into simple store + store chunks in AltaStata. |
| `query_rag.py` | RAG query: Vector retrieval + parallel AltaStata chunk fetch + LLM (Ollama/Transformers/MLX). |
| `web_app.py` | FastAPI app: GET `/` (UI), POST `/query` (JSON). |
| `templates/index.html` | Simple query form and result display. |
| `Dockerfile` | Image: AltaStata + open_llm deps + web app. |
| `docker-compose.yml` | Ollama + app; optional Caddy for HTTPS. |
| `.env.example` | Example environment variables. |

## Comparison with GCP (Vertex) RAG

| | Open LLM (this) | GCP (Vertex) |
|--|------------------|---------------|
| **Vector store** | Simple (numpy) | Vertex AI Vector Search |
| **Embeddings** | HuggingFace (local) | Vertex text-embedding-004 |
| **LLM** | Ollama (local) | Gemini 2.5 Flash |
| **Cost** | Free (your hardware) | Pay per use |
| **Data** | AltaStata (same) | AltaStata (same) |
| **Web UI** | FastAPI (HTTP/HTTPS) | CLI / custom |

Same patterns: AltaStata for encrypted storage, LangChain for chunking and orchestration, chunks stored in AltaStata and referenced by metadata in the vector store.

## See also

- Parent [RAG README](../README.md) and [QUICKSTART_RAG.md](../QUICKSTART_RAG.md) for the GCP event-driven flow.
- [CHUNKING_STRATEGIES.md](../CHUNKING_STRATEGIES.md) for chunk size and overlap.
