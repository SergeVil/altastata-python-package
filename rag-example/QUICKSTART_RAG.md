# Quick Start: Event-Driven RAG with AltaStata + Vertex AI

## 📋 What You Get

4 simple scripts for a complete event-driven RAG system:

1. **`bob_indexer.py`** - Bob listens for events and indexes documents
2. **`alice_upload_docs.py`** - Alice uploads documents from `sample_documents/`
3. **`bob_query.py`** - Query the indexed documents
4. **`cleanup.py`** - Clean up everything

## 🚀 Quick Start

### 1. One-Time Setup (Vertex AI Vector Search)

**First time only** - Create the Vertex AI infrastructure (~20-40 minutes):

```bash
cd rag-example

# Set your GCP project
export GOOGLE_CLOUD_PROJECT="altastata-coco"
gcloud auth application-default login

# Create Vertex AI Vector Search index and endpoint (one-time, 20-40 min)
python setup_vertex_search.py
```

This creates:
- Vertex AI Matching Engine Index (768-dim vectors)
- Index Endpoint (deployed on `e2-standard-2` VMs)
- `.vertex_config` file with resource IDs

**Subsequent runs**: Skip this step - the infrastructure persists.

### 2. Start Bob's Indexer (Terminal 1)

```bash
python bob_indexer.py
```

**Output:**
```
🤖 BOB INDEXER - Vertex AI Vector Search (Event-Driven)
================================================================================

📍 Project: altastata-coco, Location: us-central1

1️⃣  Loading Vertex AI Vector Search configuration...
✅ Loaded config
   Index: 2502323538074009600

2️⃣  Initializing Vertex AI...
✅ Embeddings ready (text-embedding-004, 768-dim)
✅ Connected to Vertex AI Vector Search

3️⃣  Connecting Bob...
✅ Callback server started on 127.0.0.1:25334
✅ Bob connected

4️⃣  Registering event listener...
✅ Listening for SHARE events

================================================================================
🎧 BOB IS LISTENING...
================================================================================

💡 Run: python alice_upload_docs.py
⏳ Waiting for events... (Ctrl+C to stop)
```

### 3. Upload Documents (Terminal 2)

```bash
python alice_upload_docs.py
```

**Output:**
```
📤 ALICE - Upload & Share Documents
================================================================================

1️⃣  Connecting Alice...
✅ Alice connected

2️⃣  Loading documents from sample_documents/...
   ✅ Loaded: company_policy.txt
   ✅ Loaded: security_guidelines.txt
   ✅ Loaded: remote_work_policy.txt
   ✅ Loaded: ai_usage_policy.txt

3️⃣  Creating 4 files in parallel...
   ✅ company_policy.txt: DONE
   ✅ security_guidelines.txt: DONE
   ✅ remote_work_policy.txt: DONE
   ✅ ai_usage_policy.txt: DONE

⏳ Waiting for files to be fully stored...

4️⃣  Sharing 4 files with bob123...
   ✅ company_policy.txt: Shared (1 version)
   ✅ security_guidelines.txt: Shared (1 version)
   ✅ remote_work_policy.txt: Shared (1 version)
   ✅ ai_usage_policy.txt: Shared (1 version)

================================================================================
✅ All documents shared!
================================================================================
```

**Bob's Terminal will show (for each document):**
```
================================================================================
🔔 EVENT RECEIVED: SHARE
================================================================================
📄 File: RAGDocs/policies/company_policy.txt✹alice222_1761252449182
   1️⃣  Reading file...
   ✅ Loaded: 1024 chars
   2️⃣  Chunking...
   ✅ Created 1 chunks
   3️⃣  Storing chunks in AltaStata...
      📄 Chunk 1/1 (1024 chars): Enterprise Security Guidelines  Version 2.0...
      ✅ Stored: chunks/RAGDocs/policies/company_policy.txt_0.txt
   4️⃣  Generating embeddings...
   5️⃣  Upserting to Vertex AI Vector Search...
Upserting datapoints MatchingEngineIndex index: projects/177851330934/...
MatchingEngineIndex index Upserted datapoints. Resource name: projects/177851330934/...
   ✅ Indexed 1 chunks to Vertex AI Vector Search!
   💡 Chunks stored in AltaStata at: chunks/RAGDocs_policies_*.txt
================================================================================
```

### 4. Query Documents

```bash
python bob_query.py
```

**Choose mode:**
- **Demo mode (2)**: Runs 4 example queries
- **Interactive mode (1)**: Ask your own questions

**Example:**
```
🔍 BOB QUERY - Ask Questions
================================================================================

📍 Project: altastata-coco, Location: us-central1

1️⃣  Loading indexed documents...
✅ Loaded vector store

2️⃣  Initializing Gemini...
✅ Gemini 2.5 Flash ready

Select mode:
  1. Interactive (ask your own questions)
  2. Demo (run 4 example queries)
Choice (1 or 2): 2

────────────────────────────────────────────────────────────────────────────────
📊 QUERY 1/4
────────────────────────────────────────────────────────────────────────────────
❓ What are the password requirements?

🤖 ANSWER:
   Passwords must be at least 12 characters long, contain a mix of uppercase
   and lowercase letters, numbers, and special characters, and be changed
   every 90 days. Password reuse for the last 10 passwords is not allowed.

📚 SOURCES:
   1. 📄 security_guidelines.txt (chunk 0)
      └─ Password Requirements: - Minimum 12 characters - Mix of uppercase, lowercase...
```

### 5. Cleanup

```bash
python cleanup.py
```

**Two cleanup modes:**

**Option 1 - Quick cleanup (instant):**
- Deletes AltaStata files
- Deletes metadata (`/tmp/bob_rag_metadata.json`)
- **Keeps Vertex AI infrastructure** (index + endpoint remain ready)

**Option 2 - Full cleanup (requires 30 min to recreate):**
- Everything from Option 1
- **Deletes Vertex AI Vector Search index and endpoint**
- Requires running `setup_vertex_search.py` again (~30 min)

**Recommendation:** Use Option 1 for testing iterations!

## 🎯 Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Alice uploads docs from sample_documents/ (parallel batch)          │
│    → Creates files in AltaStata (encrypted)                            │
│    → Shares with Bob                                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Bob receives SHARE events (SecureCloudEventProcessor)               │
│    → Reads from encrypted storage (fsspec)                             │
│    → Chunks document (RecursiveCharacterTextSplitter, 4000 chars)     │
│    → Stores chunks in AltaStata (encrypted, one file per chunk)        │
│    → Generates embeddings (Vertex AI text-embedding-004, 768-dim)       │
│    → Upserts to Vertex AI Vector Search (Matching Engine)              │
│    → Stores chunk_path in metadata (restricts)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Query interface (bob_query.py)                                       │
│    → Connects to Vertex AI Vector Search endpoint                     │
│    → Finds similar vectors (k-NN search on e2-standard-2 VMs)         │
│    → Retrieves chunks directly from AltaStata (using chunk_path)       │
│    → Queries Gemini 2.5 Flash with context                            │
│    → Returns answer + source citations (with chunk index)              │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🔧 Technical Stack

- **Storage**: AltaStata (end-to-end encrypted)
- **Embeddings**: Vertex AI `text-embedding-004` (768-dim)
- **Vector Store**: Vertex AI Vector Search (Matching Engine on e2-standard-2 VMs)
- **LLM**: Gemini 2.5 Flash
- **Orchestration**: LangChain document processing
- **Events**: AltaStata event listeners (Py4J callbacks + SecureCloudEventProcessor)

## 📝 Files

| File | Purpose |
|------|---------|
| `setup_vertex_search.py` | One-time setup: Create Vertex AI Vector Search infrastructure |
| `alice_upload_docs.py` | Upload & share documents from `sample_documents/` (parallel batch) |
| `bob_indexer.py` | Event listener + document indexer (Vertex AI Vector Search) |
| `bob_query.py` | Query interface (interactive or demo) |
| `cleanup.py` | Clean up storage + metadata (option to delete Vertex AI resources) |

## 🚀 Production Upgrades

To scale to production:

1. **Scale Vertex AI Vector Search**:
   - Increase replicas (`min_replica_count`, `max_replica_count`)
   - Use larger machine types (`e2-highmem-2`, `n1-standard-16`)
   - Increase shard size (`SHARD_SIZE_MEDIUM` or `SHARD_SIZE_LARGE`)

2. **Deploy on GKE with Confidential Computing**:
   - See `confidential-gke/` for deployment
   - Run Bob's indexer as a Kubernetes Deployment

3. **Add authentication and audit logging**:
   - Track who indexes and queries documents
   - Monitor Vertex AI API usage and costs

## 💡 Notes

- Bob must be running (`bob_indexer.py`) before Alice uploads
- Vectors stored on Vertex AI (cloud), chunks stored in AltaStata (encrypted)
- Uses Azure accounts (`azure.rsa.bob123`, `azure.rsa.alice222`)
- Bob uses callback port `25334`, Alice uses gateway port `25555`

## ❓ Troubleshooting

**"No documents indexed yet"**
- Run `bob_indexer.py` first
- Then run `alice_upload_docs.py`
- Wait for indexing to complete

**"Event not received"**
- Check Bob's `.user.properties` has `sqs-interval` set
- Ensure Bob and Alice use different ports
- Wait 15-20 seconds for event processing

**"Vertex AI API error"**
- Check: `export GOOGLE_CLOUD_PROJECT="altastata-coco"`
- Run: `gcloud auth application-default login`
- Verify project has Vertex AI API enabled

