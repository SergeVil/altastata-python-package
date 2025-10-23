# Quick Start: RAG Pipeline with Vertex AI

Complete guide to running `test_rag_vertex.py` - a production-ready RAG pipeline combining AltaStata encrypted storage with Google Vertex AI.

---

## 🎯 What This Does

Demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline:

1. **Stores documents** in AltaStata's encrypted storage (Azure/AWS backend)
2. **Creates vector embeddings** using Vertex AI
3. **Retrieves relevant documents** based on user queries
4. **Generates answers** using Gemini 2.5 Flash with full source citations

---

## 🤖 AI Models Used

### **1. Vertex AI Text Embeddings** (`text-embedding-004`)
- **Purpose**: Converts text into 768-dimensional vectors for similarity search
- **What it does**: Takes your document chunks and creates mathematical representations that capture their meaning
- **Why we use it**: Enables semantic search - finds relevant documents even when exact keywords don't match
- **Location**: Runs on Google Cloud (us-central1 by default)

### **2. Gemini 2.5 Flash** (`gemini-2.5-flash`)
- **Purpose**: Large Language Model for generating answers
- **What it does**: Reads retrieved documents and generates concise, accurate answers to user questions
- **Why we use it**: Latest production model with large context window (1M+ tokens) and fast response
- **Location**: Runs on Google Cloud (us-central1 by default)

---

## ⚙️ Configuration

### **Required Environment Variables**

```bash
# Your Google Cloud Project (must have Vertex AI access)
export GOOGLE_CLOUD_PROJECT="altastata-coco"

# Optional: Region for GDPR compliance
export GOOGLE_CLOUD_LOCATION="us-central1"  # or "europe-west1"
```

### **AltaStata Account Path**

Edit line 77 in `test_rag_vertex.py`:

```python
altastata_functions = AltaStataFunctions.from_account_dir(
    '/Users/YOUR_USERNAME/.altastata/accounts/azure.rsa.bob123'  # ← Update this
)
```

**Supported accounts**:
- `azure.rsa.bob123` - Azure storage backend
- `amazon.rsa.bob123` - AWS S3 storage backend

---

## 🚀 Running the Demo

### **1. Install Dependencies**

```bash
pip install altastata fsspec langchain langchain-google-vertexai faiss-cpu
```

### **2. Authenticate with Google Cloud**

```bash
gcloud auth application-default login
```

### **3. Run the Pipeline**

```bash
export GOOGLE_CLOUD_PROJECT="altastata-coco"
cd rag-example
python test_rag_vertex.py
```

### **Expected Output**

```
🚀 Testing RAG Pipeline with AltaStata + Vertex AI
================================================================================

0️⃣  Checking Google Cloud configuration...
✅ Using GCP Project: altastata-coco
✅ Using GCP Location: us-central1

1️⃣  Initializing AltaStata connection...
✅ AltaStata initialized (encrypted storage ready)

2️⃣  Loading and uploading sample documents...
   ✅ Uploaded: company_policy.txt - DONE
   ✅ Uploaded: security_guidelines.txt - DONE
   ...

5️⃣  Creating embeddings with Vertex AI...
   ✅ Embedding dimension: 768
   ✅ Vector store created successfully

6️⃣  Initializing Vertex AI with native SDK...
✅ Gemini 2.5 Flash initialized

8️⃣  Testing RAG queries with Gemini...
───────────────────────────────────────────────────────────────────────────
📊 QUERY 1/4
───────────────────────────────────────────────────────────────────────────
❓ What are the password requirements?

🤖 ANSWER:
   The password requirements are:
   *   Minimum 12 characters
   *   Mix of uppercase, lowercase, numbers, and special characters
   ...

📚 SOURCES (3 documents retrieved):
   1. 📄 security_guidelines.txt
```

---

## 🏗️ Architecture Flow

```
User Query
    ↓
┌─────────────────────────────────────┐
│  Vertex AI Text Embeddings          │  Convert query to 768-dim vector
│  (text-embedding-004)               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  FAISS Vector Store                 │  Find 3 most similar documents
│  (Local similarity search)          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Gemini 2.5 Flash                   │  Generate answer from context
│  (LLM with 1M token context)        │
└─────────────────────────────────────┘
    ↓
Answer with Source Citations
```

**Storage Layer**: AltaStata (encrypted Azure/AWS storage)

---

## 📊 What Gets Tested

1. **Document Upload** - 4 sample policy documents encrypted and stored
2. **Document Retrieval** - Files loaded via fsspec from encrypted storage
3. **Chunking** - Documents split into 4000-char chunks (optimized for Gemini)
4. **Embedding Generation** - Each chunk converted to 768-dim vector
5. **Vector Search** - Find top 3 relevant chunks per query
6. **Answer Generation** - 4 test queries with full citations
7. **Cleanup** - All test data deleted automatically

---

## 🔧 Troubleshooting

### **Error: 404 Publisher Model not found**

**Problem**: Wrong model name or no access to Vertex AI.

**Solution**:
1. Make sure you're using `gemini-2.5-flash` (not `gemini-1.5-flash`)
2. Verify project has Vertex AI API enabled:
   ```bash
   gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT
   ```

### **Error: No module named 'faiss'**

**Solution**: Install FAISS for CPU:
```bash
pip install faiss-cpu
```

### **Error: AltaStata connection failed**

**Solution**: Check your account path is correct and password is set:
```python
altastata_functions.set_password("123")  # Line 79
```

---

## 💡 Key Features

✅ **100% Vertex AI** - Embeddings + LLM from Google Cloud  
✅ **Zero-Trust Encryption** - All documents encrypted in AltaStata  
✅ **Multi-Cloud** - Storage on Azure/AWS, AI on GCP  
✅ **Source Citations** - Every answer shows which documents were used  
✅ **Production-Ready** - Error handling, cleanup, proper logging  

---

## 📚 Next Steps

- **Scale Up**: Replace FAISS with Vertex AI Matching Engine for >100K documents
- **Add Auth**: Implement user authentication and audit logging
- **Deploy**: Run on GKE with Confidential Computing
- **Monitor**: Add observability with Cloud Logging/Monitoring

See `VERTEX_AI_SETUP.md` for production deployment guide.

