# Securing RAG with AltaStata

Complete examples and documentation for building secure RAG (Retrieval-Augmented Generation) systems with AltaStata's encrypted storage.

## Quick Start

### Option 1: Basic RAG (HuggingFace - Local)

Fast development with local embeddings:

```bash
pip install altastata fsspec langchain langchain-community sentence-transformers faiss-cpu
python test_rag.py
```

### Option 2: Production RAG (Google Vertex AI) ⭐ RECOMMENDED

Production-ready with Google Cloud:

```bash
# See google_vertexai/ folder for complete documentation
pip install altastata fsspec langchain langchain-google-vertexai langchain-community faiss-cpu google-cloud-aiplatform

export GOOGLE_CLOUD_PROJECT="your-project-id"
gcloud auth application-default login

python test_rag_vertex.py
```

Both demos:
- ✅ Encrypted document storage with AltaStata
- ✅ Load documents via fsspec
- ✅ Create embeddings and vector store
- ✅ Perform semantic search with citations
- ✅ Clean up test data

## What's in This Repository

### 🚀 Getting Started

- **`test_rag.py`** - Basic RAG demo (local HuggingFace)
- **`google_vertexai/`** ⭐ - Production RAG with Vertex AI
  - Complete working demo
  - Setup guides and documentation
  - Real-world use case analysis

### 📚 Documentation

**Core Guides:**
- **`SECURING_RAG_WITH_ALTASTATA.md`** - Implementation guide with best practices
- **`RAG_SECURITY_ARCHITECTURE.md`** - Security model and compliance (GDPR, HIPAA, SOC 2)

**Technical Deep-Dives:**
- **`CHUNKING_STRATEGIES.md`** - Universal document chunking guide (applies to all RAG systems)
- **`PDF_PROCESSING.md`** - How to extract text from PDFs and preserve page numbers

**Google Vertex AI:**
- See **`google_vertexai/`** folder for:
  - Complete Vertex AI demo and setup guide
  - Production deployment documentation
  - How embeddings and metadata work with Vertex AI
  - Real-world insurance company use case
  - Cost estimates and timelines

### Sample Data

- **`sample_documents/`** - Sample policy documents for testing
  - `company_policy.txt` - Data retention policy
  - `security_guidelines.txt` - Security best practices
  - `remote_work_policy.txt` - Remote work guidelines
  - `ai_usage_policy.txt` - AI tool usage policy

## Quick Example

```python
# 1. Upload to AltaStata (encrypted storage)
altastata.upload("policy.pdf")

# 2. Load and create embeddings
from langchain_google_vertexai import VertexAIEmbeddings
embeddings = VertexAIEmbeddings()
# Store in Vertex AI Vector Search (or FAISS for development)
vectorstore = create_vector_store(chunks, embeddings)

# 3. Query with Gemini
from langchain_google_vertexai import VertexAI
qa_chain = RetrievalQA.from_chain_type(
    llm=VertexAI(model_name="gemini-1.5-flash"),
    retriever=vectorstore.as_retriever()
)

answer = qa_chain.invoke({"query": "What are the password requirements?"})
```

## Key Features

- **Zero-Trust Security** - End-to-end encryption, zero-knowledge architecture
- **Vertex AI Integration** - Embeddings, Gemini, and other GCP AI services
- **LangChain Orchestration** - Pre-built components for RAG pipelines
- **Production-Ready** - Scales from POC to enterprise
- **Compliance** - GDPR, HIPAA, SOC 2 compatible

## Production Deployment

1. **Choose Vector Store** (see `google_vertexai/` for details)
   - **Vertex AI Vector Search**: Production (recommended)
   - FAISS: Development/testing only
   - Chroma: Alternative for self-hosted

2. **Optimize Chunk Sizes** (see `CHUNKING_STRATEGIES.md`)
   - With Gemini 1.5: 4000-8000 chars (leverages large context)
   - Legal documents: 6000-8000 chars
   - General docs: 4000-5000 chars

3. **Deploy to GKE** (optional Confidential Computing)
   ```bash
   cd ../confidential-gke
   ./setup-cluster.sh
   ```

## Links

- **[google_vertexai/](google_vertexai/)** - Production RAG with Vertex AI
- **[SECURING_RAG_WITH_ALTASTATA.md](SECURING_RAG_WITH_ALTASTATA.md)** - Implementation guide
- **[RAG_SECURITY_ARCHITECTURE.md](RAG_SECURITY_ARCHITECTURE.md)** - Security architecture
- **[../fsspec-example/](../fsspec-example/)** - fsspec integration examples
- **[../README.md](../README.md)** - Main AltaStata documentation

## Support

For questions or issues:
- Review the documentation files in this directory
- Check the main [AltaStata README](../README.md)
- See [fsspec examples](../fsspec-example/) for basic usage

---

**Secure your RAG. Secure your future. 🔒**

