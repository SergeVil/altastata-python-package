# RAG with Google Vertex AI + AltaStata

Production-ready RAG pipeline using Google Cloud Vertex AI services with AltaStata's encrypted storage.

## Quick Start

**📖 See [QUICKSTART.md](QUICKSTART.md) for detailed setup guide**

### TL;DR

```bash
# Install
pip install altastata fsspec langchain langchain-google-vertexai faiss-cpu

# Configure
export GOOGLE_CLOUD_PROJECT="altastata-coco"
gcloud auth application-default login

# Run
cd ..
python test_rag_vertex.py
```

## What's in This Folder

### Working Demo

- **`../test_rag_vertex.py`** - Complete RAG pipeline (located in parent directory)
  - AltaStata encrypted storage (Azure/AWS backend)
  - Vertex AI Text Embeddings (text-embedding-004, 768-dim)
  - Gemini 2.5 Flash for answer generation
  - Full RAG chain with source citations
  - Automatic cleanup

### Documentation

- **`QUICKSTART.md`** - Quick start guide with model explanations
- **`VERTEX_AI_SETUP.md`** - Complete setup guide
  - GCP project setup and authentication
  - Environment configuration
  - Cost estimates
  - Production deployment options
  - Troubleshooting

- **`EMBEDDINGS_AND_METADATA.md`** - Technical deep-dive
  - How embeddings work with Vertex AI
  - Metadata preservation through chunking
  - Retrieving original documents from AltaStata
  - Production metadata examples

- **`Insurance_Company_Use_Case.md`** - Real-world architecture
  - Azure → AltaStata → GCP use case
  - Two implementation paths (Summarization vs RAG)
  - Cost/benefit analysis
  - Decision framework

## Architecture

```
Azure/AWS Document Storage
    ↓
AltaStata (end-to-end encrypted storage)
    ↓
LangChain document loaders (via fsspec)
    ↓
Vertex AI Text Embeddings (text-embedding-004, 768-dim)
    ↓
FAISS Vector Store (or Vertex AI Matching Engine)
    ↓
Gemini 2.5 Flash (LLM with 1M+ token context)
    ↓
Answers with source citations
```

## Key Features

- ✅ **Zero-Trust Security** - Documents encrypted in AltaStata
- ✅ **Vertex AI Integration** - Embeddings, Gemini, and other GCP services
- ✅ **LangChain Orchestration** - Pre-built RAG components
- ✅ **Production-Ready** - Scales from POC to enterprise
- ✅ **GDPR Compliant** - EU regions, encrypted storage

## Quick Example

```python
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# 1. Load documents from AltaStata (encrypted storage)
documents = load_from_altastata()

# 2. Create embeddings with Vertex AI
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-004",  # 768-dimensional vectors
    project="altastata-coco",
    location="us-central1"
)
vectorstore = FAISS.from_documents(chunks, embeddings)

# 3. Initialize Gemini 2.5 Flash
vertexai.init(project="altastata-coco", location="us-central1")
model = GenerativeModel("gemini-2.5-flash")

# 4. Retrieve and generate answer
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.get_relevant_documents("What are the password requirements?")

# Build context and query
context = "\n\n".join([doc.page_content for doc in relevant_docs])
prompt = f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: What are the password requirements?"

response = model.generate_content(prompt)
print(response.text)  # Answer from Gemini
print(relevant_docs)  # Source documents for citations
```

## Implementation Timeline

| Use Case | Timeline | Complexity |
|----------|----------|------------|
| **Document Summarization** | 1-2 weeks | Low - Direct Gemini API calls |
| **RAG System (FAISS)** | 2-4 weeks | Medium - LangChain + Vertex AI |
| **RAG at Scale (Matching Engine)** | 4-6 weeks | Medium-High - Managed vector DB |

## Vector Store Options

### FAISS (Development)
- **Best for:** < 100K documents, quick POC
- **Cost:** Free
- **Setup:** 5 minutes
- **Location:** In-memory or local disk

### Chroma (Production)
- **Best for:** 100K - 1M documents
- **Cost:** Free (self-hosted)
- **Setup:** 15 minutes
- **Location:** Persistent storage

### Vertex AI Matching Engine (Enterprise)
- **Best for:** > 1M documents, high QPS
- **Cost:** ~$0.36/hour/shard
- **Setup:** 30-45 minutes (one-time)
- **Location:** Managed by Google Cloud

## Cost Estimate

### Monthly Costs (Example: 10K documents, 1K queries/day)

**Embeddings:**
- 10K documents × 2KB average = 20MB text
- Cost: ~$5 (one-time for ingestion)

**Gemini:**
- 1K queries/day × 30 days = 30K queries
- Cost: ~$30-60/month (depends on input/output length)

**Vector Store:**
- FAISS: Free
- Chroma: Free (self-hosted)
- Matching Engine: ~$260/month (1 shard)

**Total:** $35-65/month (FAISS) or $295-325/month (Matching Engine)

## Production Checklist

Before deploying to production:

- [ ] Set up GCP project with billing
- [ ] Enable Vertex AI APIs
- [ ] Configure EU region for GDPR (if needed)
- [ ] Set up service accounts with minimal permissions
- [ ] Implement error handling and retry logic
- [ ] Add monitoring and logging
- [ ] Set up alerts for quota limits
- [ ] Optimize chunk sizes for your documents (see `../CHUNKING_STRATEGIES.md`)
- [ ] Test with production data volume
- [ ] Set up CI/CD pipeline
- [ ] Deploy to GKE or Cloud Run
- [ ] Configure autoscaling

## Troubleshooting

See `VERTEX_AI_SETUP.md` for detailed troubleshooting guide.

**Common Issues:**

1. **"Vertex AI API not enabled"**
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

2. **"Permission denied"**
   ```bash
   gcloud auth application-default login
   ```

3. **"Model not found"**
   - Use supported regions: us-central1, europe-west1, asia-southeast1

## Related Documentation

- `../README.md` - Main RAG examples overview
- `../CHUNKING_STRATEGIES.md` - Universal chunking guide (applies to all RAG systems)
- `../test_rag.py` - Basic RAG with HuggingFace (for comparison)
- `EMBEDDINGS_AND_METADATA.md` - How embeddings work with Vertex AI (in this folder)

## Support

For Vertex AI-specific questions:
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [LangChain Vertex AI Integration](https://python.langchain.com/docs/integrations/platforms/google)

For AltaStata questions:
- See main repository [README](../../README.md)

