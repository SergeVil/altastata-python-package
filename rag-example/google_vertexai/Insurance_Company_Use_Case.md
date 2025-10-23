# AltaStata for Insurance Company - Architecture Analysis

## Use Case Overview

**Scenario:** Insurance company needs to process legal documents from Azure Document Management System and send them to GCP Vertex AI for AI processing (summarization or RAG).

**Key Requirements:**
- Secure data transfer between Azure and GCP
- GDPR compliance with EU data residency
- User-level authentication and audit trails
- No plain-text document exposure

## Current Systems

### Squeezy (Legal Assistant)
- **Challenge**: Azure → GCP processing requires secure data transfer
- **Current Plan**: VPN or public internet with OAuth2
- **Workflow**: Document Management (Azure) → Extract text → GCP Bucket → Cloud Function → Vertex AI

### Daisy (Intermediary/Agent/Broker System)  
- **Challenge**: Same Azure → GCP data transfer requirements
- **Current Plan**: SpeedyPlus D4Next exposes backend via public internet

## What AltaStata Solves

| Issue | Solution | Benefit |
|-------|----------|---------|
| **VPN costs** | End-to-end encrypted messaging | Save €1,200-18k/year |
| **WAF needed** | No public Cloud Functions | Save €240-2,400/year |
| **Plain-text bucket exposure** | Real-time encrypted streaming | Zero-trust architecture |
| **Shared accounts** | User-level authentication | Audit compliance |
| **Intrusion detection** | Built-in event logs | Save €11,400/year |

## What AltaStata Does NOT Solve

1. ❌ **Vertex AI region config** - Still need to configure EU regions for GDPR
2. ❌ **GDPR DPIA** - Encryption helps but doesn't eliminate DPIA requirements
3. ❓ **SSO Integration** - Only required if end-users access the system
4. ❌ **Vendor approval** - Insurance company still needs internal approval process

## Architecture Comparison

### WITHOUT AltaStata (Current Plan)
```
Azure Document Management
    ↓
Extract text → Upload to GCP Bucket (plain-text exposure)
    ↓
Cloud Function (public HTTP + OAuth2 + WAF required)
    ↓
Vertex AI processing

Issues: Needs VPN, WAF, plain-text storage, shared accounts
```

### WITH AltaStata
```
Azure Document Management
    ↓
Extract text → Upload to AltaStata (encrypted)
    ↓
Encrypted messaging over internet
    ↓
GCP Cloud Function (private, event-triggered)
    ↓
Download from AltaStata → Process → Vertex AI

Benefits: No VPN, no WAF, zero-trust, user-level audit
```

## Two Implementation Paths

### Path A: Simple Document Summarization

**Use Case:** Process documents one-by-one for summarization

**Architecture:**
```
Document → AltaStata → Cloud Function → Gemini API → Summary
```

**Implementation:**
- Use Gemini API directly (e.g., `gemini-1.5-pro.generateContent()`)
- Simple integration
- **Timeline:** 1-2 weeks

**Perfect for:** Document-by-document processing without search

---

### Path B: RAG System (Search Historical Documents)

**Use Case:** Query across thousands of past legal documents

**Why NOT Vertex AI RAG Engine:**
- ❌ RAG Engine requires direct access to GCS buckets
- ❌ Cannot work with encrypted external storage
- ❌ Must control entire pipeline from ingestion to search

**Solution: LangChain + Vertex AI Components**

**Architecture:**
```python
# 1. Load from AltaStata
from langchain.document_loaders import CustomLoader
documents = loader.load_from_altastata()

# 2. Split into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
chunks = splitter.split_documents(documents)

# 3. Create embeddings with Vertex AI
from langchain_google_vertexai import VertexAIEmbeddings
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

# 4. Store in Vertex AI Vector Search (Matching Engine)
from google.cloud import aiplatform
# Create index and upsert embeddings to Vertex AI Vector Search
# For development/testing, can use FAISS: from langchain_community.vectorstores import FAISS

# 5. Create RAG chain with Gemini
from langchain_google_vertexai import VertexAI
from langchain.chains import RetrievalQA

llm = VertexAI(model_name="gemini-1.5-flash")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_search.as_retriever()  # Using Vertex AI Vector Search
)

# Query!
answer = qa_chain.invoke({"query": "What are the payment terms?"})
```

**What You Get:**
- ✅ Vertex AI Embeddings (textembedding-gecko)
- ✅ Vertex AI Gemini for generation
- ✅ Vertex AI Vector Search (Matching Engine)
- ✅ LangChain orchestration (pre-built components)
- ✅ Full control over pipeline

**Timeline:** 2-4 weeks (NOT 3-6 months!)

**Vector Store Options:**
- **Vertex AI Vector Search** (Matching Engine): Production choice, managed, scalable to billions (recommended)
- **FAISS** (local): Development/testing only, good for <100K documents
- **Chroma** (local): Alternative for smaller deployments

---

## Cost/Benefit Analysis

### What the Insurance Company Gets:
- Stronger security posture (zero-trust, encrypted at rest and in transit)
- Lower infrastructure costs: ~€12,840-21,800/year savings
  - No VPN: €1,200-18,000/year
  - No WAF: €240-2,400/year  
  - No IDS: €11,400/year
- Better compliance (user-level audit, zero-trust architecture)
- Reduced attack surface (no public endpoints, no plain-text storage)
- **Documents encrypted even from Google admins**

### What the Insurance Company Pays:
- **Summarization:** 1-2 weeks integration (~€10k-20k)
- **RAG with LangChain:** 2-4 weeks development (~€20k-40k)
- Additional vendor to manage (AltaStata)
- Service account setup (STS tokens for Azure → AltaStata → GCP)
- Internal compliance approval process

### Is It Worth It?

**YES if:**
- Legal documents are highly sensitive
- Regulatory compliance is critical
- Zero-trust architecture is required
- User-level audit trails are mandatory

**MAYBE if:**
- Standard GCP security is sufficient
- Cost savings outweigh engineering time
- Timeline is not constrained

## Key Technical Decisions

### 1. Authentication: SSO or Service Accounts?

**Service Accounts (Recommended for backend processing):**
- Azure service principal reads containers
- AltaStata API key for uploads
- GCP service account for Cloud Functions
- **No user SSO needed** if fully automated

**User SSO (Required for user-facing apps):**
- Azure AD SSO for employee access
- Federated identity with AltaStata
- Individual accountability
- **Needed if** lawyers/staff manually upload documents

### 2. Chunk Size for Legal Documents

**Recommended:** 1500-2500 characters
- Preserves full legal clauses
- Maintains context across paragraphs
- See `CHUNKING_STRATEGIES.md` for details

### 3. Vector Store Selection

| Documents | Vector Store | Why |
|-----------|-------------|-----|
| **Production** | **Vertex AI Vector Search** | Managed, scalable, fully integrated with Vertex AI |
| Development | FAISS | Fast local testing, <100K documents |
| Alternative | Chroma | Self-hosted persistent storage |

## Questions to Validate

Before implementation, confirm:

1. **Processing model:**
   - ☐ Summarization only (simpler)
   - ☐ RAG system for search (more complex)

2. **Document retention:**
   - ☐ Need to keep originals for audit?
   - ☐ Retention period requirements?

3. **Authentication:**
   - ☐ Backend processing or user-facing?
   - ☐ SSO already implemented?

4. **Scale:**
   - ☐ How many documents?
   - ☐ Expected query volume?

5. **Timeline:**
   - ☐ Deadline for deployment?
   - ☐ POC or production?

## Working Demo

**See the complete implementation:**
- `../test_rag_vertex.py` - Full RAG pipeline with Vertex AI (in parent directory)
- `VERTEX_AI_SETUP.md` - Setup and deployment guide
- `../CHUNKING_STRATEGIES.md` - Document chunking best practices
- `EMBEDDINGS_AND_METADATA.md` - How embeddings preserve document links

## Recommended Approach

### Phase 1: POC (2 weeks)
1. Set up AltaStata with test documents
2. Implement basic Azure → AltaStata → GCP flow
3. Test locally with FAISS for quick validation
4. Use Gemini for simple summarization

### Phase 2: Production (2-4 weeks)
1. Implement full RAG pipeline with LangChain
2. Deploy to Vertex AI Vector Search (Matching Engine)
3. Optimize chunk sizes for legal documents
4. Add user authentication and audit logging
5. Deploy to GKE with monitoring

### Phase 3: Scale
1. Add conversation memory for multi-turn dialogues
2. Integrate with Daisy/Squeezy workflows
3. Implement advanced retrieval strategies
4. Deploy Confidential Computing for maximum security

## Bottom Line

**AltaStata + LangChain + Vertex AI is viable for insurance companies.**

- ✅ **Summarization**: 1-2 weeks, minimal complexity
- ✅ **RAG System**: 2-4 weeks with LangChain
- ✅ **Security**: Zero-trust, GDPR-compliant
- ✅ **Cost Savings**: €12k-21k/year infrastructure savings

**Key Insight:** You CAN use Vertex AI's powerful capabilities (embeddings, Gemini) without using the RAG Engine. LangChain provides the orchestration layer that makes AltaStata and Vertex AI work together seamlessly.
