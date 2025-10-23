# Vertex AI RAG with AltaStata - Setup Guide

This guide shows how to run the Vertex AI RAG demo that demonstrates the exact architecture recommended for insurance companies.

## Architecture

```
AltaStata (encrypted storage)
    ↓
LangChain document loaders (fsspec integration)
    ↓
Vertex AI Embeddings (textembedding-gecko)
    ↓
FAISS Vector Store
    ↓
Vertex AI Gemini (generation)
    ↓
Full RAG with citations
```

## Prerequisites

### 1. Google Cloud Setup

**Create a GCP Project:**
```bash
gcloud projects create your-project-id
gcloud config set project your-project-id
```

**Enable Required APIs:**
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable compute.googleapis.com
```

**Authenticate:**
```bash
# Option 1: Application Default Credentials (recommended for development)
gcloud auth application-default login

# Option 2: Service Account (recommended for production)
gcloud iam service-accounts create vertex-ai-rag \
    --display-name="Vertex AI RAG Service Account"

gcloud iam service-accounts keys create ~/vertex-ai-key.json \
    --iam-account=vertex-ai-rag@your-project-id.iam.gserviceaccount.com

export GOOGLE_APPLICATION_CREDENTIALS="$HOME/vertex-ai-key.json"
```

**Grant Permissions:**
```bash
# For your user account or service account
gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:vertex-ai-rag@your-project-id.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

### 2. Python Dependencies

**Install packages:**
```bash
pip install altastata fsspec langchain langchain-google-vertexai langchain-community faiss-cpu google-cloud-aiplatform
```

**Or use requirements file:**
```bash
cat > requirements-vertex.txt <<EOF
altastata
fsspec
langchain>=0.1.0
langchain-google-vertexai
langchain-community
faiss-cpu
google-cloud-aiplatform>=1.38.0
EOF

pip install -r requirements-vertex.txt
```

### 3. Environment Variables

```bash
# Set your GCP project ID
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Optional: Set region (defaults to us-central1)
export GOOGLE_CLOUD_LOCATION="us-central1"

# For insurance companies: Use EU region for GDPR compliance
# export GOOGLE_CLOUD_LOCATION="europe-west1"
```

## Running the Test

```bash
cd rag-example
python test_rag_vertex.py
```

## Expected Output

```
🚀 Testing RAG Pipeline with AltaStata + Vertex AI
================================================================================

0️⃣  Checking Google Cloud configuration...
✅ Using GCP Project: your-project-id
✅ Using GCP Location: us-central1

1️⃣  Initializing AltaStata connection...
✅ AltaStata initialized (encrypted storage ready)

2️⃣  Uploading sample documents to encrypted storage...
   ✅ Uploaded: company_policy.txt - OPERATION_STATE_SUCCEEDED
   ...

5️⃣  Creating embeddings with Vertex AI...
   ✅ Embedding dimension: 768

8️⃣  Testing RAG queries with Gemini...
================================================================================

📊 Query 1: What are the password requirements?
--------------------------------------------------------------------------------
🤖 Gemini Answer:
Passwords must be at least 12 characters long and contain a mix of uppercase and 
lowercase letters, numbers, and special characters. Passwords must be changed every 
90 days, and the last 10 passwords cannot be reused. Multi-factor authentication 
(MFA) must be enabled for all accounts.

📚 Sources (3 documents):
   [1] security_guidelines.txt
   [2] security_guidelines.txt
   [3] remote_work_policy.txt
```

## Troubleshooting

### Error: "Vertex AI API not enabled"
```bash
gcloud services enable aiplatform.googleapis.com
```

### Error: "Permission denied"
```bash
# Check your authentication
gcloud auth application-default login

# Or check service account permissions
gcloud projects get-iam-policy your-project-id
```

### Error: "GOOGLE_CLOUD_PROJECT not set"
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### Error: "Model not found" or "Region not supported"
```bash
# Use supported regions for Gemini
export GOOGLE_CLOUD_LOCATION="us-central1"  # or europe-west1, asia-southeast1
```

## Production Deployment

### For Large Scale (>100K documents)

Replace FAISS with **Vertex AI Matching Engine**:

```python
from google.cloud import aiplatform

# Create Matching Engine index
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="legal-docs-index",
    dimensions=768,  # textembedding-gecko dimension
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
)

# Deploy index endpoint
index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="legal-docs-endpoint",
    public_endpoint_enabled=True,
)

index_endpoint.deploy_index(
    index=index,
    deployed_index_id="legal_docs_deployed",
)
```

### For Maximum Security

Deploy on **GKE with Confidential Computing**:

```bash
# Create confidential GKE cluster
gcloud container clusters create secure-rag \
    --enable-confidential-nodes \
    --machine-type=n2d-standard-4 \
    --num-nodes=3 \
    --zone=us-central1-a
```

## Cost Estimates

### Vertex AI Pricing (as of 2024)

**Embeddings:**
- textembedding-gecko: $0.00025 per 1K characters
- 1M characters ≈ $0.25

**Gemini:**
- gemini-2.5-flash: $0.000125 per 1K characters (input), $0.000375 per 1K chars (output)
- gemini-1.5-pro: $0.00125 per 1K characters (input), $0.005 per 1K chars (output)

**Example:**
- 10,000 documents × 2KB average = 20MB
- Embedding cost: ~$5
- 1,000 queries/day with Gemini Flash: ~$1-2/day

**Compare to RAG Engine:**
- Fully managed, but locked to GCP buckets
- Cannot work with AltaStata
- Similar pricing for underlying models

## What This Demo Proves

✅ **AltaStata + Vertex AI is totally viable**  
✅ **2-4 weeks development time** (not 3-6 months)  
✅ **Full control over pipeline** (chunking, retrieval, generation)  
✅ **Zero-trust security** (documents encrypted in AltaStata)  
✅ **Production-ready** (scales with Matching Engine)

## For Insurance Companies

This architecture solves:
1. ✅ Secure data transfer (Azure → AltaStata → GCP)
2. ✅ No VPN needed (encrypted messaging)
3. ✅ No public endpoints (event-driven)
4. ✅ User-level authentication (via AltaStata)
5. ✅ GDPR compliance (EU region + encryption)
6. ✅ Vertex AI capabilities (embeddings + Gemini)

Without requiring:
- ❌ Direct GCP bucket access
- ❌ Building RAG from scratch
- ❌ 3-6 months of engineering

## Next Steps

1. **Customize chunking strategy** for legal documents
   - See `CHUNKING_STRATEGIES.md` for detailed guidance
   - Recommended for legal docs: chunk_size=2000-2500
2. **Add conversation memory** for multi-turn dialogues
3. **Implement user authentication** (SSO via Azure AD)
4. **Deploy to GKE** with Confidential Computing
5. **Integrate with Daisy/Squeezy** workflows
6. **Add audit logging** for compliance

## Resources

- [LangChain Vertex AI Docs](https://python.langchain.com/docs/integrations/platforms/google)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [AltaStata Documentation](../README.md)
- [Insurance Company Architecture Analysis](./Insurance_Company_Use_Case.md)

