# Securing RAG with AltaStata

Complete examples and documentation for building secure RAG (Retrieval-Augmented Generation) systems with AltaStata's encrypted storage.

## Quick Start

### Installation

```bash
pip install altastata fsspec langchain langchain-community sentence-transformers faiss-cpu
```

### Run the Test

```bash
cd rag-example
python test_rag.py
```

This will:
- ✅ Upload sample documents to encrypted storage
- ✅ Load documents via fsspec
- ✅ Create embeddings and vector store
- ✅ Perform semantic search on 5 different queries
- ✅ Show similarity scores
- ✅ Clean up test data

## Files in This Directory

### Working Examples

- **`test_rag.py`** ⭐ - Complete, production-ready RAG pipeline
  - Uploads documents to encrypted AltaStata storage
  - Loads documents using fsspec
  - Creates vector embeddings with sentence-transformers
  - Demonstrates semantic search with FAISS
  - Includes 5 real query examples with output
  - Automatic cleanup

### Documentation

- **`SECURING_RAG_WITH_ALTASTATA.md`** - Practical implementation guide
  - Installation instructions
  - Code examples
  - Best practices
  - Real test output

- **`RAG_SECURITY_ARCHITECTURE.md`** - Security architecture deep dive
  - Multi-layer security model
  - Data flow diagrams
  - Compliance information (GDPR, HIPAA, SOC 2)
  - Deployment models

### Sample Data

- **`sample_documents/`** - Sample policy documents for testing
  - `company_policy.txt` - Data retention policy
  - `security_guidelines.txt` - Security best practices
  - `remote_work_policy.txt` - Remote work guidelines
  - `ai_usage_policy.txt` - AI tool usage policy

## Test Output Example

```
🚀 Testing RAG Pipeline with AltaStata fsspec
================================================================================

1️⃣  Initializing AltaStata connection...
✅ AltaStata initialized

2️⃣  Uploading sample documents to encrypted storage...
   ✅ Uploaded: company_policy.txt - DONE
   ✅ Uploaded: security_guidelines.txt - DONE
   ✅ Uploaded: remote_work_policy.txt - DONE
   ✅ Uploaded: ai_usage_policy.txt - DONE

3️⃣  Loading documents via fsspec...
   ✅ Loaded: company_policy.txt (1024 chars)
   ✅ Total documents loaded: 4

4️⃣  Splitting documents into chunks...
✅ Created 12 text chunks

5️⃣  Creating embeddings and vector store...
✅ Vector store created successfully

6️⃣  Testing RAG queries...
================================================================================

📊 Query 1: What are the password requirements?
Found 2 relevant chunks:
[Chunk 1] (from: security_guidelines.txt)
Enterprise Security Guidelines
Password Requirements:
- Minimum 12 characters
- Mix of uppercase, lowercase, numbers, and special characters
- Change passwords every 90 days
...
```

## Key Features

### Security
- **End-to-End Encryption** - AES-256 per file, zero-knowledge architecture
- **Automatic Versioning** - Immutable audit trail for compliance
- **Access Control** - Account-based authentication
- **Confidential Computing** - Optional hardware-level memory protection

### Integration
- **LangChain** - Seamless integration via fsspec
- **Multiple Frameworks** - Works with LlamaIndex, Hugging Face, etc.
- **Multi-Cloud** - AWS, GCP, Azure, IBM, MiniIO support

### Compliance
- ✅ GDPR compliant
- ✅ HIPAA compliant  
- ✅ SOC 2 compliant

## Next Steps

After running the test, you can:

1. **Integrate with LLM** - Add OpenAI, Anthropic, or local models
   ```python
   from langchain.chains import RetrievalQA
   qa_chain = RetrievalQA.from_chain_type(
       llm=your_llm,
       retriever=retriever
   )
   ```

2. **Add Conversation Memory** - Multi-turn dialogues
   ```python
   from langchain.memory import ConversationBufferMemory
   memory = ConversationBufferMemory()
   ```

3. **Deploy with Confidential Computing**
   ```bash
   cd ../confidential-gke
   ./setup-cluster.sh
   ```

## Documentation

- **[SECURING_RAG_WITH_ALTASTATA.md](SECURING_RAG_WITH_ALTASTATA.md)** - Complete implementation guide
- **[RAG_SECURITY_ARCHITECTURE.md](RAG_SECURITY_ARCHITECTURE.md)** - Security architecture
- **[test_rag.py](test_rag.py)** - Working code with examples
- **[../fsspec-example/](../fsspec-example/)** - fsspec integration examples
- **[../README.md](../README.md)** - Main AltaStata documentation

## Support

For questions or issues:
- Review the documentation files in this directory
- Check the main [AltaStata README](../README.md)
- See [fsspec examples](../fsspec-example/) for basic usage

---

**Secure your RAG. Secure your future. 🔒**

