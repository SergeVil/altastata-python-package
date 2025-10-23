# How Embeddings and Metadata Work Together in RAG

## The Concept

When you create embeddings for RAG, you store **three things** together:

1. **Text chunk** - The actual content
2. **Embedding vector** - Numerical representation (e.g., 768 dimensions)
3. **Metadata** - Where the text came from (file path, page number, etc.)

```
┌─────────────────────────────────────────────────────────────┐
│ Document Chunk                                              │
├─────────────────────────────────────────────────────────────┤
│ Text: "Password requirements: minimum 12 characters..."     │
│                                                             │
│ Embedding: [0.234, -0.456, 0.789, ..., 0.123]  (768 dims) │
│                                                             │
│ Metadata: {                                                │
│   "source": "RAGTest/vertex_ai/security_guidelines.txt",  │
│   "filename": "security_guidelines.txt",                  │
│   "chunk_index": 3                                        │
│ }                                                          │
└─────────────────────────────────────────────────────────────┘
```

## How It Works in Our Code

### Step 1: Load Documents with Metadata

```python
file_path = "RAGTest/vertex_ai/security_guidelines.txt"

doc = Document(
    page_content=content,  # The actual text
    metadata={
        "source": file_path,           # AltaStata path (for retrieval later!)
        "filename": filename           # Human-readable name
    }
)
```

**Key point:** The `source` metadata contains the **AltaStata path** so you can retrieve the original file later!

### Step 2: Split into Chunks (Metadata Preserved!)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,      # Larger for Gemini 1.5's context window
    chunk_overlap=800
)
chunks = text_splitter.split_documents(documents)
```

**Important:** LangChain's `split_documents()` **automatically copies metadata** to each chunk!

```
Original Document:
┌────────────────────────────────────────┐
│ Text: [20000 characters]               │
│ Metadata: {"source": "file.txt"}       │
└────────────────────────────────────────┘
                ↓ split
        ┌───────────────┬───────────────┬───────────────┐
        │ Chunk 1       │ Chunk 2       │ Chunk 3       │
        │ (4000 chars)  │ (4000 chars)  │ (4000 chars)  │
        │ source: file  │ source: file  │ source: file  │
        └───────────────┴───────────────┴───────────────┘
```

> **Note:** See `../CHUNKING_STRATEGIES.md` for optimal chunk sizes for your use case.

### Step 3: Create Embeddings + Store

```python
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
vectorstore = FAISS.from_documents(chunks, embeddings)
```

**What happens:** Each chunk's text, embedding vector (768 dimensions), and metadata are stored together.

**Vector store structure:**

```
Vector Store (FAISS/Vertex AI Vector Search)
┌────────────────────────────────────────────────────────────┐
│ ID  │ Embedding Vector      │ Text           │ Metadata    │
├─────┼──────────────────────┼────────────────┼─────────────┤
│ 0   │ [0.23, -0.45, ...]   │ "Password..."  │ {source:... }│
│ 1   │ [0.12, 0.67, ...]    │ "Financial..." │ {source:... }│
│ 2   │ [-0.34, 0.89, ...]   │ "Remote..."    │ {source:... }│
│ ... │                      │                │             │
└─────┴──────────────────────┴────────────────┴─────────────┘
```

### Step 4: Query and Get Results with Metadata

```python
result = qa_chain.invoke({"query": "What are password requirements?"})

answer = result['result']              # Gemini's answer
source_docs = result['source_documents']  # Documents with metadata!

for doc in source_docs:
    filename = doc.metadata.get('filename')  # Original filename
    source_path = doc.metadata.get('source')  # AltaStata path
    print(f"Source: {filename}")
```

**What the query returns:**

```python
{
    'result': 'Passwords must be minimum 12 characters...',
    'source_documents': [
        Document(
            page_content="Password Requirements:\n- Minimum 12 characters...",
            metadata={
                'source': 'RAGTest/vertex_ai/security_guidelines.txt',
                'filename': 'security_guidelines.txt'
            }
        ),
        Document(
            page_content="Access Control:\n- Principle of least privilege...",
            metadata={
                'source': 'RAGTest/vertex_ai/security_guidelines.txt',
                'filename': 'security_guidelines.txt'
            }
        )
    ]
}
```

## For Insurance Companies: Why This Matters

### Scenario: Legal Document RAG

```python
# 1. Upload contracts to AltaStata
altastata.upload("contracts/2024/client-A-contract.pdf")

# 2. Load and create embeddings WITH metadata
doc = Document(
    page_content=extracted_text,
    metadata={
        "source": "contracts/2024/client-A-contract.pdf",  # AltaStata path
        "contract_id": "CNT-2024-001",
        "client": "Client A",
        "date": "2024-01-15",
        "page": 5
    }
)

# 3. When user queries: "What are Client A's payment terms?"
result = qa_chain.invoke({"query": "What are Client A's payment terms?"})

# 4. Get answer + sources
print(result['result'])
# "Payment terms are Net 30 with 2% early payment discount..."

# 5. Show exact source (with AltaStata path!)
for doc in result['source_documents']:
    print(f"Source: {doc.metadata['source']}")  # contracts/2024/client-A-contract.pdf
    print(f"Page: {doc.metadata['page']}")       # 5
    
    # 6. You can now retrieve the ORIGINAL document from AltaStata!
    original = altastata.download(doc.metadata['source'])
```

## Types of Metadata You Can Store

### Basic Metadata (Our Demo):
```python
metadata = {
    "source": "RAGTest/vertex_ai/policy.txt",
    "filename": "policy.txt"
}
```

### Production Metadata (For Insurance Companies):
```python
metadata = {
    # Document identification
    "source": "altastata://contracts/2024/client-A-contract.pdf",
    "document_id": "DOC-2024-001",
    "filename": "client-A-contract.pdf",
    
    # Content location
    "page": 5,
    "section": "Payment Terms",
    "chunk_index": 12,
    
    # Classification
    "document_type": "contract",
    "client": "Client A",
    "department": "Legal",
    "classification": "Confidential",
    
    # Timestamps
    "upload_date": "2024-10-21",
    "last_modified": "2024-10-20",
    
    # Access control
    "uploaded_by": "john.smith@company.com",
    "authorized_users": ["legal-team", "finance-team"],
    
    # Compliance
    "retention_period": "7 years",
    "gdpr_category": "contract-data"
}
```

### When Retrieving:
```python
# Get documents
relevant_docs = retriever.get_relevant_documents("payment terms")

for doc in relevant_docs:
    # Get original from AltaStata using metadata
    altastata_path = doc.metadata['source']
    original_file = altastata.download(altastata_path)
    
    # Show page reference
    page = doc.metadata.get('page')
    print(f"Found on page {page} of {doc.metadata['filename']}")
    
    # Check access permissions
    if user_email in doc.metadata['authorized_users']:
        show_content(doc.page_content)
```

## Vector Store Storage Comparison

### FAISS (In-Memory/Local):
```python
# Stores in memory or saves to disk
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save to disk (includes metadata!)
vectorstore.save_local("./faiss_index")

# Load later
vectorstore = FAISS.load_local("./faiss_index", embeddings)
```

**Storage format:**
```
faiss_index/
  ├── index.faiss      # Embedding vectors (binary)
  └── index.pkl        # Metadata + text (pickle file)
```

### Vertex AI Vector Search (Managed):
```python
# Upload datapoints with metadata
datapoints = [
    {
        "datapoint_id": "doc_1",
        "feature_vector": [0.23, -0.45, ...],  # Embedding
        "restricts": [
            {"namespace": "source", "allow_list": ["contracts/doc.pdf"]},
            {"namespace": "client", "allow_list": ["Client A"]}
        ]
    }
]

index.upsert_datapoints(datapoints)
```

**What gets stored:**
- Embedding vectors (for similarity search)
- Metadata as "restricts" (for filtering)
- Datapoint IDs (to link back to your database)

**You need to store text separately** (e.g., in Cloud Storage or database)

## Key Takeaways

1. ✅ **Embeddings + Metadata are stored together** in vector stores
2. ✅ **Metadata includes the AltaStata path** to retrieve original documents  
3. ✅ **LangChain preserves metadata** when splitting documents into chunks
4. ✅ **Query results include metadata** for citations and source tracking
5. ✅ **Custom metadata** enables filtering, access control, and compliance

## See Also

- `../CHUNKING_STRATEGIES.md` - Optimal chunk sizes for Gemini 1.5 and other LLMs
- `../PDF_PROCESSING.md` - How to extract and preserve page numbers from PDFs

## For Insurance Companies

```
User Query: "What are Client A's insurance terms?"
        ↓
Query Embedding → Vector Search
        ↓
Finds 3 relevant chunks:
  - Chunk 1 (similarity: 0.92) metadata: {source: "contracts/clientA.pdf", page: 12}
  - Chunk 2 (similarity: 0.88) metadata: {source: "contracts/clientA.pdf", page: 13}
  - Chunk 3 (similarity: 0.85) metadata: {source: "policies/insurance.pdf", page: 5}
        ↓
Send chunks + metadata to Gemini
        ↓
Gemini Answer: "Client A has comprehensive coverage with..."
        ↓
Show Sources:
  - contracts/clientA.pdf (pages 12-13)
  - policies/insurance.pdf (page 5)
        ↓
User clicks source → Download from AltaStata using metadata path
```

**This is how you maintain the link between embeddings and original encrypted documents!** 🔗

