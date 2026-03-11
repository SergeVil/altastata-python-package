# RAG Chunking Strategies Guide

## Why Chunk Size Matters

Chunk size is one of the **most critical parameters** in RAG systems. It directly affects:
- **Retrieval quality** - Can you find the right information?
- **Answer quality** - Does the LLM have enough context?
- **Cost** - More chunks = more embeddings = higher costs (but also cheaper queries!)
- **Performance** - Larger chunks = fewer searches needed

## Modern Context: Large Language Models

**Important Update (2024):** Modern LLMs have massive context windows:
- **Gemini 1.5 Pro**: 2 million tokens (~1,500 pages)
- **Gemini 1.5 Flash**: 1 million tokens (~750 pages)
- **GPT-4 Turbo**: 128K tokens (~96 pages)
- **Claude 3**: 200K tokens (~150 pages)

**Does this mean you don't need chunking?** NO! You still need it for:
1. **Cost** - Sending 3 relevant chunks is 10x cheaper than entire documents
2. **Precision** - Vector search on smaller chunks finds more relevant content
3. **Quality** - LLMs perform better with focused information (avoid "lost in the middle")

**What changed:** You can use **larger chunks** now without worrying about context limits!

## Quick Reference

| Document Type | Chunk Size (Traditional) | Chunk Size (Large Context LLMs*) | Best For |
|--------------|--------------------------|----------------------------------|----------|
| **FAQs, Q&A** | 300-500 | 1000-2000 | Short, self-contained answers |
| **General docs** | 1000-1500 | 3000-5000 | Articles, blogs, documentation |
| **Legal/Technical** | 1500-3000 | 5000-8000 | Contracts, policies, specifications |
| **Books, Research** | 2000-4000 | 8000-15000 | Long-form content, academic papers |

*Large Context LLMs = Gemini 1.5 (1-2M tokens), GPT-4 Turbo (128K), Claude 3 (200K)

## Detailed Comparison

### Small Chunks (300-500 characters)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

**Example output:**
```
Chunk 1: "Password Requirements: Minimum 12 characters, mix of uppercase, 
lowercase, numbers, and special characters. Change passwords every 90 days."

Chunk 2: "Change passwords every 90 days. No password reuse for last 10 
passwords. Enable multi-factor authentication (MFA) for all accounts."
```

**Use cases:**
- ✅ FAQ systems
- ✅ Chatbot responses (short answers)
- ✅ Highly structured documents with distinct sections

**Avoid for:**
- ❌ Legal documents (will split clauses)
- ❌ Technical specifications (needs full context)
- ❌ Complex narratives

---

### Medium Chunks (1000-1500 characters) ⭐ RECOMMENDED

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**Example output:**
```
Chunk 1: "Enterprise Security Guidelines

Version 2.0 - Updated March 2024

Password Requirements:
- Minimum 12 characters
- Mix of uppercase, lowercase, numbers, and special characters
- Change passwords every 90 days
- No password reuse for last 10 passwords
- Enable multi-factor authentication (MFA) for all accounts

Data Classification:
- Public: Information freely shareable
- Internal: For company use only
- Confidential: Restricted to specific teams"
```

**Use cases:**
- ✅ Corporate policies
- ✅ Product documentation
- ✅ Technical guides
- ✅ **Best general-purpose setting**

**Why this works:**
- Maintains paragraph-level context
- Reasonable cost/benefit tradeoff
- Works with most LLM context windows

---

### Large Chunks (2000-4000 characters)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=600,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**Use cases:**
- ✅ Legal contracts (complete clauses)
- ✅ Insurance policies
- ✅ Academic papers
- ✅ Complex technical specifications

**Why larger for legal docs:**
```
Legal Clause Example:

"Article 5: Payment Terms

5.1 The Client shall pay the Contractor within thirty (30) days of invoice 
     receipt, subject to the following conditions:
     (a) All deliverables meet the acceptance criteria defined in Appendix A;
     (b) The Contractor has provided all required documentation;
     (c) No material breaches of this Agreement exist at the time of payment.

5.2 Late payments shall incur interest at the rate of 1.5% per month or the 
     maximum rate permitted by law, whichever is lower.

5.3 The Client may withhold payment if..."
```

**If chunk_size=500:** This gets split into 6+ chunks, breaking the logical structure!  
**If chunk_size=3000:** The entire clause stays together ✅

---

## Chunk Overlap Explained

Overlap ensures context isn't lost at chunk boundaries.

### Without Overlap:
```
Chunk 1: "...employees must complete annual training."
Chunk 2: "New employees must complete training within 30 days."
```
**Problem:** "New employees" loses context about what training is required.

### With 20% Overlap (recommended):
```
Chunk 1: "...employees must complete annual security training. 
          New employees must complete..."
Chunk 2: "...employees must complete annual security training. 
          New employees must complete training within 30 days of hire."
```
**Better:** Both chunks have enough context!

### Overlap Guidelines:
- **10-15%** - Minimal, for well-structured docs
- **20%** (recommended) - Good balance
- **30-40%** - Maximum, for complex legal/technical docs

---

## For Insurance Companies: Recommended Settings

### With Gemini 1.5 (Recommended for Vertex AI)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=6000,          # Larger chunks for Gemini's big context
    chunk_overlap=1000,       # Larger overlap for better continuity
    separators=[
        "\n\n",               # Paragraph breaks
        "\n",                 # Line breaks
        ". ",                 # Sentences
        " ",
        ""
    ],
    length_function=len,
)
```

**Why 6000 chars with Gemini?**
- Gemini 1.5 has 1M+ token context window
- Can handle entire contract sections (multiple clauses together)
- Better context for complex legal relationships
- Still small enough for precise retrieval
- **Cost optimization**: Fewer, larger chunks = fewer API calls

### Traditional LLMs (GPT-3.5, older models)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,          # Smaller for limited context
    chunk_overlap=400,
)
```

**Use this if:**
- Using older LLMs with smaller context windows
- Need maximum precision in retrieval
- Working with highly structured documents

### Short-form Documents (Emails, Notes)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)
```

### Very Long Documents (Annual Reports, Manuals)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3500,
    chunk_overlap=700,
)
```

---

## Advanced: Semantic Chunking

Instead of splitting by character count, split by **meaning**.

### Method 1: Split by Headers/Sections

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)
```

**Result:** Each section becomes a chunk, regardless of length.

### Method 2: Semantic Similarity Splitting

```python
from langchain_experimental.text_splitter import SemanticChunker

semantic_chunker = SemanticChunker(
    embeddings=VertexAIEmbeddings(),
    breakpoint_threshold_type="percentile"  # Split when semantic similarity drops
)
```

**Result:** Splits when topic changes, not at arbitrary character counts.

---

## Cost Impact

### Example: 10,000 legal documents × 5 pages each

**Small chunks (500 chars):**
- ~500,000 chunks
- Embedding cost (Vertex AI): ~$125
- Storage: High

**Medium chunks (1500 chars):**
- ~167,000 chunks
- Embedding cost: ~$42
- Storage: Medium
- **BEST ROI** ✅

**Large chunks (3000 chars):**
- ~83,000 chunks
- Embedding cost: ~$21
- Storage: Low
- May miss precise matches

---

## Testing Your Chunk Size

### Step 1: Start with Recommended Size
```python
chunk_size = 1500  # Good default
```

### Step 2: Test with Sample Queries

```python
# Test query
query = "What are the password requirements?"

# Retrieve chunks
docs = retriever.get_relevant_documents(query)

# Check results
for doc in docs:
    print(f"Chunk length: {len(doc.page_content)}")
    print(f"Content: {doc.page_content[:200]}...")
    print()
```

### Step 3: Evaluate

**Too small if:**
- Chunks don't make sense on their own
- Need to retrieve 5+ chunks to answer simple questions
- Answers lack context

**Too large if:**
- Chunks contain multiple unrelated topics
- Hard to pinpoint exact information
- Retrieval is imprecise

---

## Production Best Practices

### 1. Different Sizes for Different Document Types

```python
def get_splitter_for_doc_type(doc_type: str):
    if doc_type == "contract":
        return RecursiveCharacterTextSplitter(
            chunk_size=2500, 
            chunk_overlap=500
        )
    elif doc_type == "email":
        return RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=150
        )
    elif doc_type == "policy":
        return RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=300
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=300
        )
```

### 2. Store Chunk Strategy in Metadata

```python
doc = Document(
    page_content=chunk_text,
    metadata={
        "source": "contract.pdf",
        "chunk_size": 2500,
        "chunk_overlap": 500,
        "splitter_version": "v2.0"
    }
)
```

**Why:** If you need to re-chunk documents later, you know what settings were used.

### 3. A/B Test Different Strategies

```python
# Version A: Medium chunks
chunker_a = RecursiveCharacterTextSplitter(chunk_size=1500)

# Version B: Large chunks
chunker_b = RecursiveCharacterTextSplitter(chunk_size=3000)

# Test with real queries and compare quality
```

---

## Summary: What Should Insurance Companies Use?

### For Legal Document Summarization (Squeezy):
```python
chunk_size = 2000       # Full legal clauses
chunk_overlap = 400     # 20% overlap
```

### For RAG System (Query Historical Documents):
```python
chunk_size = 1500       # Balance precision and context
chunk_overlap = 300     # Standard overlap
```

### For Mixed Document Types:
```python
# Use adaptive chunking based on document type
def chunk_document(doc, doc_type):
    if doc_type in ["contract", "policy", "legal"]:
        chunk_size = 2000
    elif doc_type in ["email", "note", "memo"]:
        chunk_size = 800
    else:
        chunk_size = 1500
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.2)  # 20% overlap
    )
    return splitter.split_documents([doc])
```

---

## Updated Code Examples

All our test files now use **Gemini 1.5 optimized chunk sizes**:

- `test_rag.py` - 1500 chars (HuggingFace embeddings, traditional approach)
- `test_rag_vertex.py` - 4000 chars (Vertex AI with Gemini 1.5)

**For production with Gemini 1.5:**
- General documents: 4000-5000 chars
- Legal documents: 6000-8000 chars

---

**Key Takeaway:** With modern large-context LLMs like Gemini 1.5, use larger chunks (4000-8000 chars) to preserve context while still enabling precise retrieval. For traditional LLMs, stick with 1500-2500 chars. 📏

