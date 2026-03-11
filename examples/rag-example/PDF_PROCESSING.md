# PDF Processing for RAG Systems

## The Challenge

PDFs have structure that plain text doesn't:
- Multiple pages
- Formatting (bold, italics, tables)
- Images and diagrams
- Headers and footers
- Page numbers

**Goal:** Extract text while preserving important metadata (especially page numbers) for citations.

## Complete Workflow

### Step 1: Extract Text from PDF

There are several libraries for PDF text extraction:

#### Option A: PyPDF2 (Simple, Fast)

```python
from PyPDF2 import PdfReader
from langchain_core.documents import Document

def load_pdf_simple(pdf_path):
    """Extract text from PDF, one document per page"""
    reader = PdfReader(pdf_path)
    documents = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        
        # Create LangChain document with page metadata
        doc = Document(
            page_content=text,
            metadata={
                "source": pdf_path,
                "page": page_num,
                "total_pages": len(reader.pages)
            }
        )
        documents.append(doc)
    
    return documents
```

**Pros:** Fast, simple, pure Python  
**Cons:** Struggles with complex layouts, tables, scanned PDFs

---

#### Option B: pdfplumber (Better for Tables)

```python
import pdfplumber
from langchain_core.documents import Document

def load_pdf_with_tables(pdf_path):
    """Extract text and tables from PDF"""
    documents = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract text
            text = page.extract_text()
            
            # Extract tables (if any)
            tables = page.extract_tables()
            if tables:
                # Convert tables to markdown format
                for table in tables:
                    table_md = "\n".join([" | ".join(row) for row in table])
                    text += f"\n\n{table_md}\n"
            
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_num,
                    "has_tables": len(tables) > 0
                }
            )
            documents.append(doc)
    
    return documents
```

**Pros:** Better table extraction, more accurate  
**Cons:** Slower than PyPDF2

---

#### Option C: LangChain PyPDFLoader (Easiest)

```python
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_langchain(pdf_path):
    """Use LangChain's built-in PDF loader"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # Automatically creates one doc per page
    
    # Documents already have page metadata!
    # metadata = {"source": "file.pdf", "page": 0, ...}
    
    return documents
```

**Pros:** Dead simple, automatic page metadata  
**Cons:** Uses PyPDF2 under the hood (same limitations)

---

#### Option D: Unstructured (Most Powerful)

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

def load_pdf_unstructured(pdf_path):
    """Advanced PDF processing with layout detection"""
    loader = UnstructuredPDFLoader(
        pdf_path,
        mode="elements",  # Preserves document structure
        strategy="hi_res"  # High-resolution processing
    )
    documents = loader.load()
    
    return documents
```

**Pros:** Handles complex layouts, OCR for scanned PDFs, detects headers/footers  
**Cons:** Requires additional dependencies, slower

---

### Step 2: Apply Chunking Strategy

Once you have extracted text with page metadata, apply chunking:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
documents = load_pdf_langchain("contract.pdf")

# Apply chunking (see CHUNKING_STRATEGIES.md for optimal sizes)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=6000,      # Recommended for Gemini 1.5
    chunk_overlap=1000,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)
# Metadata is automatically preserved in each chunk!
```

**Important:** 
- LangChain's `split_documents()` preserves metadata to each chunk
- See `CHUNKING_STRATEGIES.md` for detailed chunk size recommendations

---

### Step 3: Result - Chunks with Page Numbers

```python
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i}:")
    print(f"  Page: {chunk.metadata['page']}")
    print(f"  Text: {chunk.page_content[:100]}...")
    print()
```

**Output:**
```
Chunk 0:
  Page: 1
  Text: Article 1: Introduction
This agreement is entered into between...

Chunk 1:
  Page: 1
  Text: ...party agrees to the terms outlined herein.

Article 2: Payment Terms...

Chunk 2:
  Page: 2
  Text: 2.1 The client shall pay within thirty (30) days...
```

---

## Complete Example: PDF → RAG Pipeline

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. Load PDF from AltaStata
altastata_path = "contracts/client-A-2024.pdf"
pdf_bytes = altastata_functions.download(altastata_path)

# Save temporarily (or use BytesIO)
with open("/tmp/temp.pdf", "wb") as f:
    f.write(pdf_bytes)

# 2. Extract text with page metadata
loader = PyPDFLoader("/tmp/temp.pdf")
documents = loader.load()

# Update metadata with AltaStata path
for doc in documents:
    doc.metadata["altastata_source"] = altastata_path

# 3. Chunk the documents (see CHUNKING_STRATEGIES.md for optimal sizes)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=6000,      # Larger chunks for Gemini 1.5's large context
    chunk_overlap=1000
)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks from {len(documents)} pages")

# 4. Create embeddings and vector store
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Create RAG chain
llm = VertexAI(model_name="gemini-1.5-flash")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 6. Query with page citations!
result = qa_chain.invoke({"query": "What are the payment terms?"})

print(f"Answer: {result['result']}\n")
print("Sources:")
for doc in result['source_documents']:
    print(f"  - Page {doc.metadata['page']} of {doc.metadata['altastata_source']}")
```

**Output:**
```
Answer: Payment terms are Net 30 days from invoice date, with a 2% early payment 
discount if paid within 10 days.

Sources:
  - Page 5 of contracts/client-A-2024.pdf
  - Page 5 of contracts/client-A-2024.pdf
  - Page 6 of contracts/client-A-2024.pdf
```

---

## Special Cases

### Multi-Column PDFs

Some PDFs have multiple columns. Use `pdfplumber` with column detection:

```python
import pdfplumber

def load_multicolumn_pdf(pdf_path):
    documents = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Detect columns
            text = page.extract_text(layout=True)  # Preserves layout
            
            doc = Document(
                page_content=text,
                metadata={"source": pdf_path, "page": page_num}
            )
            documents.append(doc)
    
    return documents
```

---

### Scanned PDFs (Images, Not Text)

For scanned PDFs, you need OCR (Optical Character Recognition):

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

def load_scanned_pdf(pdf_path):
    """Use OCR for scanned PDFs"""
    loader = UnstructuredPDFLoader(
        pdf_path,
        mode="elements",
        strategy="ocr_only"  # Forces OCR
    )
    documents = loader.load()
    
    return documents
```

**Note:** Requires `tesseract` to be installed:
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr

# Then install Python package
pip install unstructured[local-inference]
```

---

### PDFs with Images/Diagrams

If you need to process images within PDFs:

```python
from langchain_community.document_loaders import PDFMinerLoader

def load_pdf_with_images(pdf_path):
    """Extract text and note image locations"""
    loader = PDFMinerLoader(pdf_path)
    documents = loader.load()
    
    # Images typically appear as [IMAGE] markers
    # You may need additional processing to extract actual images
    
    return documents
```

For full image extraction:
```python
import fitz  # PyMuPDF

def extract_images_from_pdf(pdf_path, output_dir):
    """Extract images separately"""
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc, start=1):
        images = page.get_images()
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Save image
            image_path = f"{output_dir}/page{page_num}_img{img_index}.png"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            print(f"Extracted image from page {page_num}")
```

---

## Best Practices for PDFs

### 1. Preserve Page Numbers in Metadata

```python
# Always include page numbers for citations
doc = Document(
    page_content=text,
    metadata={
        "source": "contract.pdf",
        "page": page_num,
        "total_pages": total_pages,
        "document_type": "contract"
    }
)
```

### 2. Handle Headers/Footers

Remove repeated headers/footers that add noise:

```python
def remove_headers_footers(text, header_pattern=None, footer_pattern=None):
    """Remove repeated headers/footers"""
    lines = text.split('\n')
    
    # Remove first line if it matches header pattern
    if header_pattern and lines and header_pattern in lines[0]:
        lines = lines[1:]
    
    # Remove last line if it matches footer pattern
    if footer_pattern and lines and footer_pattern in lines[-1]:
        lines = lines[:-1]
    
    return '\n'.join(lines)
```

### 3. Adjust Chunk Size for PDF Structure

PDFs often have distinct sections. See `CHUNKING_STRATEGIES.md` for recommended chunk sizes:
- **Gemini 1.5**: 6000-8000 chars
- **Traditional LLMs**: 2000-2500 chars
- **Legal documents**: Use larger chunks to preserve clause structure

### 4. Handle Page Breaks Intelligently

Add page break character (`\f`) as a high-priority separator:

```python
separators=["\f", "\n\n", "\n", ". ", " ", ""]  # Page break first
```

See `CHUNKING_STRATEGIES.md` for complete chunking strategies.

---

## Troubleshooting

### Problem: "Text extraction is garbled"

**Solution:** PDF might use unusual encoding or fonts
```python
# Try pdfplumber instead of PyPDF2
import pdfplumber
with pdfplumber.open(pdf_path) as pdf:
    text = pdf.pages[0].extract_text()
```

### Problem: "Tables are not extracted correctly"

**Solution:** Use `pdfplumber` with table extraction
```python
tables = page.extract_tables()
```

### Problem: "Scanned PDF returns empty text"

**Solution:** Use OCR
```python
loader = UnstructuredPDFLoader(pdf_path, strategy="ocr_only")
```

### Problem: "Chunks lose page context"

**Solution:** Ensure you're using `split_documents()` not `split_text()`
```python
# ✅ Correct - preserves metadata
chunks = text_splitter.split_documents(documents)

# ❌ Wrong - loses metadata
# chunks = text_splitter.split_text(plain_text)
```

---

## Summary

**The Complete Pipeline:**

1. **Extract PDF** → Use PyPDFLoader, pdfplumber, or Unstructured
2. **Preserve Metadata** → Include page numbers, source, document type
3. **Apply Chunking** → Use `split_documents()` to preserve metadata
4. **Store in Vector DB** → Metadata automatically included
5. **Query & Cite** → Results include page numbers for citations

**Key Point:** Page numbers are preserved through metadata at every step, enabling proper citations!

**See Also:**
- `CHUNKING_STRATEGIES.md` - Optimal chunk sizes for different document types and LLMs
- `google_vertexai/EMBEDDINGS_AND_METADATA.md` - How metadata flows through the RAG pipeline

