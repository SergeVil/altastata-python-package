#!/usr/bin/env python3
"""
Basic RAG Pipeline with AltaStata + HuggingFace

Demonstrates:
- Encrypted document storage with AltaStata
- Local embeddings with sentence-transformers
- FAISS vector store for semantic search

Requirements:
    pip install altastata fsspec langchain langchain-community sentence-transformers faiss-cpu
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from altastata.altastata_functions import AltaStataFunctions
from altastata.fsspec import create_filesystem
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_sample_documents():
    """Load sample policy documents from disk"""
    sample_docs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_documents")
    documents = {}
    
    # List of sample files to load
    sample_files = [
        "company_policy.txt",
        "security_guidelines.txt",
        "remote_work_policy.txt",
        "ai_usage_policy.txt"
    ]
    
    for filename in sample_files:
        file_path = os.path.join(sample_docs_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents[filename] = f.read()
        except FileNotFoundError:
            print(f"⚠️  Warning: {filename} not found at {file_path}")
        except Exception as e:
            print(f"⚠️  Error loading {filename}: {e}")
    
    return documents


def test_rag_pipeline():
    """Test complete RAG pipeline with AltaStata"""
    print("🚀 Testing RAG Pipeline with AltaStata fsspec")
    print("=" * 80)
    
    # Initialize AltaStata
    print("\n1️⃣  Initializing AltaStata connection...")
    altastata_functions = AltaStataFunctions.from_account_dir(
        '/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123'
    )
    altastata_functions.set_password("123")
    
    # Create filesystem instance
    fs = create_filesystem(altastata_functions, "bob123")
    
    print("✅ AltaStata initialized")
    
    # Upload sample documents
    print("\n2️⃣  Loading and uploading sample documents to encrypted storage...")
    sample_docs = load_sample_documents()
    test_dir = "RAGTest/documents"
    
    if not sample_docs:
        print("❌ No sample documents found!")
        print(f"   Expected location: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_documents')}")
        return
    
    for filename, content in sample_docs.items():
        file_path = f"{test_dir}/{filename}"
        result = altastata_functions.create_file(file_path, content.encode('utf-8'))
        print(f"   ✅ Uploaded: {filename} - {result.getOperationStateValue()}")
    
    # Load documents using fsspec directly
    print("\n3️⃣  Loading documents via fsspec...")
    try:
        # Load each file individually using fsspec
        from langchain_core.documents import Document
        documents = []
        for filename in sample_docs.keys():
            file_path = f"{test_dir}/{filename}"
            try:
                # Read file content using fsspec
                with fs.open(file_path, "r") as f:
                    content = f.read()
                    # Create LangChain document
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path}
                    )
                    documents.append(doc)
                    print(f"   ✅ Loaded: {filename} ({len(content)} chars)")
            except Exception as e:
                print(f"   ❌ Failed to load {filename}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"✅ Total documents loaded: {len(documents)}")
    
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        raise
    
    # Split documents into chunks
    print("\n4️⃣  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,      # ~2-3 paragraphs (better for legal/policy docs)
        chunk_overlap=300,    # 20% overlap to preserve context across chunks
        separators=["\n\n", "\n", ". ", " ", ""],  # Split on paragraphs first
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks")
    
    # Create embeddings and vector store
    print("\n5️⃣  Creating embeddings and vector store...")
    print("   (This may take a minute on first run - downloading model)")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("✅ Vector store created successfully")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        raise
    
    # Test queries
    print("\n6️⃣  Testing RAG queries...")
    print("=" * 80)
    
    test_queries = [
        "What are the password requirements?",
        "How long do we keep financial records?",
        "What is the remote work policy for equipment?",
        "Which AI tools are approved for use?",
        "What encryption standards are required?"
    ]
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📊 Query {i}: {query}")
        print("-" * 80)
        
        try:
            relevant_docs = retriever.get_relevant_documents(query)
            print(f"Found {len(relevant_docs)} relevant chunks:\n")
            
            for j, doc in enumerate(relevant_docs, 1):
                print(f"[Chunk {j}] (from: {os.path.basename(doc.metadata.get('source', 'Unknown'))})")
                print(f"{doc.page_content[:200]}...")
                print()
        
        except Exception as e:
            print(f"❌ Error querying: {e}")
    
    # Test similarity search with scores
    print("\n7️⃣  Testing similarity search with scores...")
    print("=" * 80)
    query = "What are the data retention policies?"
    print(f"\nQuery: {query}\n")
    
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            print(f"[Result {i}] Similarity: {score:.4f}")
            print(f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
            print(f"Content: {doc.page_content[:150]}...")
            print()
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")
    
    # Cleanup
    print("\n8️⃣  Cleaning up test data...")
    try:
        # Delete all test documents
        for filename in sample_docs.keys():
            file_path = f"{test_dir}/{filename}"
            result = altastata_functions.delete_files(file_path, False, None, None)
            print(f"   ✅ Deleted: {filename}")
        
        # Try to delete the directory (it's okay if this fails)
        try:
            altastata_functions.delete_files(test_dir, True, None, None)
            print(f"   ✅ Deleted directory: {test_dir}")
        except:
            pass  # Directory might not be empty or might not support deletion
            
    except Exception as e:
        print(f"⚠️  Warning during cleanup: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Basic RAG Pipeline Test Completed!")
    print("=" * 80)
    print("\n🎯 Features:")
    print("  ✅ AltaStata encrypted storage")
    print("  ✅ Local embeddings (HuggingFace)")
    print("  ✅ Semantic search with FAISS")
    print("\n💡 For Production:")
    print("  • Use test_rag_vertex.py for Vertex AI integration")
    print("  • See VERTEX_AI_SETUP.md for deployment guide")


if __name__ == "__main__":
    try:
        test_rag_pipeline()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

