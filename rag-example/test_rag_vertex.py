#!/usr/bin/env python3
"""
RAG Pipeline with Vertex AI + AltaStata

Demonstrates:
- Encrypted document storage with AltaStata
- Vertex AI Embeddings and Gemini
- LangChain RAG pipeline with citations

Requirements:
    pip install altastata fsspec langchain langchain-google-vertexai faiss-cpu

Setup:
    export GOOGLE_CLOUD_PROJECT="your-project-id"
    gcloud auth application-default login
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from altastata.altastata_functions import AltaStataFunctions
from altastata.fsspec import create_filesystem
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import vertexai
from vertexai.preview.generative_models import GenerativeModel


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


def test_rag_vertex_ai():
    """Test complete RAG pipeline with AltaStata + Vertex AI"""
    print("🚀 Testing RAG Pipeline with AltaStata + Vertex AI")
    print("=" * 80)
    
    # Check environment
    print("\n0️⃣  Checking Google Cloud configuration...")
    # Use altastata-coco which has Gemini 2.5 Publisher Model access
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'altastata-coco')
    print(f"✅ Using GCP Project: {project_id}")
    
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    print(f"✅ Using GCP Location: {location}")
    
    # Initialize AltaStata
    print("\n1️⃣  Initializing AltaStata connection...")
    altastata_functions = AltaStataFunctions.from_account_dir(
        '/Users/sergevilvovsky/.altastata/accounts/azure.rsa.bob123'
    )
    altastata_functions.set_password("123")
    
    # Create filesystem instance
    fs = create_filesystem(altastata_functions, "bob123")
    
    print("✅ AltaStata initialized (encrypted storage ready)")
    
    # Upload sample documents
    print("\n2️⃣  Loading and uploading sample documents to encrypted storage...")
    sample_docs = load_sample_documents()
    test_dir = "RAGTest/vertex_ai"
    
    if not sample_docs:
        print("❌ No sample documents found!")
        print(f"   Expected location: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_documents')}")
        return
    
    for filename, content in sample_docs.items():
        file_path = f"{test_dir}/{filename}"
        result = altastata_functions.create_file(file_path, content.encode('utf-8'))
        print(f"   ✅ Uploaded: {filename} - {result.getOperationStateValue()}")
    
    # Load documents using fsspec
    print("\n3️⃣  Loading documents via fsspec...")
    try:
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
                        metadata={"source": file_path, "filename": filename}
                    )
                    documents.append(doc)
                    print(f"   ✅ Loaded: {filename} ({len(content)} chars)")
            except Exception as e:
                print(f"   ❌ Failed to load {filename}: {e}")
                raise
        
        print(f"✅ Total documents loaded: {len(documents)}")
    
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        raise
    
    # Split documents into chunks
    print("\n4️⃣  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,      # Larger chunks for Gemini's 1M+ token context window
        chunk_overlap=800,    # 20% overlap to preserve context across chunks
        separators=["\n\n", "\n", ". ", " ", ""],  # Split on paragraphs first
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks")
    print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    print(f"   💡 Using larger chunks optimized for Gemini 1.5's large context window")
    
    # Create embeddings with Vertex AI
    print("\n5️⃣  Creating embeddings with Vertex AI...")
    print("   (Using text-embedding-004 model)")
    try:
        embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=project_id,
            location=location
        )
        
        # Test embedding generation
        print("   Testing embedding generation...")
        test_embedding = embeddings.embed_query("test query")
        print(f"   ✅ Embedding dimension: {len(test_embedding)}")
        
        # Create vector store
        print("   Creating FAISS vector store...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("✅ Vector store created successfully with Vertex AI embeddings")
        print("   💡 100% Vertex AI: Embeddings + Gemini 2.5 Flash")
        
    except Exception as e:
        print(f"❌ Error creating embeddings/vector store: {e}")
        raise
    
    # Initialize Vertex AI (native SDK)
    print("\n6️⃣  Initializing Vertex AI with native SDK...")
    try:
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel("gemini-2.5-flash")
        print("✅ Gemini 2.5 Flash initialized (native Vertex AI SDK)")
    except Exception as e:
        print(f"❌ Error initializing Gemini: {e}")
        raise
    
    # Create retriever
    print("\n7️⃣  Setting up vector retriever...")
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ Retriever ready (will retrieve top 3 relevant chunks)")
    except Exception as e:
        print(f"❌ Error creating retriever: {e}")
        raise
    
    # Test RAG queries with Gemini
    print("\n8️⃣  Testing RAG queries with Gemini...")
    print("=" * 80)
    
    test_queries = [
        "What are the password requirements?",
        "How long do we keep financial records?",
        "What is the remote work policy for equipment?",
        "Which AI tools are approved for use?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 80}")
        print(f"📊 QUERY {i}/4")
        print(f"{'─' * 80}")
        print(f"❓ {query}")
        print()
        
        try:
            # Retrieve relevant documents
            relevant_docs = retriever.get_relevant_documents(query)
            
            # Build context from retrieved documents
            context = "\n\n".join([
                f"Document {j+1} ({doc.metadata.get('filename', 'Unknown')}):\n{doc.page_content}"
                for j, doc in enumerate(relevant_docs)
            ])
            
            # Create prompt for Gemini
            prompt = f"""Based on the following documents, please answer the question concisely and clearly.

Context:
{context}

Question: {query}

Answer:"""
            
            # Call Gemini directly
            response = model.generate_content(prompt)
            answer = response.text.strip()
            
            # Format the answer nicely
            print(f"🤖 ANSWER:")
            print()
            for line in answer.split('\n'):
                print(f"   {line}")
            print()
            
            print(f"📚 SOURCES ({len(relevant_docs)} documents retrieved):")
            for j, doc in enumerate(relevant_docs, 1):
                filename = doc.metadata.get('filename', 'Unknown')
                preview = doc.page_content[:80].replace('\n', ' ')
                print(f"   {j}. 📄 {filename}")
                print(f"      └─ {preview}...")
            
        except Exception as e:
            print(f"❌ Error querying: {e}")
            import traceback
            traceback.print_exc()
    
    # Test direct similarity search
    print(f"\n{'═' * 80}")
    print("9️⃣  DIRECT SIMILARITY SEARCH TEST")
    print(f"{'═' * 80}")
    query = "What are the data retention policies?"
    print(f"❓ {query}\n")
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(query)
        print(f"✅ Found {len(relevant_docs)} most relevant chunks:\n")
        
        for i, doc in enumerate(relevant_docs, 1):
            filename = doc.metadata.get('filename', 'Unknown')
            # Show relevant context (first 300 chars, preserving readability)
            content = doc.page_content[:300]
            # Add ellipsis if truncated
            if len(doc.page_content) > 300:
                content += "..."
            
            print(f"   {i}. 📄 {filename}")
            print(f"      {content}")
            print()
    
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")
    
    # Cleanup
    print("\n🧹 Cleaning up test data...")
    try:
        # Delete all test documents
        for filename in sample_docs.keys():
            file_path = f"{test_dir}/{filename}"
            result = altastata_functions.delete_files(file_path, False, None, None)
            print(f"   ✅ Deleted: {filename}")
        
        # Try to delete the directory
        try:
            altastata_functions.delete_files(test_dir, True, None, None)
            print(f"   ✅ Deleted directory: {test_dir}")
        except:
            pass  # Directory might not be empty
            
    except Exception as e:
        print(f"⚠️  Warning during cleanup: {e}")
    
    print(f"\n{'═' * 80}")
    print("✅ RAG PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print(f"{'═' * 80}")
    
    print("\n🎯 ARCHITECTURE STACK:")
    print()
    print("   🔐 AltaStata")
    print("      └─ End-to-end encrypted storage")
    print("      └─ Zero-trust security model")
    print()
    print("   🤖 Vertex AI Stack (100%)")
    print("      ├─ Text Embeddings (768-dim)")
    print("      └─ Gemini 2.5 Flash (LLM)")
    print()
    print("   🔗 Integration")
    print("      ├─ LangChain RAG pipeline")
    print("      ├─ FAISS vector store")
    print("      └─ Full source citations")
    
    print("\n💡 NEXT STEPS FOR PRODUCTION:")
    print("   ⚡ Scale to Vertex AI Matching Engine (>100K documents)")
    print("   🔐 Add user authentication and audit logging")
    print("   ☁️  Deploy to GKE with Confidential Computing")
    print("   📊 Implement monitoring and observability")
    print(f"\n{'═' * 80}")


if __name__ == "__main__":
    try:
        test_rag_vertex_ai()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

