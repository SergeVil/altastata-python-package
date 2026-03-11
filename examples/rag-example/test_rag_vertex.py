#!/usr/bin/env python3
"""
RAG Pipeline with Vertex AI Vector Search + AltaStata

Demonstrates:
- Encrypted document storage with AltaStata
- Vertex AI Embeddings and Vector Search
- Gemini for generation
- Full production-ready RAG pipeline

Requirements:
    pip install altastata fsspec langchain langchain-google-vertexai google-cloud-aiplatform

Setup:
    1. Run: python setup_vertex_search.py (one-time, 20-40 min)
    2. export GOOGLE_CLOUD_PROJECT="altastata-coco"
    3. gcloud auth application-default login
    4. python test_rag_vertex.py
"""

import sys
import os
import atexit
import signal

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from altastata.altastata_functions import AltaStataFunctions
from altastata.fsspec import create_filesystem
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# Global variable to track AltaStata functions for cleanup
altastata_functions = None

def cleanup_altastata():
    """Clean up AltaStata Java process"""
    global altastata_functions
    if altastata_functions:
        try:
            print("\n🧹 Cleaning up AltaStata Java process...")
            # AltaStataFunctions doesn't have a close() method, but we can set it to None
            # The Java process will be cleaned up when the Python process exits
            altastata_functions = None
            print("✅ AltaStata Java process will be cleaned up on exit")
        except Exception as e:
            print(f"⚠️  Warning during AltaStata cleanup: {e}")

def signal_handler(signum, _frame):
    """Handle interrupt signals"""
    print(f"\n🛑 Received signal {signum}, cleaning up...")
    cleanup_altastata()
    sys.exit(0)

# Register cleanup functions
atexit.register(cleanup_altastata)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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
    """Test complete RAG pipeline with AltaStata + Vertex AI Vector Search"""
    print("🚀 Testing RAG Pipeline with AltaStata + Vertex AI Vector Search")
    print("=" * 80)
    
    # Check environment
    print("\n0️⃣  Checking Google Cloud configuration...")
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'altastata-coco')
    print(f"✅ Using GCP Project: {project_id}")
    
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    print(f"✅ Using GCP Location: {location}")
    
    # Load Vertex AI Vector Search config
    print("\n📋 Loading Vertex AI Vector Search configuration...")
    config_path = os.path.join(os.path.dirname(__file__), ".vertex_config")
    
    if not os.path.exists(config_path):
        print(f"❌ Vertex AI config not found: {config_path}")
        print("\n📝 Run this first:")
        print("   python setup_vertex_search.py")
        print("\n   (This takes 20-40 minutes but only needs to be done once)")
        return
    
    vertex_config = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                vertex_config[key] = value
    
    print("✅ Config loaded")
    print(f"   Index: {vertex_config.get('INDEX_ID', 'N/A')}")
    
    # Initialize AltaStata
    print("\n1️⃣  Initializing AltaStata connection...")
    global altastata_functions
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
    print("   💡 Using larger chunks optimized for Gemini 1.5's large context window")
    
    # Create embeddings and index into Vertex AI Vector Search
    print("\n5️⃣  Creating embeddings and indexing...")
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
        
        # Connect to Vertex AI Vector Search index
        print("   Connecting to Vertex AI Vector Search...")
        from google.cloud import aiplatform
        aiplatform.init(project=project_id, location=location)
        
        index = aiplatform.MatchingEngineIndex(index_name=vertex_config['INDEX_ID'])
        endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=vertex_config['ENDPOINT_ID']
        )
        deployed_index_id = vertex_config['DEPLOYED_INDEX_ID']
        
        print("✅ Connected to Vertex AI Vector Search")
        
        # Generate embeddings for all chunks
        print("   Generating embeddings for all chunks...")
        texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = embeddings.embed_documents(texts)
        
        # Prepare datapoints with metadata encoded in datapoint_id
        print("   Preparing datapoints...")
        datapoints = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            # Encode metadata in datapoint_id (format: source_file_chunk_index)
            source_file = chunk.metadata.get("source", "").replace("/", "_")
            datapoint_id = f"{source_file}_{i}"
            
            datapoints.append({
                "datapoint_id": datapoint_id,
                "feature_vector": embedding
            })
        
        # Upsert to Vertex AI Vector Search
        print(f"   Upserting {len(datapoints)} vectors to Vertex AI...")
        index.upsert_datapoints(datapoints=datapoints)
        print("✅ Vectors indexed in Vertex AI Vector Search")
        print("   💡 Metadata encoded in datapoint IDs - no local JSON needed!")
        print("   💡 100% Vertex AI: Embeddings + Vector Search + Gemini 2.5 Flash")
        
    except Exception as e:
        print(f"❌ Error creating embeddings/indexing: {e}")
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
    
    # Retriever setup (using Vertex AI endpoint)
    print("\n7️⃣  Setting up vector search retriever...")
    print("✅ Retriever ready (will query Vertex AI Vector Search for top 3 chunks)")
    
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
            # Generate query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Query Vertex AI Vector Search
            response = endpoint.find_neighbors(
                deployed_index_id=deployed_index_id,
                queries=[query_embedding],
                num_neighbors=3
            )
            
            # Get results
            neighbors = response[0] if response else []
            relevant_docs = []
            
            # Apply strict similarity threshold (same as bob_query.py)
            SIMILARITY_THRESHOLD = 0.5
            relevant_neighbors = [n for n in neighbors if getattr(n, 'distance', 1.0) < SIMILARITY_THRESHOLD]
            
            # If we don't have enough with the strict threshold, take top 2 most similar
            if len(relevant_neighbors) < 2:
                print(f"   ⚠️  Only {len(relevant_neighbors)} documents passed strict threshold, taking top 2 most similar")
                relevant_neighbors = neighbors.copy()
                relevant_neighbors.sort(key=lambda n: getattr(n, 'distance', 1.0))
                relevant_neighbors = relevant_neighbors[:2]
            
            print(f"   🎯 Similarity threshold: {SIMILARITY_THRESHOLD}")
            print(f"   📊 Relevant documents: {len(relevant_neighbors)}/{len(neighbors)}")
            
            for neighbor in relevant_neighbors:
                datapoint_id = getattr(neighbor, 'id', '') or getattr(neighbor, 'datapoint_id', '')
                
                # Parse metadata from datapoint_id (format: source_file_chunk_index)
                if '_' in datapoint_id:
                    try:
                        parts = datapoint_id.rsplit('_', 1)
                        if len(parts) == 2:
                            # Convert back to path - handle both old and new directory structures
                            path_parts = parts[0].split('_')
                            
                            if path_parts[0] == 'RAGTest' and len(path_parts) >= 3:
                                # RAGTest_vertex_ai_filename -> RAGTest/vertex_ai/filename
                                source_file = f"{path_parts[0]}/{path_parts[1]}_{path_parts[2]}/" + "_".join(path_parts[3:])
                            elif path_parts[0] == 'RAGDocs' and len(path_parts) >= 2:
                                # RAGDocs_policies_filename -> RAGDocs/policies/filename
                                source_file = f"{path_parts[0]}/{path_parts[1]}/" + "_".join(path_parts[2:])
                            else:
                                # Fallback: replace all underscores with slashes
                                source_file = parts[0].replace('_', '/')
                            chunk_index = int(parts[1])
                            
                            # Read full document from AltaStata and re-chunk
                            try:
                                with fs.open(source_file, "r") as f:
                                    full_content = f.read()
                                
                                # Re-chunk the document
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=4000,
                                    chunk_overlap=800,
                                    separators=["\n\n", "\n", ". ", " ", ""]
                                )
                                chunks = text_splitter.split_text(full_content)
                                
                                # Extract specific chunk
                                if chunk_index < len(chunks):
                                    relevant_docs.append({
                                        "text": chunks[chunk_index],
                                        "filename": os.path.basename(source_file),
                                        "source": source_file
                                    })
                                    print(f"   ✅ Found valid document: {source_file} (chunk {chunk_index})")
                            except Exception as e:
                                print(f"   ⚠️  Could not read {source_file}: {e}")
                                continue
                    except (ValueError, IndexError):
                        print(f"   ⚠️  Skipping document with invalid datapoint ID: {datapoint_id}")
                        continue
            
            # Build context from retrieved documents
            context = "\n\n".join([
                f"Document {j+1} ({doc['filename']}):\n{doc['text']}"
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
            print("🤖 ANSWER:")
            print()
            for line in answer.split('\n'):
                print(f"   {line}")
            print()
            
            print(f"📚 SOURCES ({len(relevant_docs)} documents retrieved):")
            for j, doc in enumerate(relevant_docs, 1):
                filename = doc.get('filename', 'Unknown')
                preview = doc.get('text', '')[:80].replace('\n', ' ')
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
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Query Vertex AI Vector Search
        response = endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query_embedding],
            num_neighbors=3
        )
        
        neighbors = response[0] if response else []
        print(f"✅ Found {len(neighbors)} most relevant chunks:\n")
        
        for i, neighbor in enumerate(neighbors, 1):
            datapoint_id = getattr(neighbor, 'id', '') or getattr(neighbor, 'datapoint_id', '')
            distance = getattr(neighbor, 'distance', 1.0)
            
            # Parse metadata from datapoint_id (format: source_file_chunk_index)
            if '_' in datapoint_id:
                try:
                    parts = datapoint_id.rsplit('_', 1)
                    if len(parts) == 2:
                        # Convert back to path - handle both old and new directory structures
                        path_parts = parts[0].split('_')
                        
                        if path_parts[0] == 'RAGTest' and len(path_parts) >= 3:
                            # RAGTest_vertex_ai_filename -> RAGTest/vertex_ai/filename
                            source_file = f"{path_parts[0]}/{path_parts[1]}_{path_parts[2]}/" + "_".join(path_parts[3:])
                        elif path_parts[0] == 'RAGDocs' and len(path_parts) >= 2:
                            # RAGDocs_policies_filename -> RAGDocs/policies/filename
                            source_file = f"{path_parts[0]}/{path_parts[1]}/" + "_".join(path_parts[2:])
                        else:
                            # Fallback: replace all underscores with slashes
                            source_file = parts[0].replace('_', '/')
                        chunk_index = int(parts[1])
                        
                        # Read full document from AltaStata and re-chunk
                        try:
                            with fs.open(source_file, "r") as f:
                                full_content = f.read()
                            
                            # Re-chunk the document
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=4000,
                                chunk_overlap=800,
                                separators=["\n\n", "\n", ". ", " ", ""]
                            )
                            chunks = text_splitter.split_text(full_content)
                            
                            # Extract specific chunk
                            if chunk_index < len(chunks):
                                filename = os.path.basename(source_file)
                                content = chunks[chunk_index][:300]
                                
                                # Add ellipsis if truncated
                                if len(chunks[chunk_index]) > 300:
                                    content += "..."
                                
                                print(f"   {i}. 📄 {filename} (distance: {distance:.3f})")
                                print(f"      {content}")
                                print()
                        except Exception as e:
                            print(f"   ⚠️  Could not read {source_file}: {e}")
                            continue
                except (ValueError, IndexError):
                    print(f"   ⚠️  Skipping document with invalid datapoint ID: {datapoint_id}")
                    continue
    
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")
    
    # Cleanup
    print("\n🧹 Cleaning up test data...")
    try:
        # Delete all test documents from AltaStata
        for filename in sample_docs.keys():
            file_path = f"{test_dir}/{filename}"
            result = altastata_functions.delete_files(file_path, False, None, None)
            print(f"   ✅ Deleted from storage: {filename}")
        
        # Try to delete the directory
        try:
            altastata_functions.delete_files(test_dir, True, None, None)
            print(f"   ✅ Deleted directory: {test_dir}")
        except Exception:
            pass  # Directory might not be empty
        
        # Note: We keep vectors in Vertex AI Vector Search (metadata encoded in datapoint IDs)
        # Use cleanup.py to fully clean the Vertex AI resources
        print("   💡 Vectors remain in Vertex AI (use cleanup.py to remove)")
        
        # Clean up AltaStata Java process
        cleanup_altastata()
            
    except Exception as e:
        print(f"⚠️  Warning during cleanup: {e}")
        # Still try to clean up AltaStata even if other cleanup failed
        cleanup_altastata()
    
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
    print("      ├─ Text Embeddings (text-embedding-004, 768-dim)")
    print("      ├─ Vector Search (Matching Engine)")
    print("      └─ Gemini 2.5 Flash (LLM)")
    print()
    print("   🔗 Integration")
    print("      ├─ LangChain document processing")
    print("      ├─ Vertex AI Vector Search (production-ready)")
    print("      └─ Full source citations")
    
    print("\n💡 NEXT STEPS FOR PRODUCTION:")
    print("   🔐 Add user authentication and audit logging")
    print("   ☁️  Deploy to GKE with Confidential Computing")
    print("   📊 Implement monitoring and observability")
    print("   ⚡ Scale replicas for high availability")
    print(f"\n{'═' * 80}")


if __name__ == "__main__":
    try:
        test_rag_vertex_ai()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        cleanup_altastata()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        cleanup_altastata()
        import traceback
        traceback.print_exc()
        sys.exit(1)

