#!/usr/bin/env python3
"""
Bob queries indexed documents from Vertex AI Vector Search
Based on test_rag_vertex.py query pattern
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_google_vertexai import VertexAIEmbeddings
import vertexai
from vertexai.preview.generative_models import GenerativeModel


def main():
    print("=" * 80)
    print("🔍 BOB QUERY - Vertex AI Vector Search")
    print("=" * 80)
    
    # Setup Vertex AI
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'altastata-coco')
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    print(f"\n📍 Project: {project_id}, Location: {location}")
    
    # Load config
    print("\n1️⃣  Loading Vertex AI configuration...")
    config_path = os.path.join(os.path.dirname(__file__), ".vertex_config")
    
    if not os.path.exists(config_path):
        print("❌ Vertex AI config not found!")
        print("   Run: python setup_vertex_search.py")
        return
    
    vertex_config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                vertex_config[key] = value
    
    # Load metadata
    metadata_path = "/tmp/bob_rag_metadata.json"
    if not os.path.exists(metadata_path):
        print("❌ No documents indexed yet!")
        print("   Run bob_indexer.py first, then alice_upload_docs.py")
        return
    
    with open(metadata_path, 'r') as f:
        document_metadata = json.load(f)
    
    print(f"✅ Loaded {len(document_metadata)} indexed chunks")
    
    # Initialize Vertex AI
    print("\n2️⃣  Initializing Vertex AI...")
    vertexai.init(project=project_id, location=location)
    
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=project_id,
        location=location
    )
    
    model = GenerativeModel("gemini-2.5-flash")
    print("✅ Gemini 2.5 Flash ready")
    
    # Load endpoint
    print("\n3️⃣  Connecting to Vertex AI Vector Search endpoint...")
    try:
        from google.cloud import aiplatform
        endpoint_name = vertex_config['ENDPOINT_ID']
        deployed_index_id = vertex_config['DEPLOYED_INDEX_ID']
        endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
        print("✅ Connected to endpoint")
    except Exception as e:
        print(f"❌ Failed to connect to endpoint: {e}")
        return
    
    # Query mode
    print("\n" + "=" * 80)
    print("Select mode:")
    print("  1. Interactive (ask your own questions)")
    print("  2. Demo (run 4 example queries)")
    mode = input("Choice (1 or 2): ").strip()
    
    def query_rag(query_text):
        """Query using Vertex AI Vector Search + Gemini"""
        # Generate query embedding
        query_embedding = embeddings.embed_query(query_text)
        
        # Query Vertex AI Vector Search
        response = endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query_embedding],
            num_neighbors=3
        )
        
        # Get top results
        neighbors = response[0] if response else []
        
        # Build context from metadata
        docs = []
        for neighbor in neighbors:
            datapoint_id = neighbor.id
            if datapoint_id in document_metadata:
                meta = document_metadata[datapoint_id]
                docs.append({
                    "text": meta["text"],
                    "filename": meta["filename"],
                    "source": meta["source"]
                })
        
        if not docs:
            return "No relevant documents found.", []
        
        # Build context for Gemini
        context = "\n\n".join([
            f"Document {i+1} ({doc['filename']}):\n{doc['text']}"
            for i, doc in enumerate(docs)
        ])
        
        # Query Gemini
        prompt = f"""Based on the following documents, answer the question concisely.

Context:
{context}

Question: {query_text}

Answer:"""
        
        response = model.generate_content(prompt)
        return response.text.strip(), docs
    
    if mode == "2":
        # Demo queries
        queries = [
            "What are the password requirements?",
            "How long do we keep financial records?",
            "What is the remote work policy for equipment?",
            "Which AI tools are approved for use?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'─' * 80}")
            print(f"📊 QUERY {i}/4")
            print(f"{'─' * 80}")
            print(f"❓ {query}\n")
            
            answer, docs = query_rag(query)
            
            print("🤖 ANSWER:")
            for line in answer.split('\n'):
                print(f"   {line}")
            
            print(f"\n📚 SOURCES ({len(docs)} documents):")
            for j, doc in enumerate(docs, 1):
                filename = doc['filename']
                preview = doc['text'][:80].replace('\n', ' ')
                print(f"   {j}. 📄 {filename}")
                print(f"      └─ {preview}...")
            print()
    
    else:
        # Interactive mode
        print("\n💡 Ask questions (type 'quit' to exit)")
        
        while True:
            query = input("\n❓ Your question: ").strip()
            
            if not query or query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            answer, docs = query_rag(query)
            
            print("\n🤖 ANSWER:")
            for line in answer.split('\n'):
                print(f"   {line}")
            
            print(f"\n📚 SOURCES: {len(docs)} documents")
            for i, doc in enumerate(docs, 1):
                print(f"   {i}. {doc['filename']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
