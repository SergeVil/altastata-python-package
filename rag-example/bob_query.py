#!/usr/bin/env python3
"""
Bob queries indexed documents from Vertex AI Vector Search
Clean, self-contained version focused on readability
"""

import sys
import os
import json
import signal
import atexit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_google_vertexai import VertexAIEmbeddings
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import aiplatform


class BobQuery:
    """Bob's query interface - clean and self-contained"""
    
    def __init__(self):
        # Configuration
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'altastata-coco')
        self.location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        # State
        self.vertex_config = {}
        self.document_metadata = {}
        self.embeddings = None
        self.model = None
        self.endpoint = None
        
        # Setup cleanup
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup signal and exit handlers"""
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _cleanup(self):
        """Cleanup function"""
        print("\n🛑 Cleaning up...")
        print("✅ Cleanup complete")
    
    def _signal_handler(self, signum, frame):
        """Handle signals"""
        print(f"\n🛑 Received signal {signum}, cleaning up...")
        self._cleanup()
        sys.exit(0)
    
    def _load_vertex_config(self):
        """Load Vertex AI configuration"""
        config_path = os.path.join(os.path.dirname(__file__), ".vertex_config")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Vertex AI config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    self.vertex_config[key] = value
    
    def _load_metadata(self):
        """Load document metadata"""
        metadata_path = "/tmp/bob_rag_metadata.json"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("No documents indexed yet!")
        
        with open(metadata_path, 'r') as f:
            self.document_metadata = json.load(f)
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI resources"""
        vertexai.init(project=self.project_id, location=self.location)
        
        self.embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=self.project_id,
            location=self.location
        )
        
        self.model = GenerativeModel("gemini-2.5-flash")
    
    def _connect_to_endpoint(self):
        """Connect to Vertex AI Vector Search endpoint"""
        endpoint_name = self.vertex_config['ENDPOINT_ID']
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
    
    def query_rag(self, query_text: str):
        """Query using Vertex AI Vector Search + Gemini"""
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query_text)
        
        # Query Vertex AI Vector Search
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.vertex_config['DEPLOYED_INDEX_ID'],
            queries=[query_embedding],
            num_neighbors=3
        )
        
        # Get top results
        neighbors = response[0] if response else []
        
        # Build context from metadata
        docs = []
        for neighbor in neighbors:
            datapoint_id = neighbor.id
            if datapoint_id in self.document_metadata:
                meta = self.document_metadata[datapoint_id]
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
        
        response = self.model.generate_content(prompt)
        return response.text.strip(), docs
    
    def run_demo_queries(self):
        """Run demo queries"""
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
            
            answer, docs = self.query_rag(query)
            
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
    
    def run_interactive(self):
        """Run interactive query mode"""
        print("\n💡 Ask questions (type 'quit' to exit)")
        
        while True:
            query = input("\n❓ Your question: ").strip()
            
            if not query or query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            answer, docs = self.query_rag(query)
            
            print("\n🤖 ANSWER:")
            for line in answer.split('\n'):
                print(f"   {line}")
            
            print(f"\n📚 SOURCES: {len(docs)} documents")
            for i, doc in enumerate(docs, 1):
                print(f"   {i}. {doc['filename']}")
    
    def initialize(self):
        """Initialize all components"""
        print("=" * 80)
        print("🔍 BOB QUERY - Vertex AI Vector Search")
        print("=" * 80)
        print(f"\n📍 Project: {self.project_id}, Location: {self.location}")
        
        # Load config
        print("\n1️⃣  Loading Vertex AI configuration...")
        try:
            self._load_vertex_config()
            print("✅ Loaded config")
        except FileNotFoundError:
            print("❌ Vertex AI config not found!")
            print("   Run: python setup_vertex_search.py")
            return False
        
        # Load metadata
        print("\n2️⃣  Loading metadata...")
        try:
            self._load_metadata()
            print(f"✅ Loaded {len(self.document_metadata)} indexed chunks")
        except FileNotFoundError:
            print("❌ No documents indexed yet!")
            print("   Run bob_indexer.py first, then alice_upload_docs.py")
            return False
        
        # Initialize Vertex AI
        print("\n3️⃣  Initializing Vertex AI...")
        self._initialize_vertex_ai()
        print("✅ Gemini 2.5 Flash ready")
        
        # Connect to endpoint
        print("\n4️⃣  Connecting to Vertex AI Vector Search endpoint...")
        try:
            self._connect_to_endpoint()
            print("✅ Connected to endpoint")
        except Exception as e:
            print(f"❌ Failed to connect to endpoint: {e}")
            return False
        
        return True
    
    def run(self):
        """Run the query interface"""
        if not self.initialize():
            return
        
        # Query mode
        print("\n" + "=" * 80)
        print("Select mode:")
        print("  1. Interactive (ask your own questions)")
        print("  2. Demo (run 4 example queries)")
        mode = input("Choice (1 or 2): ").strip()
        
        if mode == "2":
            self.run_demo_queries()
        else:
            self.run_interactive()


def main():
    """Main function"""
    try:
        query_interface = BobQuery()
        query_interface.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()