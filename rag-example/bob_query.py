#!/usr/bin/env python3
"""
Bob queries indexed documents from Vertex AI Vector Search
Clean, self-contained version focused on readability
"""

import sys
import os
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
        self.embeddings = None
        self.model = None
        self.endpoint = None
        self.fs = None
        self.bob_altastata = None
        
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
        if self.bob_altastata:
            try:
                self.bob_altastata.shutdown()
                print("✅ AltaStata Java process terminated")
            except Exception:
                pass
        print("✅ Cleanup complete")
    
    def _signal_handler(self, signum, _frame):
        """Handle signals"""
        print(f"\n🛑 Received signal {signum}, cleaning up...")
        self._cleanup()
        sys.exit(0)
    
    def _load_vertex_config(self):
        """Load Vertex AI configuration"""
        config_path = os.path.join(os.path.dirname(__file__), ".vertex_config")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Vertex AI config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    self.vertex_config[key] = value
    
    def _get_chunk_id(self, base_path, chunk_index):
        """Generate consistent chunk ID"""
        return f"{base_path.replace('/', '_')}_{chunk_index}"
    
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
    
    def _connect_to_altastata(self):
        """Connect to AltaStata for document retrieval"""
        from altastata.altastata_functions import AltaStataFunctions
        from altastata.fsspec import create_filesystem
        
        self.bob_altastata = AltaStataFunctions.from_account_dir(
            '/Users/sergevilvovsky/.altastata/accounts/azure.rsa.bob123',
            callback_server_port=25334
        )
        self.bob_altastata.set_password("123")
        self.fs = create_filesystem(self.bob_altastata, "bob123")
    
    def query_rag(self, query_text: str):
        """Query using Vertex AI Vector Search + AltaStata + Gemini"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query_text)
        
        # Query Vertex AI Vector Search
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.vertex_config['DEPLOYED_INDEX_ID'],
            queries=[query_embedding],
            num_neighbors=20  # Search even more neighbors to find new documents
        )
        
        # Get top results
        neighbors = response[0] if response else []
        
        if not neighbors:
            return "No relevant documents found.", []
        
        # Debug: Print what we found
        print(f"🔍 Found {len(neighbors)} neighbors:")
        for i, neighbor in enumerate(neighbors):
            datapoint_id = getattr(neighbor, 'id', '') or getattr(neighbor, 'datapoint_id', '')
            distance = getattr(neighbor, 'distance', 1.0)
            print(f"  {i+1}. Distance: {distance}")
            print(f"     Datapoint ID: {datapoint_id}")
            print()
        
        # With COSINE_DISTANCE, we can rely on proper semantic ranking
        # Take only the most relevant documents (lower distance = more similar)
        relevant_neighbors = neighbors.copy()
        relevant_neighbors.sort(key=lambda n: getattr(n, 'distance', 1.0))
        
        # Use a stricter threshold since COSINE_DISTANCE ranks correctly
        SIMILARITY_THRESHOLD = 0.5  # More strict - only very similar documents
        relevant_neighbors = [n for n in relevant_neighbors if getattr(n, 'distance', 1.0) < SIMILARITY_THRESHOLD]
        
        # If we don't have enough with the strict threshold, take top 2 most similar
        if len(relevant_neighbors) < 2:
            print(f"⚠️  Only {len(relevant_neighbors)} documents passed strict threshold, taking top 2 most similar")
            relevant_neighbors = neighbors.copy()
            relevant_neighbors.sort(key=lambda n: getattr(n, 'distance', 1.0))
            relevant_neighbors = relevant_neighbors[:2]
        
        print(f"🎯 Similarity threshold: {SIMILARITY_THRESHOLD}")
        print(f"📊 Relevant documents: {len(relevant_neighbors)}/{len(neighbors)}")
        
        if not relevant_neighbors:
            return "No relevant documents found.", []
        
        # Group by source file to avoid reading the same file multiple times
        source_files = {}
        for neighbor in relevant_neighbors:
            # Extract metadata from datapoint ID (format: source_file:chunk_index)
            datapoint_id = getattr(neighbor, 'id', '') or getattr(neighbor, 'datapoint_id', '')
            
            # Handle both formats: source_file:chunk_index and source_file_chunk_index
            if ':' in datapoint_id:
                # Format: source_file:chunk_index
                try:
                    source_file, chunk_index_str = datapoint_id.rsplit(':', 1)
                    chunk_index = int(chunk_index_str)
                except (ValueError, IndexError):
                    print(f"   ⚠️  Skipping document with invalid datapoint ID: {datapoint_id}")
                    continue
            elif '_' in datapoint_id:
                # Format: source_file_chunk_index (current format)
                try:
                    # Find the last underscore and split
                    parts = datapoint_id.rsplit('_', 1)
                    if len(parts) == 2:
                        # Convert back to path format: RAGDocs_policies_filename -> RAGDocs/policies/filename
                        source_file = parts[0].replace('_', '/', 2)  # Replace first two underscores
                        chunk_index = int(parts[1])
                    else:
                        print(f"   ⚠️  Skipping document with invalid datapoint ID: {datapoint_id}")
                        continue
                except (ValueError, IndexError):
                    print(f"   ⚠️  Skipping document with invalid datapoint ID: {datapoint_id}")
                    continue
            else:
                print(f"   ⚠️  Skipping document without proper datapoint ID: {datapoint_id}")
                continue
            
            # Only process documents that start with RAGDocs/
            if source_file.startswith('RAGDocs/'):
                if source_file not in source_files:
                    source_files[source_file] = []
                source_files[source_file].append(chunk_index)
                print(f"   ✅ Found valid document: {source_file} (chunk {chunk_index})")
            else:
                print(f"   ⚠️  Skipping document without proper path: {source_file}")
        
        # Retrieve content from AltaStata and re-chunk
        docs = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=800,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for source_file, chunk_indices in source_files.items():
            try:
                # Read full document from AltaStata
                with self.fs.open(source_file, "r") as f:
                    full_content = f.read()
                
                # Re-chunk the document
                chunks = text_splitter.split_text(full_content)
                
                # Extract specific chunks
                for chunk_index in chunk_indices:
                    if chunk_index < len(chunks):
                        docs.append({
                            "text": chunks[chunk_index],
                            "filename": os.path.basename(source_file),
                            "source": source_file
                        })
                        
            except Exception as e:
                print(f"⚠️  Could not read {source_file}: {e}")
                continue
        
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
            print(f"❓ QUESTION: {query}\n")
            
            answer, docs = self.query_rag(query)
            
            print("🤖 ANSWER:")
            print("─" * 40)
            for line in answer.split('\n'):
                print(f"   {line}")
            print("─" * 40)
            
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
            print("─" * 40)
            for line in answer.split('\n'):
                print(f"   {line}")
            print("─" * 40)
            
            print(f"\n📚 SOURCES: {len(docs)} documents")
            for i, doc in enumerate(docs, 1):
                print(f"   {i}. {doc['filename']}")
    
    def initialize(self):
        """Initialize all components"""
        print("=" * 80)
        print("🔍 BOB QUERY - Vertex AI Vector Search")
        print("=" * 80)
        print(f"\n📍 Project: {self.project_id}, Location: {self.location}")
        
        # Check GCP authentication
        print("\n🔐 Checking GCP authentication...")
        try:
            # Try to import and test Google Cloud libraries directly
            from google.auth import default
            from google.auth.exceptions import DefaultCredentialsError
            
            # This will work in cloud environments with service accounts
            credentials, project = default()
            print("✅ GCP authentication verified (service account or ADC)")
        except DefaultCredentialsError:
            # Fallback: try gcloud command (for local development)
            try:
                import subprocess
                result = subprocess.run(['gcloud', 'auth', 'application-default', 'print-access-token'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    print("❌ GCP authentication required!")
                    print("   Local: gcloud auth application-default login")
                    print("   Cloud: Ensure service account is attached")
                    return False
                print("✅ GCP authentication verified (gcloud)")
            except Exception:
                print("❌ GCP authentication required!")
                print("   Local: gcloud auth application-default login")
                print("   Cloud: Ensure service account is attached")
                return False
        except Exception as e:
            print(f"❌ GCP authentication error: {e}")
            print("   Local: gcloud auth application-default login")
            print("   Cloud: Ensure service account is attached")
            return False
        
        # Load config
        print("\n1️⃣  Loading Vertex AI configuration...")
        try:
            self._load_vertex_config()
            print("✅ Loaded config")
        except FileNotFoundError:
            print("❌ Vertex AI config not found!")
            print("   Run: python setup_vertex_search.py")
            return False
        
        # Connect to AltaStata
        print("\n2️⃣  Connecting to AltaStata...")
        try:
            self._connect_to_altastata()
            print("✅ Connected to AltaStata")
        except Exception as e:
            print(f"❌ Failed to connect to AltaStata: {e}")
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