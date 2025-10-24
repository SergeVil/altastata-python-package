#!/usr/bin/env python3
"""
Bob listens for SHARE events and indexes documents to Vertex AI Vector Search
Clean, self-contained version focused on readability
"""

import sys
import os
import time
import threading
import queue
import signal
import atexit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altastata.altastata_functions import AltaStataFunctions
from altastata.fsspec import create_filesystem
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document
import vertexai
from google.cloud import aiplatform


class BobIndexer:
    """Bob's event-driven indexer - clean and self-contained"""
    
    def __init__(self):
        # Configuration
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'altastata-coco')
        self.location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        # State
        self.embeddings = None
        self.fs = None
        self.vertex_index = None
        self.vertex_config = {}
        self.processed_files = set()
        
        # Event processing
        self.event_queue = queue.Queue()
        self.processing_thread = None
        self.stop_processing = False
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
        print("\n🛑 Cleaning up Bob...")
        self.stop_processing = True
        if self.bob_altastata:
            try:
                self.bob_altastata.shutdown()
            except Exception:
                pass
        print("✅ Bob cleanup complete")
    
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
    
    def _event_handler(self, event_name, data):
        """Bob's event handler - queues events for sequential processing"""
        thread_name = threading.current_thread().name
        print(f"\n🔔 EVENT QUEUED: {event_name} (Thread: {thread_name})")
        self.event_queue.put((event_name, data))
    
    def _process_event_queue(self):
        """Process events from the queue sequentially"""
        while not self.stop_processing:
            try:
                event_name, data = self.event_queue.get(timeout=1.0)
                
                print("\n" + "=" * 80)
                print(f"🔔 PROCESSING: {event_name}")
                print("=" * 80)
                
                if event_name == "SHARE":
                    self._process_share_event(data)
                elif event_name == "DELETE":
                    print(f"🗑️  File deleted: {data}")
                
                print("=" * 80)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing event: {e}")
                import traceback
                traceback.print_exc()
    
    def _process_share_event(self, data):
        """Process a SHARE event"""
        file_path = str(data)
        base_path = file_path.split('✹')[0] if '✹' in file_path else file_path
        
        print(f"📄 File: {file_path}")
        
        # Check if already processed
        if base_path in self.processed_files:
            print("   ⏭️  Skipping (already indexed)")
            return
        
        self.processed_files.add(base_path)
        
        try:
            # Read from encrypted storage
            print("   1️⃣  Reading file...")
            with self.fs.open(file_path, "r") as f:
                content = f.read()
            print(f"   ✅ Loaded: {len(content)} chars")
            
            # Create document and chunk
            doc = Document(
                page_content=content,
                metadata={"source": file_path, "filename": os.path.basename(file_path)}
            )
            
            print("   2️⃣  Chunking...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=800,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents([doc])
            print(f"   ✅ Created {len(chunks)} chunks")
            
            # Generate embeddings
            print("   3️⃣  Generating embeddings...")
            texts = [chunk.page_content for chunk in chunks]
            chunk_embeddings = self.embeddings.embed_documents(texts)
            
            # Prepare datapoints for Vertex AI
            print("   4️⃣  Upserting to Vertex AI Vector Search...")
            datapoints = []
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                datapoint_id = self._get_chunk_id(base_path, i)
                
                datapoints.append({
                    "datapoint_id": datapoint_id,
                    "feature_vector": embedding
                })
            
            # Upsert to Vertex AI
            self.vertex_index.upsert_datapoints(datapoints=datapoints)
            print(f"   ✅ Indexed {len(datapoints)} chunks to Vertex AI Vector Search!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def initialize(self):
        """Initialize all components"""
        print("=" * 80)
        print("🤖 BOB INDEXER - Vertex AI Vector Search (Event-Driven)")
        print("=" * 80)
        print(f"\n📍 Project: {self.project_id}, Location: {self.location}")
        
        # Load Vertex AI config
        print("\n1️⃣  Loading Vertex AI Vector Search configuration...")
        try:
            self._load_vertex_config()
            print(f"✅ Loaded config - Index: {self.vertex_config.get('INDEX_ID', 'N/A')}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("\n📝 Run this first: python setup_vertex_search.py")
            return False
        
        # Initialize Vertex AI
        print("\n2️⃣  Initializing Vertex AI...")
        vertexai.init(project=self.project_id, location=self.location)
        
        self.embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=self.project_id,
            location=self.location
        )
        print("✅ Embeddings ready (text-embedding-004, 768-dim)")
        
        # Connect to Vertex AI index
        try:
            index_name = self.vertex_config['INDEX_ID']
            self.vertex_index = aiplatform.MatchingEngineIndex(index_name=index_name)
            print("✅ Connected to Vertex AI Vector Search")
        except Exception as e:
            print(f"❌ Failed to connect to Vertex AI Vector Search: {e}")
            return False
        
        # No local metadata needed - everything stored in Vertex AI
        
        # Connect to AltaStata
        print("\n3️⃣  Connecting Bob...")
        self.bob_altastata = AltaStataFunctions.from_account_dir(
            '/Users/sergevilvovsky/.altastata/accounts/azure.rsa.bob123',
            callback_server_port=25334
        )
        self.bob_altastata.set_password("123")
        self.fs = create_filesystem(self.bob_altastata, "bob123")
        print("✅ Bob connected")
        
        # Start event processing
        print("\n4️⃣  Starting event processing thread...")
        self.processing_thread = threading.Thread(target=self._process_event_queue, daemon=True)
        self.processing_thread.start()
        print("✅ Event processor started")
        
        # Register event listener
        print("\n5️⃣  Registering event listener...")
        self.bob_altastata.add_event_listener(self._event_handler)
        print("✅ Listening for SHARE events")
        
        return True
    
    def run(self):
        """Run the indexer"""
        if not self.initialize():
            return
        
        print("\n" + "=" * 80)
        print("🎧 BOB IS LISTENING...")
        print("=" * 80)
        print("\n💡 Run: python alice_upload_docs.py")
        print("⏳ Waiting for events... (Ctrl+C to stop)\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping...")
            self.stop_processing = True
            if self.bob_altastata:
                try:
                    self.bob_altastata.remove_event_listener(self._event_handler)
                except Exception:
                    pass
            if self.processing_thread:
                self.processing_thread.join(timeout=5)
            if self.bob_altastata:
                self.bob_altastata.shutdown()
            print("✅ Stopped")


def main():
    """Main function"""
    try:
        indexer = BobIndexer()
        indexer.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()