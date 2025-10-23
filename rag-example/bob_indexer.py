#!/usr/bin/env python3
"""
Bob listens for SHARE events and indexes documents to Vertex AI Vector Search
Based on bob_listener.py + test_rag_vertex.py patterns
"""

import sys
import os
import time
import json
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


# Global state
embeddings = None
fs = None
processed_files = set()  # Track processed files to avoid duplicates
vertex_index = None
vertex_config = {}
document_metadata = {}  # Store metadata for each datapoint ID

# Event processing queue
event_queue = queue.Queue()
processing_thread = None
stop_processing = False
bob_altastata = None  # Global reference for cleanup


def cleanup_on_exit():
    """Cleanup function called on exit"""
    global stop_processing, bob_altastata
    if bob_altastata:
        print("\n🛑 Cleaning up on exit...")
        stop_processing = True
        try:
            bob_altastata.shutdown()
        except:
            pass
        print("✅ Cleanup complete")


def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT"""
    print(f"\n🛑 Received signal {signum}, cleaning up...")
    cleanup_on_exit()
    sys.exit(0)


# Register cleanup handlers
atexit.register(cleanup_on_exit)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def bob_event_handler(event_name, data):
    """Bob's event handler - queues events for sequential processing"""
    import threading
    thread_name = threading.current_thread().name
    print(f"\n🔔 EVENT QUEUED: {event_name} (Thread: {thread_name})")
    
    # Queue the event for sequential processing
    event_queue.put((event_name, data))


def process_event_queue():
    """Process events from the queue sequentially"""
    global embeddings, fs, processed_files, vertex_index, document_metadata, stop_processing
    
    while not stop_processing:
        try:
            # Get event from queue (blocking with timeout)
            event_name, data = event_queue.get(timeout=1.0)
            
            print("\n" + "=" * 80)
            print(f"🔔 PROCESSING: {event_name}")
            print("=" * 80)
            
            if event_name == "SHARE":
                # data is already a string (file path), not a Java object
                file_path = str(data)
                
                # Extract base filename (without version suffix like ✹alice222_1761243866388)
                base_path = file_path.split('✹')[0] if '✹' in file_path else file_path
                
                print(f"📄 File: {file_path}")
                
                # Check if already processed
                if base_path in processed_files:
                    print(f"   ⏭️  Skipping (already indexed)")
                    print("=" * 80)
                    continue
                
                processed_files.add(base_path)
                
                try:
                    # Read from encrypted storage
                    print("   1️⃣  Reading file...")
                    with fs.open(file_path, "r") as f:
                        content = f.read()
                    print(f"   ✅ Loaded: {len(content)} chars")
                    
                    # Create document
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path, "filename": os.path.basename(file_path)}
                    )
                    
                    # Chunk
                    print("   2️⃣  Chunking...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=4000,
                        chunk_overlap=800,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )
                    chunks = text_splitter.split_documents([doc])
                    print(f"   ✅ Created {len(chunks)} chunks")
                    
                    # Generate embeddings and upsert to Vertex AI Vector Search
                    print("   3️⃣  Generating embeddings...")
                    texts = [chunk.page_content for chunk in chunks]
                    chunk_embeddings = embeddings.embed_documents(texts)
                    
                    print("   4️⃣  Upserting to Vertex AI Vector Search...")
                    datapoints = []
                    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                        datapoint_id = f"{base_path.replace('/', '_')}_{i}"
                        
                        # Store metadata separately (Vertex AI Vector Search doesn't store text)
                        document_metadata[datapoint_id] = {
                            "text": chunk.page_content,
                            "source": file_path,
                            "filename": os.path.basename(file_path),
                            "base_path": base_path
                        }
                        
                        datapoints.append({
                            "datapoint_id": datapoint_id,
                            "feature_vector": embedding
                        })
                    
                    # Upsert to Vertex AI
                    vertex_index.upsert_datapoints(datapoints=datapoints)
                    print(f"   ✅ Indexed {len(datapoints)} chunks to Vertex AI Vector Search!")
                    
                    # Save metadata to disk
                    metadata_path = "/tmp/bob_rag_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(document_metadata, f)
                    print(f"   💾 Metadata saved to {metadata_path}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif event_name == "DELETE":
                print(f"🗑️  File deleted: {data}")
            
            print("=" * 80)
            
        except queue.Empty:
            # Timeout - continue loop
            continue
        except Exception as e:
            print(f"❌ Error processing event: {e}")
            import traceback
            traceback.print_exc()


def main():
    global embeddings, fs, vertex_index, vertex_config, document_metadata, processing_thread, stop_processing, bob_altastata
    
    print("=" * 80)
    print("🤖 BOB INDEXER - Vertex AI Vector Search (Event-Driven)")
    print("=" * 80)
    
    # Setup Vertex AI
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'altastata-coco')
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    print(f"\n📍 Project: {project_id}, Location: {location}")
    
    # Load Vertex AI Vector Search config
    print("\n1️⃣  Loading Vertex AI Vector Search configuration...")
    config_path = os.path.join(os.path.dirname(__file__), ".vertex_config")
    
    if not os.path.exists(config_path):
        print(f"❌ Vertex AI config not found: {config_path}")
        print(f"\n📝 Run this first:")
        print(f"   python setup_vertex_search.py")
        return
    
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                vertex_config[key] = value
    
    print(f"✅ Loaded config")
    print(f"   Index: {vertex_config.get('INDEX_ID', 'N/A')}")
    
    # Initialize Vertex AI
    print("\n2️⃣  Initializing Vertex AI...")
    vertexai.init(project=project_id, location=location)
    
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=project_id,
        location=location
    )
    print("✅ Embeddings ready (text-embedding-004, 768-dim)")
    
    # Load Vertex AI index
    try:
        from google.cloud import aiplatform
        index_name = vertex_config['INDEX_ID']
        vertex_index = aiplatform.MatchingEngineIndex(index_name=index_name)
        print("✅ Connected to Vertex AI Vector Search")
    except Exception as e:
        print(f"❌ Failed to connect to Vertex AI Vector Search: {e}")
        return
    
    # Load existing metadata if available
    metadata_path = "/tmp/bob_rag_metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                document_metadata = json.load(f)
            print(f"✅ Loaded existing metadata ({len(document_metadata)} chunks)")
        except:
            print("⚠️  Could not load existing metadata")
    
    # Connect Bob
    print("\n3️⃣  Connecting Bob...")
    bob_altastata = AltaStataFunctions.from_account_dir(
        '/Users/sergevilvovsky/.altastata/accounts/azure.rsa.bob123',
        callback_server_port=25334
    )
    bob_altastata.set_password("123")
    fs = create_filesystem(bob_altastata, "bob123")
    print("✅ Bob connected")
    
    # Start event processing thread
    print("\n4️⃣  Starting event processing thread...")
    processing_thread = threading.Thread(target=process_event_queue, daemon=True)
    processing_thread.start()
    print("✅ Event processor started")
    
    # Register event listener
    print("\n5️⃣  Registering event listener...")
    listener = bob_altastata.add_event_listener(bob_event_handler)
    print("✅ Listening for SHARE events")
    
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
        stop_processing = True
        bob_altastata.remove_event_listener(listener)
        if processing_thread:
            processing_thread.join(timeout=5)
        # Properly shutdown the gateway to kill Java process
        bob_altastata.shutdown()
        print("✅ Stopped")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
