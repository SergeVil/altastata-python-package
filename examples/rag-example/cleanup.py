#!/usr/bin/env python3
"""
Cleanup script with two modes:
1. Quick cleanup - Clear data only (metadata + files), keeps Vertex AI infrastructure
2. Full cleanup - Clear data + delete Vertex AI index/endpoint (requires 30 min to recreate)
Clean, self-contained version focused on readability
"""

import sys
import os
import time
import signal
import atexit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altastata.altastata_functions import AltaStataFunctions
from google.cloud import aiplatform


class CleanupService:
    """Cleanup service for RAG system - clean and self-contained"""
    
    def __init__(self):
        # State
        self.bob_altastata = None
        self.alice_altastata = None
        
        # Setup cleanup
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup signal and exit handlers"""
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _cleanup(self):
        """Cleanup function"""
        print("\n🛑 Final cleanup...")
        if self.bob_altastata:
            try:
                self.bob_altastata.shutdown()
                print("✅ Bob connection closed")
            except Exception:
                pass
        if self.alice_altastata:
            try:
                self.alice_altastata.shutdown()
                print("✅ Alice connection closed")
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
            return None
        
        vertex_config = {}
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    vertex_config[key] = value
        
        return vertex_config
    
    def _cleanup_storage(self, account_path: str, password: str, test_dir: str = "RAGDocs"):
        """Clean up AltaStata storage"""
        print(f"\n🗑️  Cleaning up {test_dir}/...")
        
        try:
            altastata = AltaStataFunctions.from_account_dir(
                account_path
            )
            altastata.set_password(password)
            
            # Store reference for cleanup
            if "bob123" in account_path:
                self.bob_altastata = altastata
            elif "alice222" in account_path:
                self.alice_altastata = altastata
            
            # Time window
            now_ms = int(time.time() * 1000)
            start_time = str(now_ms - 2000000000)
            end_time = str(now_ms + 60000)
            
            # List and delete files
            try:
                files_iterator = altastata.list_cloud_files_versions(
                    test_dir,
                    True,
                    start_time,
                    end_time
                )
                
                files = []
                for file_info in files_iterator:
                    if len(file_info) > 0:
                        files.append(file_info[0])
                
                if files:
                    print(f"   Found {len(files)} file(s)")
                    for file_path in files:
                        altastata.delete_files(file_path, False, start_time, end_time)
                        print(f"   ✅ Deleted: {file_path}")
                    
                    # Delete directory
                    try:
                        altastata.delete_files(test_dir, True, start_time, end_time)
                        print(f"   ✅ Deleted directory: {test_dir}")
                    except Exception:
                        pass
                else:
                    print("   ℹ️  No files found")
            
            except Exception:
                print("   ℹ️  Directory not found or empty")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    def _cleanup_metadata(self):
        """Clean up metadata file (legacy - no longer used)"""
        print("\n🗑️  Cleaning up legacy metadata...")
        metadata_path = "/tmp/bob_rag_metadata.json"
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            print("   ✅ Deleted legacy metadata file")
        else:
            print("   ℹ️  No legacy metadata file found")
    
    def _cleanup_vertex_index_data(self):
        """Clear all datapoints from Vertex AI Vector Search index"""
        print("\n🗑️  Clearing Vertex AI index data...")
        
        vertex_config = self._load_vertex_config()
        if not vertex_config:
            print("   ℹ️  No Vertex AI config found")
            return
        
        try:
            from google.cloud import aiplatform as gcp_aiplatform
            from google.cloud.aiplatform import matching_engine
            from langchain_google_vertexai import VertexAIEmbeddings
            
            # Initialize
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'altastata-coco')
            location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
            gcp_aiplatform.init(project=project_id, location=location)
            
            # Get the index
            index = matching_engine.MatchingEngineIndex(
                index_name=vertex_config['INDEX_ID'],
                project=project_id,
                location=location
            )
            
            print(f"   📊 Connected to index: {index.display_name}")
            
            # Get embeddings for broad search
            embeddings = VertexAIEmbeddings(
                model_name="text-embedding-004",
                project=project_id,
                location=location
            )
            
            # Get endpoint
            endpoint = matching_engine.MatchingEngineIndexEndpoint(
                index_endpoint_name=vertex_config['ENDPOINT_ID'],
                project=project_id,
                location=location
            )
            
            print("   🔍 Checking if index has data...")
            
            # Simple test query with timeout
            def timeout_handler(signum, frame):
                raise TimeoutError("Search timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            try:
                query_embedding = embeddings.embed_query("test")
                response = endpoint.find_neighbors(
                    deployed_index_id=vertex_config['DEPLOYED_INDEX_ID'],
                    queries=[query_embedding],
                    num_neighbors=10  # Just check if there's any data
                )
                signal.alarm(0)  # Cancel timeout
                
                neighbors = response[0] if response else []
                
                if not neighbors:
                    print("   ℹ️  Index is already empty - no data to clear")
                    return
                
                # Extract datapoint IDs from the test results
                all_datapoint_ids = set()
                for neighbor in neighbors:
                    if hasattr(neighbor, 'datapoint_id'):
                        all_datapoint_ids.add(neighbor.datapoint_id)
                    elif hasattr(neighbor, 'id'):
                        all_datapoint_ids.add(neighbor.id)
                        
            except TimeoutError:
                signal.alarm(0)
                print("   ⚠️  Search timed out - index might be empty or slow")
                print("   ℹ️  Skipping index cleanup (assuming empty)")
                return
            
            print(f"   📊 Found {len(all_datapoint_ids)} unique datapoint IDs")
            
            if all_datapoint_ids:
                print("   🗑️  Removing datapoints...")
                datapoint_ids_list = list(all_datapoint_ids)
                
                # Remove in batches (in case there are limits)
                batch_size = 100
                for i in range(0, len(datapoint_ids_list), batch_size):
                    batch = datapoint_ids_list[i:i+batch_size]
                    print(f"      Removing batch {i//batch_size + 1}: {len(batch)} datapoints")
                    
                    try:
                        index.remove_datapoints(datapoint_ids=batch)
                        print(f"      ✅ Removed {len(batch)} datapoints")
                    except Exception as e:
                        print(f"      ⚠️  Error removing batch: {e}")
                
                print("   ✅ Index data cleared!")
            else:
                print("   ℹ️  No datapoint IDs found - index might already be empty")
                
        except Exception as e:
            print(f"   ❌ Error clearing index data: {e}")
    
    def _cleanup_vertex_resources(self):
        """Delete Vertex AI Vector Search index and endpoint"""
        print("\n🗑️  Cleaning up Vertex AI Vector Search...")
        
        vertex_config = self._load_vertex_config()
        if not vertex_config:
            print("   ℹ️  No Vertex AI config found")
            return
        
        try:
            # Delete endpoint
            if 'ENDPOINT_ID' in vertex_config:
                print("   Deleting endpoint...")
                endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=vertex_config['ENDPOINT_ID']
                )
                endpoint.delete(force=True)
                print("   ✅ Endpoint deleted")
            
            # Delete index
            if 'INDEX_ID' in vertex_config:
                print("   Deleting index...")
                index = aiplatform.MatchingEngineIndex(
                    index_name=vertex_config['INDEX_ID']
                )
                index.delete()
                print("   ✅ Index deleted")
            
            # Delete config file
            config_path = os.path.join(os.path.dirname(__file__), ".vertex_config")
            os.remove(config_path)
            print("   ✅ Config file deleted")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    def run_quick_cleanup(self):
        """Run quick cleanup (data only)"""
        print("\n🧹 Quick cleanup mode...")
        
        # Clean up Bob's storage
        self._cleanup_storage(
            '/Users/sergevilvovsky/.altastata/accounts/azure.rsa.bob123',
            "123"
        )
        
        # Clean up Alice's storage
        self._cleanup_storage(
            '/Users/sergevilvovsky/.altastata/accounts/azure.rsa.alice222',
            "123"
        )
        
        # Clean up metadata
        self._cleanup_metadata()
        
        # Clear Vertex AI index data
        self._cleanup_vertex_index_data()
        
        print("\n✅ Vertex AI infrastructure preserved (quick cleanup mode)")
        print("   💡 Data cleared, but index still ready for new documents")
    
    def run_full_cleanup(self):
        """Run full cleanup (data + infrastructure)"""
        print("\n🧹 Full cleanup mode...")
        
        # Clean up data first
        self.run_quick_cleanup()
        
        # Clean up Vertex AI resources
        print("\n⚠️  You selected FULL cleanup - this will delete Vertex AI resources!")
        print("   (You'll need to run setup_vertex_search.py again, ~30 min)")
        final_confirm = input("   Really delete? (yes/no): ").strip().lower()
        
        if final_confirm == "yes":
            self._cleanup_vertex_resources()
        else:
            print("   ℹ️  Skipped Vertex AI cleanup")
    
    def run(self):
        """Run the cleanup service"""
        print("=" * 80)
        print("🧹 CLEANUP OPTIONS")
        print("=" * 80)
        
        print("\nSelect cleanup mode:")
        print("  1. Quick cleanup (clear data only)")
        print("     • Deletes AltaStata files")
        print("     • Clears Vertex AI index data (datapoints)")
        print("     • KEEPS Vertex AI infrastructure (instant)")
        print()
        print("  2. Full cleanup (clear data + infrastructure)")
        print("     • Everything from option 1")
        print("     • DELETES Vertex AI index & endpoint (⚠️  requires 20-40 min to recreate)")
        print()
        
        choice = input("Choice (1 or 2): ").strip()
        
        if choice not in ["1", "2"]:
            print("❌ Invalid choice. Cancelled.")
            return
        
        print("\n⚠️  This will delete:")
        print("   - All documents in RAGDocs/")
        print("   - All datapoints in Vertex AI index")
        if choice == "2":
            print("   - Vertex AI Vector Search index and endpoint")
        
        confirm = input("\nContinue? (yes/no): ").strip().lower()
        
        if confirm != "yes":
            print("❌ Cancelled")
            return
        
        if choice == "1":
            self.run_quick_cleanup()
        else:
            self.run_full_cleanup()
        
        print("\n" + "=" * 80)
        print("✅ Cleanup complete!")
        print("=" * 80)


def main():
    """Main function"""
    try:
        cleanup_service = CleanupService()
        cleanup_service.run()
    except KeyboardInterrupt:
        print("\n⚠️  Cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()