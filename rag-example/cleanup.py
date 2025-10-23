#!/usr/bin/env python3
"""
Cleanup script with two modes:
1. Quick cleanup - Clear data only (metadata + files), keeps Vertex AI infrastructure
2. Full cleanup - Clear data + delete Vertex AI index/endpoint (requires 30 min to recreate)
"""

import sys
import os
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altastata.altastata_functions import AltaStataFunctions


def cleanup_storage(account_path, password, test_dir="RAGDocs"):
    """Clean up AltaStata storage"""
    print(f"\n🗑️  Cleaning up {test_dir}/...")
    
    try:
        altastata = AltaStataFunctions.from_account_dir(
            account_path,
            enable_callback_server=False,
            port=25666
        )
        altastata.set_password(password)
        
        # Time window
        now_ms = int(time.time() * 1000)
        start_time = str(now_ms - 2000000000)
        end_time = str(now_ms + 60000)
        
        # List files
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
                    result = altastata.delete_files(file_path, False, start_time, end_time)
                    print(f"   ✅ Deleted: {file_path}")
                
                # Delete directory
                try:
                    altastata.delete_files(test_dir, True, start_time, end_time)
                    print(f"   ✅ Deleted directory: {test_dir}")
                except:
                    pass
            else:
                print(f"   ℹ️  No files found")
        
        except:
            print(f"   ℹ️  Directory not found or empty")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


def cleanup_vertex_resources():
    """Delete Vertex AI Vector Search index and endpoint"""
    print("\n🗑️  Cleaning up Vertex AI Vector Search...")
    
    config_path = os.path.join(os.path.dirname(__file__), ".vertex_config")
    
    if not os.path.exists(config_path):
        print("   ℹ️  No Vertex AI config found")
        return
    
    # Load config
    vertex_config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                vertex_config[key] = value
    
    try:
        from google.cloud import aiplatform
        
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
        os.remove(config_path)
        print("   ✅ Config file deleted")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


def main():
    print("=" * 80)
    print("🧹 CLEANUP OPTIONS")
    print("=" * 80)
    
    print("\nSelect cleanup mode:")
    print("  1. Quick cleanup (clear data only)")
    print("     • Deletes AltaStata files")
    print("     • Deletes metadata file")
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
    print("   - Local metadata (/tmp/bob_rag_metadata.json)")
    if choice == "2":
        print("   - Vertex AI Vector Search index and endpoint")
    
    confirm = input("\nContinue? (yes/no): ").strip().lower()
    
    if confirm != "yes":
        print("❌ Cancelled")
        return
    
    # Clean up Bob's storage
    cleanup_storage(
        '/Users/sergevilvovsky/.altastata/accounts/azure.rsa.bob123',
        "123"
    )
    
    # Clean up Alice's storage
    cleanup_storage(
        '/Users/sergevilvovsky/.altastata/accounts/azure.rsa.alice222',
        "123"
    )
    
    # Clean up metadata
    print("\n🗑️  Cleaning up metadata...")
    metadata_path = "/tmp/bob_rag_metadata.json"
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
        print(f"   ✅ Deleted metadata file")
    else:
        print(f"   ℹ️  No metadata file found")
    
    # Clean up Vertex AI (if full cleanup)
    if choice == "2":
        print("\n⚠️  You selected FULL cleanup - this will delete Vertex AI resources!")
        print("   (You'll need to run setup_vertex_search.py again, ~30 min)")
        final_confirm = input("   Really delete? (yes/no): ").strip().lower()
        
        if final_confirm == "yes":
            cleanup_vertex_resources()
        else:
            print("   ℹ️  Skipped Vertex AI cleanup")
    else:
        print("\n✅ Vertex AI infrastructure preserved (quick cleanup mode)")
        print("   💡 Data cleared, but index still ready for new documents")
    
    print("\n" + "=" * 80)
    print("✅ Cleanup complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
