#!/usr/bin/env python3
"""
Recreate Vertex AI Vector Search index with COSINE_DISTANCE
This fixes the similarity ranking issue by using proper cosine similarity
"""

import sys
import os

project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'altastata-coco')
location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')

print("=" * 80)
print("🔄 RECREATE INDEX WITH COSINE_DISTANCE")
print("=" * 80)
print(f"\n📍 Project: {project_id}")
print(f"📍 Location: {location}")
print("\n⚠️  WARNING: This will DELETE the existing index and all data!")
print("   You will need to re-run bob_indexer.py to re-index documents.")

confirm = input("\n🤔 Continue? (yes/no): ").strip().lower()
if confirm != 'yes':
    print("❌ Cancelled")
    sys.exit(0)

try:
    from google.cloud import aiplatform
    
    print("\n1️⃣  Initializing Vertex AI...")
    aiplatform.init(project=project_id, location=location)
    
    # Delete existing index (if any)
    print("\n2️⃣  Checking for existing index...")
    try:
        # Try to find existing index
        existing_indexes = aiplatform.MatchingEngineIndex.list()
        for index in existing_indexes:
            if "altastata-rag-docs" in index.display_name:
                print(f"   Found existing index: {index.display_name}")
                print(f"   Resource: {index.resource_name}")
                
                # Check if it's deployed
                try:
                    endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
                    for endpoint in endpoints:
                        if "altastata-rag-endpoint" in endpoint.display_name:
                            print(f"   Found endpoint: {endpoint.display_name}")
                            print("   ⚠️  You may need to undeploy the index first")
                            print("   ⚠️  Check the Google Cloud Console to manage deployments")
                            break
                except Exception as e:
                    print(f"   ⚠️  Could not check deployments: {e}")
                
                break
        else:
            print("   No existing index found")
    except Exception as e:
        print(f"   ⚠️  Could not list existing indexes: {e}")
    
    # Create new index with COSINE_DISTANCE
    print("\n3️⃣  Creating new Vector Search Index with COSINE_DISTANCE...")
    print("   (This takes 5-10 minutes...)")
    print("   Using COSINE_DISTANCE for better semantic similarity")
    
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="altastata-rag-docs-cosine",
        dimensions=768,  # text-embedding-004
        approximate_neighbors_count=10,
        distance_measure_type="COSINE_DISTANCE",  # 🎯 This is the key change!
        shard_size="SHARD_SIZE_SMALL",  # For e2-standard-2 compatibility
        index_update_method="STREAM_UPDATE",  # Enable real-time updates
        description="AltaStata RAG document embeddings with COSINE_DISTANCE (stream updates enabled)",
        labels={"app": "altastata-rag", "env": "dev", "metric": "cosine"}
    )
    
    print(f"✅ Index created: {index.resource_name}")
    
    # Create endpoint
    print("\n4️⃣  Creating Index Endpoint...")
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name="altastata-rag-endpoint-cosine",
        description="Endpoint for AltaStata RAG queries with COSINE_DISTANCE",
        public_endpoint_enabled=True,
        labels={"app": "altastata-rag", "env": "dev", "metric": "cosine"}
    )
    
    print(f"✅ Endpoint created: {endpoint.resource_name}")
    
    # Deploy index
    print("\n5️⃣  Deploying index to endpoint...")
    print("   (This takes 10-30 minutes...)")
    
    deployed_index = endpoint.deploy_index(
        index=index,
        deployed_index_id="altastata_rag_cosine",
        display_name="AltaStata RAG Documents (Cosine Distance)"
    )
    
    print(f"✅ Index deployed: {deployed_index.resource_name}")
    
    # Save configuration
    print("\n6️⃣  Saving configuration...")
    config_path = os.path.join(os.path.dirname(__file__), ".vertex_config")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f"INDEX_ID={index.resource_name}\n")
        f.write(f"ENDPOINT_ID={endpoint.resource_name}\n")
        f.write("DEPLOYED_INDEX_ID=altastata_rag_cosine\n")
        f.write("DISTANCE_METRIC=COSINE_DISTANCE\n")
    
    print(f"✅ Configuration saved to {config_path}")
    
    print("\n" + "=" * 80)
    print("🎉 INDEX RECREATED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📋 Next steps:")
    print("1. Run: python bob_indexer.py  # Re-index all documents")
    print("2. Run: python bob_query.py     # Test queries")
    print("\n💡 The new index uses COSINE_DISTANCE for better semantic similarity!")
    print("   This should fix the ranking issues you experienced.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
