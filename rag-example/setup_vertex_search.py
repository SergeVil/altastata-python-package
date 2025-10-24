#!/usr/bin/env python3
"""
Setup Vertex AI Vector Search infrastructure

Run this ONCE before using bob_indexer.py

This creates:
1. Vertex AI Matching Engine Index
2. Index Endpoint
3. Deploys index to endpoint
"""

import sys
import os

project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'altastata-coco')
location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')

print("=" * 80)
print("🔧 SETUP VERTEX AI VECTOR SEARCH")
print("=" * 80)
print(f"\n📍 Project: {project_id}")
print(f"📍 Location: {location}")

try:
    from google.cloud import aiplatform
    
    print("\n1️⃣  Initializing Vertex AI...")
    aiplatform.init(project=project_id, location=location)
    
    # Create index
    print("\n2️⃣  Creating Vector Search Index...")
    print("   (This takes 5-10 minutes...)")
    print("   Using SMALL shard size for demo (< 10K vectors)")
    
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="altastata-rag-docs",
        dimensions=768,  # text-embedding-004
        approximate_neighbors_count=10,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        shard_size="SHARD_SIZE_SMALL",  # For e2-standard-2 compatibility
        index_update_method="STREAM_UPDATE",  # Enable real-time updates
        description="AltaStata RAG document embeddings (stream updates enabled)",
        labels={"app": "altastata-rag", "env": "dev"}
    )
    
    print(f"✅ Index created: {index.resource_name}")
    
    # Create endpoint
    print("\n3️⃣  Creating Index Endpoint...")
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name="altastata-rag-endpoint",
        description="Endpoint for AltaStata RAG queries",
        public_endpoint_enabled=True,
        labels={"app": "altastata-rag", "env": "dev"}
    )
    
    print(f"✅ Endpoint created: {endpoint.resource_name}")
    
    # Deploy index
    print("\n4️⃣  Deploying index to endpoint...")
    print("   (This takes 10-30 minutes...)")
    
    import time
    deployed_index_id = f"altastata_rag_{int(time.time())}"  # Unique ID
    
    endpoint.deploy_index(
        index=index,
        deployed_index_id=deployed_index_id,
        display_name="AltaStata RAG Index",
        machine_type="e2-standard-2",
        min_replica_count=1,
        max_replica_count=1
    )
    
    print("✅ Index deployed!")
    
    # Save config
    config_path = os.path.join(os.path.dirname(__file__), ".vertex_config")
    with open(config_path, 'w') as f:
        f.write(f"PROJECT_ID={project_id}\n")
        f.write(f"LOCATION={location}\n")
        f.write(f"INDEX_ID={index.name}\n")
        f.write(f"ENDPOINT_ID={endpoint.name}\n")
        f.write(f"DEPLOYED_INDEX_ID={deployed_index_id}\n")
    
    print(f"\n💾 Config saved: {config_path}")
    
    print("\n" + "=" * 80)
    print("✅ SETUP COMPLETE!")
    print("=" * 80)
    print("\n📝 Next: python bob_indexer.py")
    
except ImportError:
    print("❌ google-cloud-aiplatform not installed")
    print("   pip install google-cloud-aiplatform")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

