#!/usr/bin/env python3
"""
Alice uploads documents from sample_documents/ and shares with Bob
Based on alice_sender.py pattern
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altastata.altastata_functions import AltaStataFunctions


def main():
    print("=" * 80)
    print("📤 ALICE - Upload & Share Documents")
    print("=" * 80)
    
    # Connect Alice
    print("\n1️⃣  Connecting Alice...")
    alice_altastata = AltaStataFunctions.from_account_dir(
        '/Users/sergevilvovsky/.altastata/accounts/azure.rsa.alice222',
        port=25555,
        enable_callback_server=False
    )
    alice_altastata.set_password("123")
    print("✅ Alice connected")
    
    # Load documents from sample_documents/
    print("\n2️⃣  Loading documents from sample_documents/...")
    sample_docs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_documents")
    
    documents = {}
    for filename in ["company_policy.txt", "security_guidelines.txt", 
                     "remote_work_policy.txt", "ai_usage_policy.txt"]:
        file_path = os.path.join(sample_docs_path, filename)
        try:
            with open(file_path, 'r') as f:
                documents[filename] = f.read()
            print(f"   ✅ Loaded: {filename}")
        except:
            print(f"   ⚠️  Skipped: {filename}")
    
    if not documents:
        print("❌ No documents found!")
        return
    
    # Upload and share (in parallel batches)
    test_dir = "RAGDocs/policies"
    now_ms = int(time.time() * 1000)
    time_start = str(now_ms - 2000000000)
    time_end = str(now_ms + 60000)
    
    print(f"\n3️⃣  Creating {len(documents)} files in parallel...")
    
    # Step 1: Create all files at once
    file_paths = []
    for filename, content in documents.items():
        file_path = f"{test_dir}/{filename}"
        file_paths.append((file_path, filename))
        result = alice_altastata.create_file(file_path, content.encode('utf-8'))
        print(f"   ✅ {filename}: {result.getOperationStateValue()}")
    
    # Wait once for all files to be stored (async operation)
    print(f"\n⏳ Waiting for files to be fully stored...")
    time.sleep(1)
    
    # Step 2: Share all files
    print(f"\n4️⃣  Sharing {len(file_paths)} files with bob123...")
    for file_path, filename in file_paths:
        share_result = alice_altastata.share_files(
            cloud_path_prefix=file_path,
            including_subdirectories=True,
            time_interval_start=time_start,
            time_interval_end=time_end,
            users=["bob123"]
        )
        if share_result:
            print(f"   ✅ {filename}: Shared ({len(share_result)} version)")
        else:
            print(f"   ⚠️  {filename}: No versions shared (file may not be ready yet)")
    
    print("\n" + "=" * 80)
    print("✅ All documents shared!")
    print("=" * 80)
    print("\n💡 Bob's indexer should have received events and indexed the documents")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
