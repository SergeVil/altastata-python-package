#!/usr/bin/env python3
"""
Alice's File Operations Script

Alice (alice222) performs file operations that trigger events for Bob.

Usage:
    First, run bob_listener.py in another terminal
    Then run: python alice_sender.py
"""

import sys
import os
import time
from datetime import datetime

# Add the altastata package to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altastata.altastata_functions import AltaStataFunctions


def main():
    print("\n" + "=" * 80)
    print("👤 ALICE's FILE OPERATIONS")
    print("=" * 80)
    print("\nAlice (alice222) will perform file operations...")
    
    # Initialize Alice's AltaStata connection
    print("\n1️⃣  Connecting to AltaStata as alice222...")
    try:
        alice_altastata = AltaStataFunctions.from_account_dir(
            '/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.alice222',
            port=25555,  # Alice uses a different gateway port (separate JVM from Bob)
            enable_callback_server=False  # Alice doesn't need to listen for events
        )
        alice_altastata.set_password("123")
        print("✅ Alice connected successfully")
    except Exception as e:
        print(f"❌ Failed to connect as alice222: {e}")
        print("\nMake sure alice222 account exists at:")
        print("   /Users/sergevilvovsky/.altastata/accounts/amazon.rsa.alice222")
        print("\nIf not, create it first.")
        return
    
    # Define the test file path
    test_file_path = "SharedEvents/alice_to_bob.txt"
    file_content = f"""Hello Bob!

This is a test file created by Alice at {datetime.now()}.

This file is being shared with you to demonstrate AltaStata's 
event notification system.

- Alice creates this file
- Alice shares it with Bob
- Bob receives a FILE_SHARED event
- Alice then deletes the file
- Bob receives a FILE_DELETED event

Pretty cool, right? 🚀
""".encode('utf-8')
    
    print("\n" + "=" * 80)
    print("📝 STEP 1: Creating File")
    print("=" * 80)
    print(f"File path: {test_file_path}")
    print("Creating file...")
    
    # Use wide time window for share/delete operations (~23 days ago to 60 seconds in the future)
    now_ms = int(time.time() * 1000)
    file_start_time = str(now_ms - 2000000000)
    file_end_time = str(now_ms + 60000)  # 60 seconds in the future
    try:
        result = alice_altastata.create_file(test_file_path, file_content)
        print(f"✅ File created: {result.getOperationStateValue()}")
    except Exception as e:
        print(f"❌ Failed to create file: {e}")
        return
    
    print("\n" + "=" * 80)
    print("🤝 STEP 2: Sharing File with Bob")
    print("=" * 80)
    print(f"Sharing {test_file_path} with bob123...")
    
    try:
        # Share the SPECIFIC FILE (like the Java test)
        share_result = alice_altastata.share_files(
            cloud_path_prefix=test_file_path,       # Specific file path
            including_subdirectories=True,
            time_interval_start=file_start_time,    # Wide time window
            time_interval_end=file_end_time,        # Wide time window
            users=["bob123"]
        )
        
        print(f"✅ File shared successfully: {len(share_result)} version(s)")
        if share_result:
            print("   → Bob should receive a 'SHARE' event!")
        else:
            print("   ⚠️  No files were shared (empty result)")
    except Exception as e:
        print(f"⚠️  Share operation failed: {e}")
    
    # IMPORTANT: Wait between share and delete to allow Bob's event processor
    # to poll and detect the SHARE event before the file is deleted
    print("\n⏳ Waiting 15 seconds for Bob to receive SHARE event...")
    print("   (Bob's SecureCloudEventProcessor needs time to poll cloud storage)")
    time.sleep(15)
    
    print("\n" + "=" * 80)
    print("🗑️  STEP 3: Deleting File")
    print("=" * 80)
    print(f"Deleting {test_file_path}...")
    
    try:
        delete_result = alice_altastata.delete_files(
            cloud_path_prefix=test_file_path,     # Specific file path
            including_subdirectories=True,
            time_interval_start=file_start_time,  # Wide time window
            time_interval_end=file_end_time       # Wide time window
        )
        print(f"✅ File deleted successfully: {len(delete_result)} version(s)")
        print("   → Bob should receive a 'DELETE' event!")
    except Exception as e:
        print(f"❌ Failed to delete file: {e}")
    
    print("\n" + "=" * 80)
    print("✅ ALL OPERATIONS COMPLETED")
    print("=" * 80)
    print("\n📊 Summary:")
    print("   ✓ Created file")
    print("   ✓ Shared with bob123")
    print("   ✓ Deleted file")
    print("\n💡 Check Bob's terminal to see the events he received!")
    print("\n👋 Alice disconnected\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

