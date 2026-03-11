#!/usr/bin/env python3
"""
Alice uploads documents from sample_documents/ and shares with Bob
Clean, self-contained version focused on readability
"""

import sys
import os
import time
import signal
import atexit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altastata.altastata_functions import AltaStataFunctions


class AliceUploader:
    """Alice's document uploader - clean and self-contained"""
    
    def __init__(self):
        # State
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
        print("\n🛑 Cleaning up Alice...")
        if self.alice_altastata:
            try:
                self.alice_altastata.shutdown()
            except:
                pass
        print("✅ Alice cleanup complete")
    
    def _signal_handler(self, signum, frame):
        """Handle signals"""
        print(f"\n🛑 Received signal {signum}, cleaning up Alice...")
        self._cleanup()
        sys.exit(0)
    
    def _load_documents(self):
        """Load documents from sample_documents/"""
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
            return None
        
        return documents
    
    def _upload_and_share(self, documents):
        """Upload and share documents"""
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
            result = self.alice_altastata.create_file(file_path, content.encode('utf-8'))
            print(f"   ✅ {filename}: {result.getOperationStateValue()}")
        
        # Wait for files to be stored (async operation)
        print(f"\n⏳ Waiting for files to be fully stored...")
        time.sleep(1)
        
        # Step 2: Share all files
        print(f"\n4️⃣  Sharing {len(file_paths)} files with bob123...")
        for file_path, filename in file_paths:
            share_result = self.alice_altastata.share_files(
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
    
    def initialize(self):
        """Initialize Alice connection"""
        print("=" * 80)
        print("📤 ALICE - Upload & Share Documents")
        print("=" * 80)
        
        # Connect Alice
        print("\n1️⃣  Connecting Alice...")
        self.alice_altastata = AltaStataFunctions.from_account_dir(
            '/Users/sergevilvovsky/.altastata/accounts/azure.rsa.alice222',
            port=25555,
            enable_callback_server=False
        )
        self.alice_altastata.set_password("123")
        print("✅ Alice connected")
        
        return True
    
    def run(self):
        """Run the uploader"""
        if not self.initialize():
            return
        
        # Load documents
        documents = self._load_documents()
        if not documents:
            return
        
        # Upload and share
        self._upload_and_share(documents)
        
        print("\n" + "=" * 80)
        print("✅ All documents shared!")
        print("=" * 80)
        print("\n💡 Bob's indexer should have received events and indexed the documents")
        
        # Explicit cleanup before exit
        print("\n🛑 Cleaning up Alice...")
        try:
            self.alice_altastata.shutdown()
            print("✅ Alice cleanup complete")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


def main():
    """Main function"""
    try:
        uploader = AliceUploader()
        uploader.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()