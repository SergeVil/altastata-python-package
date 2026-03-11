#!/usr/bin/env python3
"""
Simple fsspec test - basic functionality demonstration
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only what we need
from altastata.altastata_functions import AltaStataFunctions
from altastata.fsspec import create_filesystem

def test_basic_fsspec():
    """Test basic fsspec functionality with a small file."""
    print("🚀 Testing basic fsspec functionality")
    print("=" * 50)
    
    # Create AltaStata connection
    altastata_functions = AltaStataFunctions.from_account_dir('/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123')
    altastata_functions.set_password("123")
    
    # Create fsspec filesystem
    fs = create_filesystem(altastata_functions, "bob123")
    
    # Create a small test file
    test_content = b"Hello from fsspec!\nThis is a simple test file.\n"
    result = altastata_functions.create_file('StoreTest/simple_test.txt', test_content)
    print(f"✅ Created test file: {result.getOperationStateValue()}")
    
    test_file = "StoreTest/simple_test.txt"
    
    # Test file existence
    print(f"✅ File exists: {fs.exists(test_file)}")
    
    # Test file info
    try:
        info = fs.info(test_file)
        print(f"✅ File info: {info}")
    except Exception as e:
        print(f"❌ Error getting file info: {e}")
    
    # Test reading file
    print("\n📖 Testing file reading...")
    try:
        with fs.open(test_file, "r") as f:
            content = f.read()
            print(f"✅ Read content: {content[:50]}...")
            print(f"✅ Content length: {len(content)} characters")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
    
    # Test binary reading
    print("\n📖 Testing binary reading...")
    try:
        with fs.open(test_file, "rb") as f:
            content = f.read()
            print(f"✅ Read binary content: {len(content)} bytes")
    except Exception as e:
        print(f"❌ Error reading binary: {e}")
    
    # Test seeking
    print("\n🎯 Testing file seeking...")
    try:
        with fs.open(test_file, "r") as f:
            # Seek to middle
            f.seek(10)
            content = f.read(20)
            print(f"✅ Seek to position 10, read 20 chars: '{content}'")
            
            # Seek to end
            f.seek(0, 2)  # Seek to end
            pos = f.tell()
            print(f"✅ Seek to end, position: {pos}")
    except Exception as e:
        print(f"❌ Error with seeking: {e}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    try:
        result = altastata_functions.delete_files('StoreTest/simple_test.txt', False, None, None)
        print(f"✅ Cleanup: {result[0].getOperationStateValue()}")
    except Exception as e:
        print(f"⚠️  Error during cleanup: {e}")
    
    print("\n✅ Basic fsspec test completed successfully!")

if __name__ == "__main__":
    try:
        test_basic_fsspec()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
