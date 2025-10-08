#!/usr/bin/env python3
"""
Simple AltaStata fsspec Example

This example shows basic usage of the fsspec filesystem interface.
"""

import os
from altastata import AltaStataFunctions
from altastata.fsspec import create_filesystem

# Try to import LangChain components
try:
    from langchain_community.document_loaders import TextLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def main():
    """Run basic fsspec example."""
    print("AltaStata fsspec Example")
    print("=" * 30)
    
    # Use the same credentials pattern as test_script.py
    account_dir = '/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.alice222'
    password = "123"
    
    # Create AltaStata connection
    try:
        altastata = AltaStataFunctions.from_account_dir(account_dir)
        altastata.set_password(password)
        print("✓ AltaStata connection established")
    except Exception as e:
        print(f"❌ Failed to connect to AltaStata: {e}")
        altastata = None
    
    # Create fsspec filesystem
    if altastata:
        try:
            fs = create_filesystem(altastata, "alice222")
            print("✓ fsspec filesystem created")
            
            # Test basic operations
            print("\nTesting fsspec operations:")
            
            # First, create a test file using AltaStataFunctions
            test_content = b"Hello from fsspec example!\nThis is a test file created for fsspec integration.\n"
            result = altastata.create_file("StoreTest/fsspec_test.txt", test_content)
            print(f"✓ Created test file: {result.getOperationStateValue()}")
            
            # List files in StoreTest directory
            try:
                files = fs.ls("StoreTest")
                print(f"✓ Found {len(files)} files in StoreTest:")
                for file in files:
                    print(f"  - {file}")
            except Exception as e:
                print(f"⚠ Could not list files: {e}")
            
            # Test file existence
            test_file = "StoreTest/fsspec_test.txt"
            if fs.exists(test_file):
                print(f"✓ File exists: {test_file}")
                
                # Get file info
                try:
                    info = fs.info(test_file)
                    print(f"✓ File info: {info}")
                except Exception as e:
                    print(f"⚠ Could not get file info: {e}")
                
                # Read file content
                try:
                    with fs.open(test_file, "r") as f:
                        content = f.read(100)  # Read first 100 characters
                        print(f"✓ File content (first 100 chars): {content[:100]}...")
                except Exception as e:
                    print(f"⚠ Could not read file: {e}")
            else:
                print(f"⚠ File does not exist: {test_file}")
            
            # Clean up test file
            try:
                altastata.delete_files("StoreTest/fsspec_test.txt", False, None, None)
                print("✓ Cleaned up test file")
            except Exception as e:
                print(f"⚠ Could not clean up: {e}")
            
            # LangChain integration example
            if LANGCHAIN_AVAILABLE:
                print("\n✓ LangChain integration available:")
                print("from langchain_community.document_loaders import TextLoader")
                print("loader = TextLoader('altastata://StoreTest/meeting_saved_chat.txt')")
            else:
                print("\n⚠ Install langchain-community for LangChain integration")
            
        except Exception as e:
            print(f"❌ Failed to create fsspec filesystem: {e}")
    else:
        print("⚠ Skipping filesystem creation (demo mode)")
    
    print("\n✓ Example completed!")


if __name__ == "__main__":
    main()
