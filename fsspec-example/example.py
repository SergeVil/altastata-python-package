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
    
    # Get credentials
    account_id = os.environ.get("ALTASTATA_ACCOUNT_ID", "demo_account")
    account_dir = os.environ.get("ALTASTATA_ACCOUNT_DIR")
    password = os.environ.get("ALTASTATA_PASSWORD")
    
    if account_id == "demo_account":
        print("⚠ Demo mode - set environment variables for real usage:")
        print("  export ALTASTATA_ACCOUNT_ID='your_account'")
        print("  export ALTASTATA_ACCOUNT_DIR='/path/to/account'")
        print("  export ALTASTATA_PASSWORD='your_password'")
    
    # Create AltaStata connection
    try:
        if account_dir:
            altastata = AltaStataFunctions.from_account_dir(account_dir)
            if password:
                altastata.set_password(password)
            print("✓ AltaStata connection established")
        else:
            print("⚠ No account directory provided - demo mode")
            altastata = None
    except Exception as e:
        print(f"❌ Failed to connect to AltaStata: {e}")
        altastata = None
    
    # Create fsspec filesystem
    if altastata:
        try:
            fs = create_filesystem(altastata, account_id)
            print("✓ fsspec filesystem created")
            
            # Example operations
            print("\nAvailable operations:")
            print("- fs.ls('path')           # List files")
            print("- fs.exists('path')       # Check if file exists")
            print("- fs.open('path', 'r')    # Open file for reading")
            print("- fs.info('path')         # Get file information")
            
            # LangChain integration example
            if LANGCHAIN_AVAILABLE:
                print("\n✓ LangChain integration available:")
                print("from langchain_community.document_loaders import TextLoader")
                print("loader = TextLoader('altastata://Public/Documents/file.txt')")
            else:
                print("\n⚠ Install langchain-community for LangChain integration")
            
        except Exception as e:
            print(f"❌ Failed to create fsspec filesystem: {e}")
    else:
        print("⚠ Skipping filesystem creation (demo mode)")
    
    print("\n✓ Example completed!")


if __name__ == "__main__":
    main()
