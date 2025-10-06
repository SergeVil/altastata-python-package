#!/usr/bin/env python3
"""
Simple AltaStata fsspec Example

This example shows basic usage of the fsspec filesystem interface.
"""

import fsspec
from altastata import AltaStataFileSystem


def main():
    """Run basic fsspec example."""
    print("AltaStata fsspec Example")
    print("=" * 30)
    
    # Register filesystem
    AltaStataFileSystem.register()
    print("✓ Filesystem registered")
    
    # Create filesystem instance (demo mode)
    print("\nCreating filesystem instance...")
    print("Note: This is a demo. For real usage, provide actual credentials:")
    print("fs = fsspec.filesystem('altastata', account_id='your_account')")
    
    try:
        # This would work with real credentials
        fs = fsspec.filesystem("altastata", account_id="demo")
        print("✓ Filesystem created")
        
        # Example operations
        print("\nAvailable operations:")
        print("- fs.ls('path')           # List files")
        print("- fs.exists('path')       # Check if file exists")
        print("- fs.open('path', 'r')    # Open file for reading")
        print("- fs.info('path')         # Get file information")
        
        print("\n✓ Example completed!")
        print("\nFor LangChain integration:")
        print("from langchain_community.document_loaders import TextLoader")
        print("loader = TextLoader('altastata://Public/Documents/file.txt')")
        
    except Exception as e:
        print(f"⚠ Demo mode: {e}")
        print("✓ Example structure shown successfully!")


if __name__ == "__main__":
    main()
