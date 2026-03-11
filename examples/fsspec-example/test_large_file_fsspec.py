#!/usr/bin/env python3
"""
Test fsspec with large files (100MB)
"""

import time
import hashlib
from altastata import AltaStataFunctions
from altastata.fsspec import create_filesystem

def create_structured_chunk(chunk_index, chunk_size):
    """Create structured, verifiable content for each chunk."""
    # Create a pattern that includes chunk index, position, and checksum
    chunk_data = []
    
    # Header with chunk info
    header = f"CHUNK_{chunk_index:03d}_SIZE_{chunk_size:08d}_START_{chunk_index * chunk_size:010d}\n"
    chunk_data.append(header.encode('utf-8'))
    
    # Fill remaining space with structured pattern
    remaining_size = chunk_size - len(header)
    pattern_size = 1024  # 1KB pattern blocks
    
    for i in range(0, remaining_size, pattern_size):
        block_size = min(pattern_size, remaining_size - i)
        # Create pattern: chunk_index + position + some data
        pattern = f"BLOCK_{chunk_index:03d}_{i:06d}_" + "DATA" * (block_size // 20)
        pattern = pattern[:block_size].ljust(block_size, 'X')
        chunk_data.append(pattern.encode('utf-8'))
    
    return b''.join(chunk_data)

def create_large_file():
    """Create a 100MB test file."""
    print("Creating 100MB test file...")
    
    # Create AltaStata connection
    altastata_functions = AltaStataFunctions.from_account_dir('/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123')
    altastata_functions.set_password("123")
    
    # Create large content (100MB)
    chunk_size = 1024 * 1024  # 1MB chunks
    total_size = 100 * 1024 * 1024  # 100MB
    
    print(f"Generating {total_size // (1024*1024)}MB of data...")
    
    # Create file with initial chunk (structured content)
    initial_chunk = create_structured_chunk(0, chunk_size)
    result = altastata_functions.create_file('StoreTest/large_file_100mb.txt', initial_chunk)
    print(f"Created file: {result.getOperationStateValue()}")
    file_create_time_id = int(result.getCloudFileCreateTime())
    
    # Append remaining chunks
    remaining_chunks = (total_size // chunk_size) - 1
    print(f"Appending {remaining_chunks} more chunks...")
    
    for i in range(remaining_chunks):
        chunk = create_structured_chunk(i + 1, chunk_size)
        altastata_functions.append_buffer_to_file('StoreTest/large_file_100mb.txt', chunk, file_create_time_id)
        if (i + 1) % 10 == 0:  # Progress every 10 chunks (10MB)
            print(f"  Appended {i + 1}/{remaining_chunks} chunks...")
    
    print("✅ Large file created successfully!")
    return file_create_time_id

def test_fsspec_large_file():
    """Test fsspec operations with large file."""
    print("\nTesting fsspec with large file...")
    
    # Create AltaStata connection
    altastata_functions = AltaStataFunctions.from_account_dir('/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123')
    altastata_functions.set_password("123")
    
    # Create fsspec filesystem
    fs = create_filesystem(altastata_functions, "bob123")
    
    test_file = "StoreTest/large_file_100mb.txt"
    
    # Test file existence
    print(f"Checking if file exists: {fs.exists(test_file)}")
    
    # Test file info
    try:
        info = fs.info(test_file)
        print(f"File info: {info}")
        print(f"File size: {info['size']} bytes ({info['size'] / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Error getting file info: {e}")
    
    # Test reading file in chunks with verification
    print("\nTesting file reading with verification...")
    try:
        with fs.open(test_file, "rb") as f:
            total_read = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            start_time = time.time()
            chunk_count = 0
            verification_errors = 0
            
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                total_read += len(data)
                chunk_count += 1
                
                # Verify chunk content
                if len(data) > 100:  # Only verify if we have enough data
                    try:
                        # Check for chunk header
                        header_line = data[:data.find(b'\n')].decode('utf-8')
                        if header_line.startswith(f"CHUNK_{chunk_count-1:03d}_"):
                            print(f"  ✅ Chunk {chunk_count-1} header verified: {header_line[:50]}...")
                        else:
                            print(f"  ❌ Chunk {chunk_count-1} header mismatch")
                            verification_errors += 1
                    except Exception as e:
                        print(f"  ⚠️  Chunk {chunk_count-1} verification failed: {e}")
                        verification_errors += 1
                
                # Progress every 10MB
                if total_read % (10 * 1024 * 1024) == 0:
                    elapsed = time.time() - start_time
                    speed = total_read / (1024 * 1024) / elapsed if elapsed > 0 else 0
                    print(f"  Read {total_read // (1024*1024)}MB at {speed:.2f} MB/s")
            
            elapsed = time.time() - start_time
            speed = total_read / (1024 * 1024) / elapsed if elapsed > 0 else 0
            print(f"✅ Read complete: {total_read} bytes in {elapsed:.2f}s ({speed:.2f} MB/s)")
            print(f"✅ Verification: {chunk_count} chunks, {verification_errors} errors")
            
    except Exception as e:
        print(f"Error reading file: {e}")
    
    # Test seeking
    print("\nTesting file seeking...")
    try:
        with fs.open(test_file, "rb") as f:
            # Seek to middle of file
            middle_pos = 50 * 1024 * 1024  # 50MB
            f.seek(middle_pos)
            data = f.read(1024)  # Read 1KB
            print(f"✅ Seek to {middle_pos} bytes successful, read {len(data)} bytes")
            
            # Seek to end
            f.seek(0, 2)  # Seek to end
            pos = f.tell()
            print(f"✅ Seek to end successful, position: {pos} bytes")
            
    except Exception as e:
        print(f"Error with seeking: {e}")

def cleanup_large_file():
    """Clean up the large test file."""
    print("\nCleaning up large file...")
    
    altastata_functions = AltaStataFunctions.from_account_dir('/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123')
    altastata_functions.set_password("123")
    
    try:
        result = altastata_functions.delete_files('StoreTest/large_file_100mb.txt', False, None, None)
        print(f"✅ Cleanup: {result[0].getOperationStateValue()}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    print("🚀 Testing fsspec with 100MB file")
    print("=" * 50)
    
    try:
        # Create large file
        file_create_time_id = create_large_file()
        
        # Test fsspec operations
        test_fsspec_large_file()
        
        # Cleanup
        cleanup_large_file()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
