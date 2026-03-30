import time
import sys
import os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')
import altastata_config
from altastata.altastata_functions import PLAIN_CHUNK_MAX_SIZE, TEMP_FILE_THRESHOLD

af = altastata_config.altastata_functions

LOCAL_FILE = "/Users/sergevilvovsky/Movies/Кроличье сердце v8.mov"
CLOUD_PREFIX = "test_large_file/"

local_size = os.path.getsize(LOCAL_FILE)
print(f"Local file size: {local_size:,} bytes ({local_size / (1024**3):.2f} GB)")
print(f"Thresholds: Base64 single <={PLAIN_CHUNK_MAX_SIZE//1024//1024}MB, "
      f"Base64 stream <={TEMP_FILE_THRESHOLD//1024//1024}MB, "
      f"temp file >{TEMP_FILE_THRESHOLD//1024//1024}MB")

# --- Find the versioned cloud path ---
def find_cloud_path():
    versions = af.list_cloud_files_versions(CLOUD_PREFIX, False, None, None)
    for java_array in versions:
        for element in java_array:
            path = str(element)
            print(f"  Found: {path}")
            return path
    return None

print("\nListing cloud files...")
cloud_path = find_cloud_path()

if cloud_path is None:
    print("No file found in cloud. Uploading first...")
    t0 = time.time()
    af.store([LOCAL_FILE], "/Users/sergevilvovsky/Movies/", CLOUD_PREFIX, True)
    upload_sec = time.time() - t0
    print(f"Upload completed in {upload_sec:.1f}s ({local_size / upload_sec / (1024**2):.1f} MB/s)")
    print("\nListing again...")
    cloud_path = find_cloud_path()

print(f"\nCloud path: {cloud_path}")

# --- Get size ---
size_str = af.get_file_attribute(cloud_path, None, "size")
print(f"Cloud file size: {size_str}")
cloud_size = int(size_str) if size_str else 0

if cloud_size == 0:
    print("ERROR: Could not get file size")
    sys.exit(1)

# --- Test 1: Temp file approach ---
print(f"\n{'='*50}")
print(f"Test 1: TEMP FILE approach (streamToFile)")
print(f"{'='*50}")
t0 = time.time()
data = af.get_buffer(cloud_path, None, 0, 4, cloud_size)
tempfile_sec = time.time() - t0
print(f"Downloaded {len(data):,} bytes in {tempfile_sec:.1f}s ({len(data) / tempfile_sec / (1024**2):.1f} MB/s)")

with open(LOCAL_FILE, 'rb') as f:
    local_data = f.read()
if data == local_data:
    print("Verification: PASSED")
else:
    print(f"Verification: FAILED - local={len(local_data)}, downloaded={len(data)}")
del data, local_data

# --- Test 2: Base64 streaming (bypass threshold) ---
print(f"\n{'='*50}")
print(f"Test 2: BASE64 STREAMING approach (8MB chunks over Py4J)")
print(f"{'='*50}")
import base64
snapshot = int(time.time() * 1000)
t0 = time.time()
java_stream = af.altastata_file_system.getFileInputStream(cloud_path, snapshot, 0, 4)
try:
    buf = bytearray(cloud_size)
    view = memoryview(buf)
    offset = 0
    while offset < cloud_size:
        b64_str = af.altastata_file_system.readBufferFromInputStreamAsBase64(
            java_stream, min(PLAIN_CHUNK_MAX_SIZE, cloud_size - offset),
        )
        if b64_str is None:
            break
        chunk = base64.b64decode(b64_str)
        n = len(chunk)
        view[offset:offset + n] = chunk
        offset += n
        if offset % (200 * 1024 * 1024) < PLAIN_CHUNK_MAX_SIZE:
            elapsed = time.time() - t0
            print(f"  {offset / (1024**2):.0f} MB / {cloud_size / (1024**2):.0f} MB  ({offset / elapsed / (1024**2):.1f} MB/s)")
    data2 = bytes(buf[:offset])
finally:
    java_stream.close()
stream_sec = time.time() - t0
print(f"Downloaded {len(data2):,} bytes in {stream_sec:.1f}s ({len(data2) / stream_sec / (1024**2):.1f} MB/s)")

with open(LOCAL_FILE, 'rb') as f:
    local_data = f.read()
if data2 == local_data:
    print("Verification: PASSED")
else:
    print(f"Verification: FAILED - local={len(local_data)}, downloaded={len(data2)}")
del data2, local_data

# --- Summary ---
print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"Temp file:      {tempfile_sec:.1f}s  ({cloud_size / tempfile_sec / (1024**2):.1f} MB/s)")
print(f"Base64 stream:  {stream_sec:.1f}s  ({cloud_size / stream_sec / (1024**2):.1f} MB/s)")
print(f"Temp file is {stream_sec / tempfile_sec:.1f}x faster")

# --- Cleanup ---
print("\nDeleting cloud file...")
af.delete_files(CLOUD_PREFIX, True, None, None)
print("Done.")
