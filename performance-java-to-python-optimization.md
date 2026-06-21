# Performance Optimization: Java-to-Python Data Transfer

## Problem Statement

The AltaStata Python package transfers data from cloud storage through a Java/Scala core into Python. Profiling revealed several bottlenecks in this pipeline that caused excessive memory usage, unnecessary data copies, and underutilized parallelism. For files ≤ 64 MB, all data transfers happen in-memory (no decrypted data on disk). For files > 64 MB, a temporary file with strict cleanup is used for performance.

## Architecture

```
Cloud Storage
    │
    ▼
┌──────────────────────────────────┐
│  Java/Scala Core (JVM)           │
│  ┌────────────────────────────┐  │
│  │ AltaStataChunkedInputStream│  │
│  │  • 8 MB encrypted chunks   │  │
│  │  • Parallel download pool  │  │
│  │    (40 threads)            │  │
│  │  • Decrypt + decompress    │  │
│  │  • In-memory chunk cache   │  │
│  │  • Expanded-range prefetch │  │
│  └────────────────────────────┘  │
│              │                   │
│  ┌────────────────────────────┐  │
│  │ AltaStataFileSystem API    │  │
│  │  • getBufferAsBase64()     │  │
│  │  • readBufferFromInput     │  │
│  │    StreamAsBase64()        │  │
│  │  • streamToFile()          │  │
│  │  • getBuffer()             │  │
│  │  • getFileInputStream()    │  │
│  └────────────────────────────┘  │
└──────────────┬───────────────────┘
               │  Legacy bridge (TCP socket)
               │  Base64 String transfer
               ▼
┌──────────────────────────────────┐
│  Python (altastata package)      │
│  • AltaStataFunctions            │
│  •   get_buffer() — three-tier    │
│  •     ≤8 MB: single Base64 call │
│  •     8–64 MB: streaming chunks │
│  •     >64 MB: temp file         │
│  •   get_java_input_stream()     │
│  • PyTorch / TensorFlow datasets │
│  •   LRU cache (_cache_put)      │
│  • fsspec filesystem             │
│  •   in-memory buffer via        │
│  •   get_buffer()                │
└──────────────────────────────────┘
```

## Bottlenecks Identified

### 1. Legacy bridge byte[] serialization (Critical)

The legacy bridge's internal handling of Java `byte[]` is slow. Even though it uses Base64 internally for byte arrays, the encoding/decoding and string concatenation overhead in the protocol layer makes large `byte[]` transfers very expensive (~5-20 MB/s). In contrast, returning a pre-encoded `String` from Java bypasses this overhead entirely — string values transfer as a single bulk UTF-8 blob (~500+ MB/s).

**Before:** All data transferred as `byte[]` through the legacy bridge's slow internal encoding.

### 2. Decrypted data written to local filesystem without cleanup (Security)

The old temp-file approach (`fillMappedFile`, `fillMappedFileDirect`) wrote decrypted plaintext to local disk without guaranteed cleanup. The new `streamToFile` approach (used only for >64 MB files) writes to a `tempfile.mkstemp` file and deletes it immediately in a `try...finally` block.

### 3. Python mmap path copies immediately (High)

The Python side opened the file Java wrote, created an mmap, then immediately copied the entire mmap content into a `bytes` object and deleted the file — defeating the purpose of memory mapping.

### 4. ~~Background chunk prefetch~~ (Removed — redundant)

The Java `AltaStataChunkedInputStream` had a commented-out `Future` block for prefetching the next chunk window. Analysis showed that `retrieveChunksWithExpandedRange` already expands each request to `readChunksTogether` chunks (default 8 = 64 MB window). The background prefetch mostly re-downloaded already-cached chunks (7 of 8 overlap) while competing for network/CPU/thread-pool resources with the main download. It also had a TOCTOU race on the guard flag. Removed.

### 5. fsspec uses 1024-byte reads (Medium)

`AltaStataFile.read(-1)` called `get_buffer_from_input_stream(stream, 1024)` — for a 10 MB file this would require ~10,000 bridge round-trips. Also, `seek()` used Java `mark()` incorrectly (mark is for reset, not seeking).

### 6. TensorFlow training bypasses cache (Medium)

The `_load_and_preprocess` map function called `_read_from_altastata` with `should_cache=False`, and `_read_from_altastata` itself never checked the cache before fetching from cloud.

### 7. No cache eviction policy (Medium)

Both PyTorch and TensorFlow dataset caches used a plain `dict` with a hard size limit. Once the cache filled up, new files could not be cached.

### 8. Cache size accounting bug on duplicate keys (Minor)

Cache insertion did not check if a path was already cached before inserting, causing `current_cache_size` to be inflated on duplicate inserts.

### 9. Version parsing duplicated in Python (Minor)

Both datasets parsed the `✹` version suffix from file paths to extract timestamps. Java already handles versioned paths internally — the Python parsing was redundant. Python now passes `None` for `snapshotTime`, which gets converted to current time on both sides.

## Optimizations Applied

### Phase 1: Core Transfer Layer (`altastata_functions.py`)

#### Base64 String transfer (bypasses legacy bridge byte[] bottleneck)

The single biggest win. We Base64-encode on the Java side and return a `String`, which the legacy bridge transfers efficiently as a single bulk value. This bypasses slow internal `byte[]` handling entirely.

```java
// Java: single-call Base64 for small files
public String getBufferAsBase64(String cloudFilePath, ...) throws IOException {
    byte[] data = getBuffer(cloudFilePath, ...);
    return Base64.getEncoder().encodeToString(data);
}
```

#### Tiered read strategy in `get_buffer`

`get_buffer` now has a three-tier strategy based on file size:

| File Size | Strategy | Why |
|-----------|----------|-----|
| ≤ 8 MB (1 chunk) | `getBufferAsBase64()` — single Base64 call | One chunk — no pipelining benefit; minimal round-trips |
| 8–64 MB | Streaming Base64 per chunk via `readBufferFromInputStreamAsBase64()` | Independent chunks pipelined; bounded Java heap (~19 MB); no temp file |
| > 64 MB | `streamToFile()` — Java streams to temp file, Python reads + deletes | Faster for large files (1.2x vs streaming); avoids 267+ bridge round-trips for a 2 GB file |

The streaming path uses a pre-allocated `bytearray` with `memoryview` to avoid reallocations. The temp file path uses Java's `streamToFile()` which copies directly from `AltaStataChunkedInputStream` to `FileOutputStream` in 8 MB chunks (never loads the full file into Java heap):

```python
def get_buffer(self, cloudFilePath, snapshotTime, startPosition,
               howManyChunksInParallel, size):
    if snapshotTime is None:
        snapshotTime = int(time.time() * 1000)
    if size <= PLAIN_CHUNK_MAX_SIZE:  # 8 MB
        return base64.b64decode(
            self.altastata_file_system.getBufferAsBase64(...))
    if size > TEMP_FILE_THRESHOLD:  # 64 MB
        return self._get_buffer_via_temp_file(...)
    # 8–64 MB: streaming Base64 chunks
    java_stream = self.altastata_file_system.getFileInputStream(...)
    try:
        buf = bytearray(size)
        view = memoryview(buf)
        ...
    finally:
        java_stream.close()
```

```java
// Java: stream directly to file without heap buffering
public long streamToFile(String outputFilePath, String cloudFilePath,
        Long snapshotTime, Long startPosition, int howManyChunksInParallel)
        throws IOException {
    try (InputStream in = getFileInputStream(...);
         FileOutputStream out = new FileOutputStream(outputFilePath)) {
        byte[] buf = new byte[Constants.PLAIN_CHUNK_MAX_SIZE()];
        long total = 0;
        int n;
        while ((n = in.read(buf)) != -1) {
            out.write(buf, 0, n);
            total += n;
        }
        return total;
    }
}
```

#### Security: controlled temp-file usage with strict cleanup

Files ≤ 64 MB transfer entirely in-memory through the legacy gateway — no decrypted data touches disk. The old `fillMappedFile` / `fillMappedFileDirect` / `get_buffer_via_mapped_file` methods (which left temp files around) were removed.

For files > 64 MB, the new `streamToFile` approach creates a temporary file that is immediately deleted after Python reads it (in a `try...finally` block). The temp file is created via `tempfile.mkstemp` with a unique name and removed even if an exception occurs. This trades a brief disk presence for significantly better performance (1.2x faster than pure streaming for large files).

### Phase 2: Java/Scala Cleanup

#### `fillMappedFile` / `fillMappedFileDirect` — removed; `streamToFile` added

The old temp-file methods were deleted from `AltaStataFileSystem.java` — they wrote decrypted plaintext to local disk without cleanup guarantees. Replaced by `streamToFile`, which streams chunks directly from `AltaStataChunkedInputStream` to `FileOutputStream` in 8 MB buffers (never loads the full file into Java heap). Python handles strict cleanup (immediate delete).

#### `snapshotTime` null guard (Java + Python)

Java's `getFileInputStream` now guards against `null` `snapshotTime` (which Python sends as `None`):
```java
if (snapshotTime == null) {
    snapshotTime = System.currentTimeMillis();
}
```
Python-side methods (`get_buffer`, `get_file_attribute`, `get_java_input_stream`) also explicitly convert `None` to `int(time.time() * 1000)` before calling Java. This belt-and-suspenders approach prevents `NullPointerException` when the Scala `AltaStataChunkedInputStream` unboxes `Long` to `long`.

#### Background prefetch in `SecureCloudStream.scala` — removed

The redundant `Future` block and `@volatile backgroundFetchInProgress` field were removed. The existing `retrieveChunksWithExpandedRange` provides sufficient read-ahead.

### Phase 3: Dataset Layer

#### LRU cache eviction (both PyTorch and TensorFlow)

Caches now use `collections.OrderedDict` with LRU eviction. `_cache_put` handles duplicate-key accounting correctly:

```python
def _cache_put(self, path, data):
    data_len = len(data)
    if data_len > self.max_file_size_for_cache:
        return
    if path in self.file_content_cache:
        self.current_cache_size -= len(self.file_content_cache.pop(path))
    while self.current_cache_size + data_len > self.cache_size_limit and self.file_content_cache:
        _, evicted_data = self.file_content_cache.popitem(last=False)
        self.current_cache_size -= len(evicted_data)
    self.file_content_cache[path] = data
    self.current_cache_size += data_len
```

Cache reads use `move_to_end()` to mark entries as recently used:

```python
if path in self.file_content_cache:
    self.file_content_cache.move_to_end(path)
    return self.file_content_cache[path]
```

#### File reads use `get_file_attribute` + `get_buffer`

Both datasets get the file size via `get_file_attribute`, then call `get_buffer` which uses the tiered strategy (single Base64 for ≤8 MB, streaming for >8 MB). This is two control calls per file, but enables streaming for large files.

#### Version parsing removed from Python

Both datasets no longer parse the `✹` version suffix from paths. `None` is passed for `snapshotTime` — Java handles versioned paths internally.

#### TensorFlow: cache and code cleanup

- `_read_from_altastata` checks the LRU cache first, fixing the bypass bug
- The duplicate `_read_file_py` inner function was removed; `_load_and_preprocess` now calls `self._read_file` directly
- `should_cache` parameter was removed — always caches

### Phase 4: fsspec Layer

#### In-memory buffer via `get_buffer`

`AltaStataFile` lazily loads the full file content on first `read()` using `get_file_attribute` + `get_buffer` (streaming for >8 MB). Subsequent reads and seeks operate on the in-memory buffer.

This replaces the original design (eager InputStream with broken seek and 1024-byte reads) with correct seek semantics (all `whence` modes) at the cost of holding the full content in Python memory.

```python
def _ensure_content(self):
    if self._content is None:
        af = self.filesystem.altastata_functions
        size_str = af.get_file_attribute(self.path, None, "size")
        try:
            size = int(size_str) if size_str else 0
        except (ValueError, TypeError):
            size = 0
        self._content = af.get_buffer(self.path, None, 0, 4, size)
```

For callers that need true streaming without loading everything into memory, `get_java_input_stream` is available as a low-level API.

## Performance Impact

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| **10 MB file via legacy bridge** | `byte[]` through slow bridge handling (~5-20 MB/s) | Base64 `String` (~500+ MB/s) | **~25-100x faster transfer** |
| **64 MB file, Java heap** | ~150 MB (byte[] + bridge overhead) | ~19 MB (one 8 MB chunk + Base64 at a time) | **~8x less Java memory** |
| **2.1 GB file via temp file** | Base64 streaming: 234s / 9.3 MB/s (267 round-trips) | `streamToFile` + temp file: 196s / 11.1 MB/s | **1.2x faster, fewer round-trips** |
| **fsspec `read(-1)` on 10 MB file** | ~10,000 round-trips (1 KB each) | 1-2 calls via `get_buffer` | **~10,000x fewer round-trips** |
| **TF training, 2nd+ epoch** | Re-fetches all files from cloud | Served from LRU cache | **Near-zero I/O** |
| **Cache full, new files needed** | New files cannot be cached | LRU eviction makes room | **Continuous caching** |

### Benchmark: 2.12 GB video file (real cloud transfer)

| Operation | Time | Throughput |
|-----------|------|------------|
| **Upload** | 121.6s | 17.8 MB/s |
| **Download (temp file, >64 MB tier)** | 195.9s | 11.1 MB/s |
| **Download (Base64 streaming, forced)** | 234.0s | 9.3 MB/s |

Both download methods verified data integrity against the local source file. The temp file approach avoids 267 round-trips (one per 8 MB chunk) and lets Java write contiguously to disk at I/O speed.

## Why Base64 String instead of byte[] on the legacy bridge?

The legacy bridge uses a text-based protocol. While it does handle `byte[]` arrays (via internal Base64), the overhead of string concatenation and encoding/decoding within the protocol layer makes large `byte[]` transfers slow. By Base64-encoding on the Java side and returning a `String`, we bypass this overhead — strings transfer as a single bulk UTF-8 blob.

This limitation is specific to the old bridge layer. Other Java interop mechanisms transfer byte arrays efficiently. The Java API exposes both `getBuffer()` (raw byte[]) and `getBufferAsBase64()` (String) so clients can choose the optimal path for their bridge.

| Bridge type | Best method | Why |
|-------------|-------------|-----|
| Legacy Python bridge | `getBufferAsBase64` / `readBufferFromInputStreamAsBase64` | String is bulk, byte[] is slow |
| JNI / native (.NET IKVM) | `getBuffer` | byte[] via pointer, no serialization |
| gRPC / REST | `getBuffer` | Binary frames, no legacy bridge overhead |
| Shared memory | Direct write | Zero-copy |

## Files Changed

### Python (`altastata-python-package/altastata/`)

| File | Changes |
|------|---------|
| `altastata_functions.py` | `get_buffer()` rewritten with three-tier strategy (≤8 MB single Base64 call, 8–64 MB streaming, >64 MB temp file via `streamToFile`). Added `_get_buffer_via_temp_file()`, `PLAIN_CHUNK_MAX_SIZE`, `TEMP_FILE_THRESHOLD` constants. All methods guard `snapshotTime=None` → `int(time.time() * 1000)`. Removed `get_buffer_via_mapped_file`, `get_file_with_size`, `get_buffer_from_input_stream`, `read_input_stream_position`, `mark_input_stream_position`, `import mmap`. |
| `altastata_pytorch_dataset.py` | `_read_file()` uses `get_file_attribute` + `get_buffer`. Added `_cache_put()` with LRU eviction via `OrderedDict`. Removed version parsing, temp file path, `import tempfile`. |
| `altastata_tensorflow_dataset.py` | `_read_from_altastata()` uses `get_file_attribute` + `get_buffer`, checks LRU cache first. Added `_cache_put()`. Removed version parsing, temp file path, `_read_file_py` duplication, `should_cache` parameter, `import tempfile`. |
| `fsspec.py` | `AltaStataFile` rewritten: lazy content loading via `get_file_attribute` + `get_buffer`, correct seek (all whence modes). Removed eager InputStream, broken mark-based seek. |

### Java (`altastata-core/src/main/java/com/altastata/api/`)

| File | Changes |
|------|---------|
| `AltaStataFileSystem.java` | Added `getBufferAsBase64()`, `readBufferFromInputStreamAsBase64()`, `streamToFile()`. Added `snapshotTime` null guard in `getFileInputStream()`. Removed `fillMappedFile`, `fillMappedFileDirect`, `getFileWithSize`. |

### Scala (`altastata-core/src/main/scala/...securecloud/`)

| File | Changes |
|------|---------|
| `SecureCloudStream.scala` | Removed redundant background prefetch (`Future` block), removed `@volatile backgroundFetchInProgress` field. |

## Future Optimizations

### Batch multi-file reads

A Java method like `getFilesWithSize(String[] paths)` that returns multiple files in one call would reduce round-trips proportionally for dataset loading.

### Bypass legacy bridge for data transfer

The `streamToFile` temp-file approach already bypasses the legacy bridge for the bulk data path (>64 MB), reducing it to a single control call. For further improvement:
- **Unix domain socket**: Java writes raw bytes, Python reads directly. ~3-5 GB/s.
- **`memfd_create`** (Linux): anonymous memory-only file descriptors that never touch disk. Zero-copy, secure.
- **gRPC sidecar**: binary protobuf frames with built-in streaming. Well-supported in both languages.

### Non-streaming retrieval for known-size bulk reads

The Scala `retrieveCloudFileContent` with `isStreaming=false` launches all chunk `Future`s at once (up to the 40-thread pool). Exposing a dedicated Java API for this bulk mode would maximize download parallelism for dataset loading.

### Cross-worker shared cache for PyTorch DataLoader

When `num_workers > 0`, each DataLoader worker process has its own cache. Using `multiprocessing.shared_memory` to share a single cache across workers would eliminate redundant cloud reads.
