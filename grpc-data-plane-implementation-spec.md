# Implementation Spec: gRPC Data-Plane Optimization (Phase 1)

**Audience:** an implementing engineer/model executing the change.
**Repo:** `altastata-python-package`
**Companion docs:** `grpc-migration-design.md` (decision & phases),
`performance-java-to-python-optimization.md` (legacy tiering background),
`GRPC_VS_PY4J_BENCHMARK.md` (benchmark commands).

This spec is **self-contained**. Do exactly what is written here. Do **not**
redesign, do **not** add new dependencies, do **not** introduce Apache Arrow.

---

## 0. Context (read first)

We are standardizing the Java ↔ Python transport on **gRPC**. This task is
**Phase 1: harden the gRPC data plane** in the Python
client. It is purely client-side Python (the Java gateway already implements
the `ReadStream` RPC — see `tests/js-grpc-ui` "read stream").

**Data-plane decision (already made — do not revisit):**
- The transport moves **opaque encrypted file bytes**, not structured/columnar
  data. Therefore **gRPC binary streaming is sufficient**.
- **Apache Arrow / Arrow Flight is explicitly out of scope.** It helps only for
  columnar/tabular data and adds no throughput for byte blobs.
- For remote reads, the bottleneck is cloud network + decryption (~11 MB/s in
  our 2 GB benchmark), not the local hop, so do not chase zero-copy beyond what
  is specified here.

**Constraints:**
- Do **not** change any public method signature in
  `altastata/altastata_functions.py`.
- Do **not** flip the default transport in this task (that is a later phase).
- Do **not** modify the Java side (it lives in another repo, `mycloud`).
- Do **not** run `git commit` or `git push` (assistant git policy).

---

## 1. Tasks overview

| # | Task | Priority | Risk |
|---|------|----------|------|
| 1 | Stream large reads in `AltaStataGrpcClient.get_buffer` (no 64 MB cap) | High | Low |
| 2 | Remove the redundant `bytes(resp.data)` copy on the unary path | High | Trivial |
| 3 | Add optional Unix-domain-socket endpoint to `GrpcEndpoint` (TCP stays default) | Medium | Low |
| 4 | Tests for streaming assembly + endpoint target | High | Low |

All changes are in **`altastata/grpc_client.py`** except the tests.

---

## 2. Task 1 + 2 — Streaming `get_buffer` and copy removal

### 2.1 Current code (for reference)

The current unary implementation in `altastata/grpc_client.py`:

```376:399:altastata/grpc_client.py
    def create_file(self, file_path: str, content: bytes) -> Dict[str, str]:
        req = self._fileops_pb2.CreateFileRequest(file_path=file_path, content=content)
        resp = self._fileops_stub.CreateFile(req, metadata=self._metadata)
        return self._status_to_dict(resp.status)

    def get_buffer(
        self,
        file_path: str,
        size: int,
        snapshot_time: int = 0,
        start_position: int = 0,
        parallel_chunks: int = 4,
        trust_cached_size: bool = False,
    ) -> bytes:
        req = self._fileops_pb2.GetBufferRequest(
            file_path=file_path,
            snapshot_time=snapshot_time,
            start_position=start_position,
            parallel_chunks=parallel_chunks,
            size=size,
            trust_cached_size=trust_cached_size,
        )
        resp = self._fileops_stub.GetBuffer(req, metadata=self._metadata)
        return bytes(resp.data)
```

The streaming RPC already exists and is used by `read_stream`:

```493:511:altastata/grpc_client.py
    def read_stream(
        self,
        cloud_file_path: str,
        snapshot_time: Optional[int] = None,
        start_position: int = 0,
        parallel_chunks: int = 4,
        chunk_size: int = 8 * 1024 * 1024,
        trust_cached_size: bool = False,
    ):
        req = self._fileops_pb2.ReadStreamRequest(
            file_path=cloud_file_path,
            snapshot_time=0 if snapshot_time is None else snapshot_time,
            start_position=start_position,
            parallel_chunks=parallel_chunks,
            chunk_size=chunk_size,
            trust_cached_size=trust_cached_size,
        )
        for chunk in self._fileops_stub.ReadStream(req, metadata=self._metadata):
            yield bytes(chunk.data)
```

The gRPC channel caps messages at 64 MB:

```250:258:altastata/grpc_client.py
    @staticmethod
    def _create_channel(endpoint: GrpcEndpoint) -> grpc.Channel:
        options = [
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ]
        if endpoint.secure:
            creds = grpc.ssl_channel_credentials()
            return grpc.secure_channel(endpoint.target, creds, options=options)
        return grpc.insecure_channel(endpoint.target, options=options)
```

### 2.2 Problem

- Unary `GetBuffer` returns the whole file in one protobuf message. With the
  64 MB receive cap, any file near/over that size **fails**.
- `bytes(resp.data)` makes an **extra full copy**: a protobuf `bytes` field
  already returns an immutable `bytes` object.

### 2.3 Required change

Add a module-level threshold constant near the top of `grpc_client.py` (after
the imports, before `_default_client_hint`):

```python
# Files at/above this size are read via the server-streaming ReadStream RPC
# instead of a single unary GetBuffer, so we never hit the 64 MB gRPC message
# cap and never hold two full copies of a large payload at once. Kept well
# under grpc.max_receive_message_length (64 MB) so the unary path's single
# message (+ protobuf framing) always fits.
GRPC_UNARY_MAX_SIZE = 32 * 1024 * 1024  # 32 MB
```

Replace the body of `get_buffer` with the tiered logic below. **Keep the
signature identical.**

```python
    def get_buffer(
        self,
        file_path: str,
        size: int,
        snapshot_time: int = 0,
        start_position: int = 0,
        parallel_chunks: int = 4,
        trust_cached_size: bool = False,
    ) -> bytes:
        # Small, known-size reads: one unary call. Return resp.data directly —
        # it is already an immutable `bytes`, so wrapping it in bytes() would
        # only add a redundant full copy.
        if 0 < size < GRPC_UNARY_MAX_SIZE:
            req = self._fileops_pb2.GetBufferRequest(
                file_path=file_path,
                snapshot_time=snapshot_time,
                start_position=start_position,
                parallel_chunks=parallel_chunks,
                size=size,
                trust_cached_size=trust_cached_size,
            )
            resp = self._fileops_stub.GetBuffer(req, metadata=self._metadata)
            return resp.data

        # Large or unknown-size reads: stream and assemble. When the size is
        # known we pre-allocate one bytearray and copy each chunk in once
        # (single copy per byte); when it is unknown we fall back to join.
        req = self._fileops_pb2.ReadStreamRequest(
            file_path=file_path,
            snapshot_time=snapshot_time,
            start_position=start_position,
            parallel_chunks=parallel_chunks,
            chunk_size=8 * 1024 * 1024,
            trust_cached_size=trust_cached_size,
        )
        stream = self._fileops_stub.ReadStream(req, metadata=self._metadata)

        if size and size > 0:
            buf = bytearray(size)
            view = memoryview(buf)
            offset = 0
            for chunk in stream:
                data = chunk.data
                n = len(data)
                end = offset + n
                if end > size:
                    # Server returned more than the declared size: stop being
                    # clever and fall back to a safe concatenation.
                    parts = [bytes(view[:offset]), bytes(data)]
                    parts.extend(bytes(c.data) for c in stream)
                    return b"".join(parts)
                view[offset:end] = data
                offset = end
            if offset != size:
                return bytes(view[:offset])
            return bytes(buf)

        parts = [bytes(chunk.data) for chunk in stream]
        return b"".join(parts)
```

Notes for the implementer:
- `snapshot_time` here is an `int` (default `0`); pass it straight through to
  `ReadStreamRequest` (it already treats `0` as "latest").
- Do **not** change `read_stream` — it stays as a public streaming helper.
- Do **not** raise the channel message-size limits; streaming removes the need.

---

## 3. Task 3 — Optional Unix-domain-socket endpoint

**Goal:** allow the co-located gateway to be reached over a Unix domain socket
(`unix:<path>`) for lower latency/higher throughput, while **TCP loopback stays
the default** so nothing breaks. Server-side socket binding lives in `mycloud`
and is a **follow-up** — this task only makes the Python client capable.

### 3.1 Current code

```35:43:altastata/grpc_client.py
@dataclass
class GrpcEndpoint:
    host: str = "127.0.0.1"
    port: int = 9877
    secure: bool = False

    @property
    def target(self) -> str:
        return f"{self.host}:{self.port}"
```

### 3.2 Required change

Add an optional `socket_path`. When set, `target` becomes a gRPC `unix:` target
and `secure` is ignored (UDS is local-only).

```python
@dataclass
class GrpcEndpoint:
    host: str = "127.0.0.1"
    port: int = 9877
    secure: bool = False
    # When set, connect over a Unix domain socket instead of TCP. Local-only,
    # so `secure` is ignored. Example: socket_path="/tmp/altastata.sock".
    socket_path: Optional[str] = None

    @property
    def target(self) -> str:
        if self.socket_path:
            return f"unix:{self.socket_path}"
        return f"{self.host}:{self.port}"
```

In `_create_channel`, force an insecure channel for the UDS case (TLS over a
local socket is pointless and `ssl_channel_credentials()` expects a TCP authority):

```python
    @staticmethod
    def _create_channel(endpoint: GrpcEndpoint) -> grpc.Channel:
        options = [
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ]
        if endpoint.secure and not endpoint.socket_path:
            creds = grpc.ssl_channel_credentials()
            return grpc.secure_channel(endpoint.target, creds, options=options)
        return grpc.insecure_channel(endpoint.target, options=options)
```

Also confirm `Optional` is imported (it already is:
`from typing import Dict, List, Optional, Sequence, Tuple`).

**Do not** change the auto-start logic (`_is_port_open`, `_start_local_grpc_service`)
in this task; on a UDS-only setup the port check would need rework, which is a
follow-up tied to the server change. Leave a one-line comment noting that the
auto-start/port-probe path still assumes TCP.

---

## 4. Task 4 — Tests

Add tests to `tests/test_altastata_functions_transport.py` (or a new
`tests/test_grpc_client_get_buffer.py`). Use `unittest` + `unittest.mock`,
matching the existing style in that folder. Mock the fileops stub so no real
gateway is needed.

Cover at least:

1. **Unary path (small file):** `size < GRPC_UNARY_MAX_SIZE` calls `GetBuffer`
   once, does **not** call `ReadStream`, and returns the exact bytes
   (verify identity/no corruption).
2. **Streaming path (large file):** `size >= GRPC_UNARY_MAX_SIZE` calls
   `ReadStream` (not `GetBuffer`), and the assembled result equals the
   concatenation of the chunk payloads. Use 2–3 chunks of fake bytes.
3. **Unknown size (`size == 0`):** uses the streaming/join path and returns the
   concatenation.
4. **Endpoint target:** `GrpcEndpoint(socket_path="/tmp/x.sock").target ==
   "unix:/tmp/x.sock"`; default `GrpcEndpoint().target == "127.0.0.1:9877"`.

Sketch for the streaming test (adapt as needed):

```python
from unittest.mock import MagicMock
from altastata.grpc_client import AltaStataGrpcClient, GRPC_UNARY_MAX_SIZE

def make_client_with_mocked_stub():
    client = AltaStataGrpcClient.__new__(AltaStataGrpcClient)  # bypass __init__
    client._metadata = [("authorization", "Bearer test")]
    client._fileops_stub = MagicMock()
    import altastata.grpc.v1.fileops_pb2 as fileops_pb2
    client._fileops_pb2 = fileops_pb2
    return client

def test_streaming_assembles_chunks():
    client = make_client_with_mocked_stub()
    chunk_a = MagicMock(data=b"a" * 1000)
    chunk_b = MagicMock(data=b"b" * 500)
    client._fileops_stub.ReadStream.return_value = iter([chunk_a, chunk_b])
    out = client.get_buffer("f", size=GRPC_UNARY_MAX_SIZE + 1500)
    assert out == b"a" * 1000 + b"b" * 500
    client._fileops_stub.GetBuffer.assert_not_called()
```

> If `AltaStataGrpcClient.__new__` bypass is awkward, instead patch
> `AltaStataGrpcClient` construction as the existing transport test does, or
> extract the stub calls — but keep it a pure unit test (no real channel).

Run the suite:

```bash
cd /Users/sergevilvovsky/eclipse-workspace/mcloud/altastata-python-package
python -m pytest tests/ -q
```

---

## 5. Acceptance criteria

- [ ] `get_buffer` streams when `size >= GRPC_UNARY_MAX_SIZE` or `size == 0`,
      and uses one unary call otherwise.
- [ ] Files larger than 64 MB can be read via gRPC without a message-size error.
- [ ] No `bytes(resp.data)` copy remains on the unary path (return `resp.data`).
- [ ] `GrpcEndpoint` supports `socket_path` → `unix:<path>` target; TCP default
      unchanged; TLS skipped for UDS.
- [ ] New unit tests pass; full `pytest tests/` stays green.
- [ ] No public signature in `altastata_functions.py` changed.
- [ ] No new dependency added; no Arrow; no Java/`mycloud` edits; no git commit.

---

## 6. Out of scope (do NOT do here)

- Flipping the default `transport` to `grpc` (later phase).
- Removing deprecated legacy bridge notes (later phase).
- Server-side Unix-socket binding and UDS-aware auto-start (mycloud follow-up).
- Apache Arrow / Arrow Flight (deferred; see `grpc-migration-design.md` §9).
- Compression, shared memory, `memfd` (only if profiling later justifies them).
