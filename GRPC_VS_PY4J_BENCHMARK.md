# gRPC vs Py4J Benchmark

This benchmark compares control-plane latency for equivalent calls:

- Py4J: `get_file_attribute(file_path, None, "size")`
- gRPC: `GetAttribute(file_path, "size")`

## Exact Commands

### 1) Start gRPC service (from `mycloud`)

```bash
cd /Users/sergevilvovsky/eclipse-workspace/mcloud/mycloud
./gradlew :altastata-grpc:run
```

### 2) Initialize gateway user/session with account files

```bash
curl -sS -X PUT "http://127.0.0.1:9880/setUserProperties/bob123" \
  -H "Content-Type: text/plain" \
  --data-binary @"/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123/altastata-myorgrsa444-bob123.user.properties"

curl -sS -X PUT "http://127.0.0.1:9880/setPrivateKey/bob123" \
  -H "Content-Type: text/plain" \
  --data-binary @"/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123/private.key"

curl -sS -X PUT "http://127.0.0.1:9880/setPassword/bob123" \
  -H "Content-Type: text/plain" \
  --data "123"
```

### 3) Generate Python gRPC stubs (from `altastata-python-package`)

```bash
cd /Users/sergevilvovsky/eclipse-workspace/mcloud/altastata-python-package
python3 -m pip install grpcio-tools
python3 scripts/generate_grpc_stubs.py
```

### 4) Run benchmark

```bash
python3 examples/benchmark_grpc_vs_py4j.py \
  --account-dir "/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123" \
  --password "123" \
  --user-name "bob123" \
  --file-path "Applications/AltaStata Investor Deck_prev.pdf" \
  --iterations 120 \
  --warmup 20
```

## Latest Local Run

- Py4J avg: `51.504 ms` (p50 `43.665`, p95 `117.801`)
- gRPC avg: `53.668 ms` (p50 `45.773`, p95 `124.328`)
- Speedup (Py4J/gRPC avg): `0.96x` for this specific attribute call

Notes:
- This is a control-plane benchmark (`GetAttribute`), not data-plane throughput.
- File upload/download should continue to use S3/boto3 paths.

## getBuffer Benchmark (Py4J vs gRPC)

### Exact Command

```bash
cd /Users/sergevilvovsky/eclipse-workspace/mcloud/altastata-python-package
python3 examples/benchmark_getbuffer_grpc_vs_py4j.py \
  --account-dir "/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123" \
  --password "123" \
  --user-name "bob123" \
  --sizes-mb 1 4 12 18 \
  --kinds text binary \
  --warmup 3 \
  --iterations 12
```

### Latest Local Run

- text 1MB: Py4J `116.663 ms`, gRPC `95.730 ms`, speedup `1.22x`
- binary 1MB: Py4J `94.159 ms`, gRPC `88.411 ms`, speedup `1.07x`
- text 4MB: Py4J `219.119 ms`, gRPC `177.481 ms`, speedup `1.23x`
- binary 4MB: Py4J `209.982 ms`, gRPC `183.634 ms`, speedup `1.14x`
- text 12MB: Py4J `1169.160 ms`, gRPC `854.483 ms`, speedup `1.37x`
- binary 12MB: Py4J `1018.798 ms`, gRPC `925.517 ms`, speedup `1.10x`
- text 18MB: Py4J `1504.720 ms`, gRPC `1132.824 ms`, speedup `1.33x`
- binary 18MB: Py4J `1477.364 ms`, gRPC `1419.787 ms`, speedup `1.04x`

Summary from this run:
- Overall average speedup: `1.19x`
- Text payload average speedup: `1.29x`
- Binary payload average speedup: `1.09x`
