#!/usr/bin/env python3
"""
Benchmark control-plane latency for gRPC vs Py4J.

This benchmark compares equivalent "get file attribute" calls:
  - gRPC: GetAttribute(file_path, "size")
  - Py4J: get_file_attribute(file_path, None, "size")
"""

from __future__ import annotations

import argparse
import statistics
import time

from altastata import AltaStataFunctions, AltaStataGrpcClient, GrpcEndpoint


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def run_benchmark(fn, iterations: int, warmup: int):
    for _ in range(warmup):
        fn()

    samples_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)

    values = sorted(samples_ms)
    return {
        "count": len(values),
        "avg_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def print_stats(label, stats):
    print(f"\n{label}")
    print(f"  count  : {stats['count']}")
    print(f"  avg ms : {stats['avg_ms']:.3f}")
    print(f"  p50 ms : {stats['p50_ms']:.3f}")
    print(f"  p95 ms : {stats['p95_ms']:.3f}")
    print(f"  p99 ms : {stats['p99_ms']:.3f}")
    print(f"  min ms : {stats['min_ms']:.3f}")
    print(f"  max ms : {stats['max_ms']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-dir", required=True, help="AltaStata account directory")
    parser.add_argument("--password", required=True, help="Account password")
    parser.add_argument("--user-name", required=False, default=None,
                        help="Override user name (default: inferred from account dir)")
    parser.add_argument("--file-path", required=True, help="Cloud file path to query attribute for")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--grpc-host", default="127.0.0.1")
    parser.add_argument("--grpc-port", type=int, default=9877)
    args = parser.parse_args()

    py4j_client = AltaStataFunctions.from_account_dir(args.account_dir)
    py4j_client.set_password(args.password)

    grpc_client = AltaStataGrpcClient.from_account_dir(
        account_dir_path=args.account_dir,
        password=args.password,
        user_name=args.user_name,
        endpoint=GrpcEndpoint(host=args.grpc_host, port=args.grpc_port, secure=False),
    )

    def py4j_call():
        py4j_client.get_file_attribute(args.file_path, None, "size")

    def grpc_call():
        grpc_client.get_attribute(args.file_path, "size")

    py4j_stats = run_benchmark(py4j_call, args.iterations, args.warmup)
    grpc_stats = run_benchmark(grpc_call, args.iterations, args.warmup)

    print_stats("Py4J", py4j_stats)
    print_stats("gRPC", grpc_stats)

    if grpc_stats["avg_ms"] > 0:
        speedup = py4j_stats["avg_ms"] / grpc_stats["avg_ms"]
        print(f"\nSpeedup (Py4J/gRPC avg): {speedup:.2f}x")

    grpc_client.close()
    py4j_client.shutdown()


if __name__ == "__main__":
    main()
