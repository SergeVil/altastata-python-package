#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from altastata import AltaStataFunctions, AltaStataGrpcClient, GrpcEndpoint


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def stats(samples_ms: List[float]) -> Dict[str, float]:
    values = sorted(samples_ms)
    return {
        "count": float(len(values)),
        "avg_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def run_benchmark(call: Callable[[], bytes], warmup: int, iterations: int) -> Dict[str, float]:
    for _ in range(warmup):
        call()
    samples: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        call()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return stats(samples)


def make_payload(size_bytes: int, kind: str) -> bytes:
    if kind == "binary":
        return os.urandom(size_bytes)
    base = ("AltaStata getBuffer benchmark line 0123456789 abcdefghijklmnopqrstuvwxyz\n").encode("utf-8")
    repeat = (size_bytes // len(base)) + 1
    return (base * repeat)[:size_bytes]


def fmt_mbps(size_bytes: int, avg_ms: float) -> float:
    if avg_ms <= 0:
        return 0.0
    seconds = avg_ms / 1000.0
    return (size_bytes / (1024 * 1024)) / seconds


def print_stats(label: str, s: Dict[str, float], size_bytes: int) -> None:
    print(f"  {label}")
    print(f"    avg={s['avg_ms']:.3f} ms p50={s['p50_ms']:.3f} ms p95={s['p95_ms']:.3f} ms p99={s['p99_ms']:.3f} ms")
    print(f"    min={s['min_ms']:.3f} ms max={s['max_ms']:.3f} ms throughput={fmt_mbps(size_bytes, s['avg_ms']):.2f} MB/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark getBuffer performance for gRPC vs Py4J")
    parser.add_argument("--account-dir", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--user-name", required=False, default=None,
                        help="Override user name (default: inferred from account dir)")
    parser.add_argument("--grpc-host", default="127.0.0.1")
    parser.add_argument("--grpc-port", type=int, default=9877)
    parser.add_argument("--sizes-mb", nargs="+", type=int, default=[1, 4, 12, 18])
    parser.add_argument("--kinds", nargs="+", default=["text", "binary"], choices=["text", "binary"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    args = parser.parse_args()

    py4j_client = AltaStataFunctions.from_account_dir(args.account_dir)
    py4j_client.set_password(args.password)
    grpc_client = AltaStataGrpcClient.from_account_dir(
        account_dir_path=args.account_dir,
        password=args.password,
        user_name=args.user_name,
        endpoint=GrpcEndpoint(host=args.grpc_host, port=args.grpc_port, secure=False),
    )

    run_id = int(time.time())
    print(f"run_id={run_id}")
    print("")

    try:
        for size_mb in args.sizes_mb:
            size_bytes = size_mb * 1024 * 1024
            for kind in args.kinds:
                payload = make_payload(size_bytes, kind)
                name = f"bench_getbuffer/{run_id}/{kind}_{size_mb}mb.bin"
                py4j_path = f"{name}/py4j"
                grpc_path = f"{name}/grpc"

                py4j_client.create_file(py4j_path, payload)
                grpc_client.create_file(grpc_path, payload)

                def py4j_call() -> bytes:
                    return py4j_client.get_buffer(py4j_path, None, 0, 4, size_bytes)

                def grpc_call() -> bytes:
                    return grpc_client.get_buffer(grpc_path, size=size_bytes, parallel_chunks=4)

                py_stats = run_benchmark(py4j_call, args.warmup, args.iterations)
                grpc_stats = run_benchmark(grpc_call, args.warmup, args.iterations)

                py4j_client.delete_files(f"{name}/py4j", True, None, None)
                grpc_client.delete_files(f"{name}/grpc", including_subdirectories=True)

                speedup = py_stats["avg_ms"] / grpc_stats["avg_ms"] if grpc_stats["avg_ms"] > 0 else 0.0
                print(f"{kind} {size_mb}MB")
                print_stats("Py4J", py_stats, size_bytes)
                print_stats("gRPC", grpc_stats, size_bytes)
                print(f"  speedup (Py4J/gRPC avg time): {speedup:.2f}x")
                print("")
    finally:
        grpc_client.close()
        py4j_client.shutdown()


if __name__ == "__main__":
    main()
