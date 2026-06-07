#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import time
from typing import List, Tuple

from altastata import AltaStataFunctions, AltaStataGrpcClient, GrpcEndpoint


def find_cloud_version_path_py4j(py4j_client: AltaStataFunctions, prefix: str) -> str | None:
    it = py4j_client.list_cloud_files_versions(prefix, True, None, None)
    for arr in it:
        for item in arr:
            s = str(item)
            if "✹" in s:
                return s.split("✹")[0]
    return None


def find_cloud_version_path_grpc(grpc_client: AltaStataGrpcClient, prefix: str) -> str | None:
    rows = grpc_client.list_cloud_files_versions(prefix, True, None, None)
    for row in rows:
        for s in row:
            if "✹" in s:
                return s.split("✹")[0]
    return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def bench_py4j(py4j_client: AltaStataFunctions, file_path: str, run_prefix: str) -> Tuple[float, float, bool]:
    content = read_file(file_path)
    size = len(content)
    base = os.path.basename(file_path)
    local_prefix = os.path.dirname(file_path)
    cloud_prefix = f"{run_prefix}/py4j/{base}"

    t0 = time.perf_counter()
    py4j_client.store([file_path], local_prefix, cloud_prefix, True)
    upload_s = time.perf_counter() - t0

    cloud_path = find_cloud_version_path_py4j(py4j_client, cloud_prefix)
    if not cloud_path:
        raise RuntimeError(f"Py4J upload succeeded but no cloud path found for {cloud_prefix}")

    t1 = time.perf_counter()
    downloaded = py4j_client.get_buffer(cloud_path, None, 0, 4, size)
    download_s = time.perf_counter() - t1

    ok = sha256_bytes(content) == sha256_bytes(downloaded)
    py4j_client.delete_files(cloud_prefix, True, None, None)
    return upload_s, download_s, ok


def bench_grpc(grpc_client: AltaStataGrpcClient, file_path: str, run_prefix: str) -> Tuple[float, float, bool]:
    content = read_file(file_path)
    size = len(content)
    base = os.path.basename(file_path)
    cloud_path = f"{run_prefix}/grpc/{base}/{base}"
    cloud_prefix = f"{run_prefix}/grpc/{base}"

    t0 = time.perf_counter()
    grpc_client.create_file(cloud_path, content)
    upload_s = time.perf_counter() - t0

    resolved_cloud_path = find_cloud_version_path_grpc(grpc_client, cloud_prefix)
    if not resolved_cloud_path:
        resolved_cloud_path = cloud_path

    t1 = time.perf_counter()
    downloaded = grpc_client.get_buffer(resolved_cloud_path, size=size, parallel_chunks=4)
    download_s = time.perf_counter() - t1

    ok = sha256_bytes(content) == sha256_bytes(downloaded)
    grpc_client.delete_files(cloud_prefix, including_subdirectories=True)
    return upload_s, download_s, ok


def fmt_rate(size_bytes: int, sec: float) -> float:
    return (size_bytes / (1024 * 1024)) / sec if sec > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-dir", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--user-name", required=False, default=None,
                        help="Override user name (default: inferred from account dir)")
    parser.add_argument("--files", nargs="+", required=True, help="Absolute paths to files")
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

    run_prefix = f"bench_compare/{int(time.time())}"
    print(f"Run prefix: {run_prefix}")
    print("")

    for fp in args.files:
        size = os.path.getsize(fp)
        print(f"File: {os.path.basename(fp)} ({size/1024/1024:.2f} MB)")

        py_up, py_down, py_ok = bench_py4j(py4j_client, fp, run_prefix)
        grpc_up, grpc_down, grpc_ok = bench_grpc(grpc_client, fp, run_prefix)

        print(f"  Py4J  upload: {py_up:.3f}s ({fmt_rate(size, py_up):.2f} MB/s) | "
              f"download: {py_down:.3f}s ({fmt_rate(size, py_down):.2f} MB/s) | verify={py_ok}")
        print(f"  gRPC  upload: {grpc_up:.3f}s ({fmt_rate(size, grpc_up):.2f} MB/s) | "
              f"download: {grpc_down:.3f}s ({fmt_rate(size, grpc_down):.2f} MB/s) | verify={grpc_ok}")
        print(f"  Upload speedup (Py4J/gRPC): {(py_up / grpc_up):.2f}x time ratio")
        print(f"  Download speedup (Py4J/gRPC): {(py_down / grpc_down):.2f}x time ratio")
        print("")

    grpc_client.close()
    py4j_client.shutdown()


if __name__ == "__main__":
    main()
