#!/usr/bin/env python3
# Note: proto/altastata/grpc/v1/ files are the AltaStata gRPC contracts and
# must stay in sync with mycloud/altastata-grpc/src/main/proto/altastata/grpc/v1/.
# Sync that directory before regenerating stubs so server and client never drift.
# The protobuf package remains altastata.v1 (wire API version); only the file
# path uses grpc/v1 for clarity in Python and Java build trees.
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    proto_root = repo_root / "proto"
    out_root = repo_root

    proto_files = sorted((proto_root / "altastata" / "grpc" / "v1").glob("*.proto"))
    if not proto_files:
        print("No proto files found.", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_root}",
        f"--python_out={out_root}",
        f"--grpc_python_out={out_root}",
    ] + [str(p) for p in proto_files]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Generated stubs into", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
