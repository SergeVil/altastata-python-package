import argparse
import shlex
import subprocess
import sys

from .grpc_client import (
    _build_grpc_subprocess_env,
    _resolve_local_grpc_startup_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start AltaStata gRPC server from Python package runtime."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved startup command and exit.",
    )
    args = parser.parse_args()

    command, working_dir = _resolve_local_grpc_startup_command()
    if args.dry_run:
        print("cwd:", working_dir or ".")
        print("command:", shlex.join(command))
        return 0

    # Reuse the same env builder as the in-process launcher so the bundled
    # AltaStata Console SPA (when present) is served on the same port as gRPC.
    process = subprocess.Popen(command, cwd=working_dir, env=_build_grpc_subprocess_env())
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            return process.wait(timeout=5)
        except Exception:
            return 130


if __name__ == "__main__":
    sys.exit(main())
