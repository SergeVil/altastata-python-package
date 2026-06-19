"""Shared JVM heap / thread-stack defaults for embedded AltaStata Java runtimes."""

import os
from typing import List

# Keep in sync with altastata-services/build.gradle and altastata-services Dockerfiles.
# Rule of thumb: container RAM >= Xmx + ~1.5 GiB for OpsExecutors thread stacks.
DEFAULT_JAVA_MEMORY_OPTS = [
    "-Xms1g",
    "-Xmx4g",
    "-XX:ThreadStackSize=256k",
]


def resolve_java_memory_opts() -> List[str]:
    """Heap/stack flags to pass on the ``java`` command line.

    Local ``pip install`` runs usually have no Java tuning env, so embed the
    defaults on the argv. When ``JAVA_TOOL_OPTIONS`` or ``JAVA_OPTS`` already
    set ``-Xmx`` (typical in Docker compose / k8s), return ``[]`` so the
    environment is the single source of truth and compose overrides work.
    """
    combined = " ".join(
        (
            os.environ.get("JAVA_TOOL_OPTIONS", ""),
            os.environ.get("JAVA_OPTS", ""),
        )
    )
    if "-Xmx" in combined:
        return []
    return list(DEFAULT_JAVA_MEMORY_OPTS)
