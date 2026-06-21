#!/bin/bash
# Single source of truth for versions used by build / push scripts.
#
#  JUPYTER_VERSION        = Docker image tag for jupyter-datascience-{arm64,amd64,s390x}.
#                           Bump only when the Jupyter image gains a real feature
#                           change (kernel, base image, layout). A new altastata
#                           wheel from PyPI does NOT require bumping this — pip
#                           just installs the newer wheel into the same image.
#  RAG_VERSION            = Docker image tag for rag-open-llm-s390x (and the
#                           :${RAG_VERSION}_zdnn research variant). Bumped on
#                           every meaningful RAG image change (entrypoint logic,
#                           llama.cpp flags, new GGUF default, etc.).
#  ALTASTATA_PYPI_VERSION = altastata Python package version published to PyPI.
#                           Extracted dynamically from setup.py so we never need
#                           to bump it in two places.
#
# Why two image versions: Jupyter and RAG release on different cadences. Jupyter
# has been stable since 2026c (just absorbing altastata wheel bumps); RAG went
# through several iterations (2026d..2026g) for llama.cpp + GGUF + zDNN gating
# work. Separating the tags keeps each image's history truthful and avoids
# end-user confusion ("did Jupyter change between 2026e and 2026g?" — no).

JUPYTER_VERSION="2026h_latest"
RAG_VERSION="2026k_latest"

# Resolve repo root from this script's location so callers can `source version.sh`
# from any cwd. Falls back to the current dir if BASH_SOURCE isn't available.
_VERSION_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ALTASTATA_PYPI_VERSION="$(
  awk -F"['\"]" '/^[[:space:]]*version[[:space:]]*=/ {print $2; exit}' \
    "${_VERSION_SH_DIR}/setup.py"
)"
unset _VERSION_SH_DIR

if [ -z "$ALTASTATA_PYPI_VERSION" ]; then
  echo "version.sh: ERROR: failed to extract version from setup.py" >&2
  return 1 2>/dev/null || exit 1
fi
