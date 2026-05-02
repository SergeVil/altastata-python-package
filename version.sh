#!/bin/bash
# Single source of truth for versions used by build / push scripts.
#
#  VERSION                = Docker image tag (jupyter-datascience-*, rag-open-llm-*).
#                           Edit this file to bump.
#  ALTASTATA_PYPI_VERSION = altastata Python package version published to PyPI.
#                           Extracted dynamically from setup.py so we never need
#                           to bump it in two places.

VERSION="2026f_latest"

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
