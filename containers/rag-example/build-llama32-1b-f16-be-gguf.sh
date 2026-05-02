#!/usr/bin/env bash
# Self-convert unsloth/Llama-3.2-1B-Instruct-F16.gguf (little-endian) to a
# big-endian copy that runs on s390x AND is eligible for the llama.cpp zDNN
# backend (NNPA acceleration on z16 / z17). zDNN only accelerates F32/F16/BF16
# datatypes (per https://github.com/taronaeo/llama.cpp-s390x/blob/master/docs/backend/zDNN.md);
# the default Q5_K_S BE GGUF baked into our image runs fine on s390x but never
# touches the NNPA. As of writing no F16 BE GGUFs are published on HF, so we
# byte-swap an existing LE F16 GGUF locally.
#
# Run on Mac (or any LE host); the resulting BE F16 file is portable and the
# same file works on z15 (CPU fallback, slower than Q5_K_S), z16/z17 (NNPA).
#
# Prereqs: python3 (>=3.9), ~6 GB free disk (~2.5 GB LE source + ~2.5 GB BE
# output + working space). One-time pip install of huggingface_hub and gguf
# into a venv inside $OUT_DIR (no system-wide changes).
#
# Usage:
#   ./containers/rag-example/build-llama32-1b-f16-be-gguf.sh
#   OUT_DIR=$HOME/llama_models ./containers/rag-example/build-llama32-1b-f16-be-gguf.sh
#
# Optional overrides (env):
#   OUT_DIR    Where to keep both files (default: $HOME/llama_models)
#   SRC_REPO   HF repo holding the LE F16 GGUF (default: unsloth/Llama-3.2-1B-Instruct-GGUF)
#   SRC_FILE   File name in that repo                (default: Llama-3.2-1B-Instruct-F16.gguf)
#   OUT_FILE   Final BE file name (matches the entrypoint NNPA auto-detect logic):
#                                                    (default: llama-3.2-1b-instruct-be.f16.gguf)

set -euo pipefail

SRC_REPO="${SRC_REPO:-unsloth/Llama-3.2-1B-Instruct-GGUF}"
SRC_FILE="${SRC_FILE:-Llama-3.2-1B-Instruct-F16.gguf}"
OUT_DIR="${OUT_DIR:-$HOME/llama_models}"
OUT_FILE="${OUT_FILE:-llama-3.2-1b-instruct-be.f16.gguf}"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "==> Output dir: $OUT_DIR"
echo "==> Source:     $SRC_REPO/$SRC_FILE"
echo "==> Target:     $OUT_FILE"

if [ ! -x ".venv/bin/python" ]; then
  echo "==> Creating venv at $OUT_DIR/.venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet --upgrade "huggingface_hub" "gguf"

if [ ! -f "$SRC_FILE" ]; then
  echo "==> Downloading $SRC_FILE (~2.5 GB) ..."
  python - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="$SRC_REPO", filename="$SRC_FILE", local_dir="$OUT_DIR")
PY
else
  echo "==> Reusing cached $SRC_FILE"
fi

echo "==> Copying $SRC_FILE -> $OUT_FILE"
cp -f "$SRC_FILE" "$OUT_FILE"

echo "==> Byte-swapping $OUT_FILE to big-endian (rewrites in place; ~2.5 GB I/O) ..."
echo "YES" | python -m gguf.scripts.gguf_convert_endian "$OUT_FILE" big

echo "==> Verifying $OUT_FILE is big-endian ..."
python -m gguf.scripts.gguf_convert_endian --dry-run "$OUT_FILE" big

echo
echo "Done."
ls -la "$OUT_DIR/$OUT_FILE"

cat <<EOF

To deploy on LinuxONE with auto-detect (z16/z17 → F16+zDNN, z15 → Q5_K_S):

  # One-time: scp the BE F16 to the GGUF cache on the LinuxONE host
  scp "$OUT_DIR/$OUT_FILE" root@<server>:/root/llama_models/$OUT_FILE

  # Then just run the existing pull-and-run script. The container's entrypoint
  # auto-picks F16 when /proc/cpuinfo reports nnpa AND the file is in /models;
  # otherwise it falls back to Q5_K_S.
  ./containers/rag-example/pull-and-run-rag-s390x-from-icr.sh

To force F16 even on z15 (smoke test, slower than Q5_K_S):

  LLAMA_CPP_MODEL_REPO= LLAMA_CPP_MODEL_FILE=$OUT_FILE \\
    ./containers/rag-example/pull-and-run-rag-s390x-from-icr.sh
EOF
