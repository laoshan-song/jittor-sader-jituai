#!/usr/bin/env bash
# Source before importing Jittor when CUDA lacks a system cuDNN SDK.
set -euo pipefail

PREPARE_PYTHON=${1:-${PYTHON:-python3}}
if [ -n "${nvcc_path:-}" ] && [ -x "$nvcc_path" ]; then
  PREPARE_NVCC=$nvcc_path
else
  PREPARE_NVCC=$(command -v nvcc || true)
fi
if [ -z "${PREPARE_NVCC:-}" ]; then
  for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-12.4/bin/nvcc /usr/local/cuda-12.0/bin/nvcc; do
    if [ -x "$candidate" ]; then
      PREPARE_NVCC=$candidate
      break
    fi
  done
fi
test -n "${PREPARE_NVCC:-}" && test -x "$PREPARE_NVCC" || { echo "CUDA nvcc was not found" >&2; return 1; }

PREPARE_CUDA_HOME=$(CDPATH= cd -- "$(dirname -- "$PREPARE_NVCC")/.." && pwd -P)
if [ -f "$PREPARE_CUDA_HOME/include/cudnn.h" ] && [ -e "$PREPARE_CUDA_HOME/lib64/libcudnn.so" ]; then
  export nvcc_path=$PREPARE_NVCC
  return 0
fi

PREPARE_CUDNN_ROOT=$("$PREPARE_PYTHON" -c '
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("nvidia.cudnn")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("nvidia-cudnn-cu12 is required when the system CUDA SDK lacks cuDNN")
root = Path(next(iter(spec.submodule_search_locations))).resolve()
if not (root / "include" / "cudnn.h").is_file() or not (root / "lib" / "libcudnn.so.8").is_file():
    raise SystemExit(f"invalid nvidia.cudnn installation: {root}")
print(root)
') || return 1
test -n "${ML_CACHE_ROOT:-}" || { echo "ML_CACHE_ROOT must be set" >&2; return 1; }

PREPARE_WRAP="$ML_CACHE_ROOT/cuda_toolchain"
mkdir -p "$PREPARE_WRAP/bin" "$PREPARE_WRAP/include" "$PREPARE_WRAP/lib64"
if [ ! -x "$PREPARE_WRAP/bin/nvcc" ]; then
  cp -a "$PREPARE_CUDA_HOME/bin/." "$PREPARE_WRAP/bin/"
fi
[ -e "$PREPARE_WRAP/nvvm" ] || ln -s "$PREPARE_CUDA_HOME/nvvm" "$PREPARE_WRAP/nvvm"
for entry in "$PREPARE_CUDA_HOME/include/"* "$PREPARE_CUDNN_ROOT/include/"*; do
  [ -e "$entry" ] || continue
  target="$PREPARE_WRAP/include/$(basename "$entry")"
  [ -e "$target" ] || ln -s "$entry" "$target"
done
for entry in "$PREPARE_CUDA_HOME/lib64/"* "$PREPARE_CUDNN_ROOT/lib/"*; do
  [ -e "$entry" ] || continue
  target="$PREPARE_WRAP/lib64/$(basename "$entry")"
  [ -e "$target" ] || ln -s "$entry" "$target"
done
for entry in "$PREPARE_WRAP"/lib64/libcudnn*.so.8; do
  [ -e "$entry" ] || continue
  target=${entry%.8}
  [ -e "$target" ] || ln -s "$(basename "$entry")" "$target"
done
test -x "$PREPARE_WRAP/bin/nvcc" && test -f "$PREPARE_WRAP/include/cudnn.h" && test -e "$PREPARE_WRAP/lib64/libcudnn.so" || { echo "failed to prepare CUDA/cuDNN toolchain" >&2; return 1; }
export nvcc_path="$PREPARE_WRAP/bin/nvcc"
export PATH="$PREPARE_WRAP/bin:$PATH"
export LD_LIBRARY_PATH="$PREPARE_WRAP/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
