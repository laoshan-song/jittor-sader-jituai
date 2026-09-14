#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
DATA="${1:?usage: run_inference.sh data_B.zip output-dir [gpu]}"
OUT="${2:?usage: run_inference.sh data_B.zip output-dir [gpu]}"
GPU="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export JT_USE_CUDA=1
export use_cutt=0 use_cutlass=0 use_nccl=0 use_mkl=0
export JITTOR_HOME="${JITTOR_HOME:-$(dirname "$OUT")/.jittor-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$(dirname "$OUT")/.pycache}"
export TMPDIR="${TMPDIR:-$(dirname "$OUT")/.tmp}"
export ML_CACHE_ROOT="${ML_CACHE_ROOT:-$(dirname "$OUT")/.runtime}"
mkdir -p "$JITTOR_HOME" "$PYTHONPYCACHEPREFIX" "$TMPDIR"
LOCKED_BASE="$ML_CACHE_ROOT/locked_assets/frozen_base.ckpt"
"$PYTHON" "$ROOT/code/restore_locked_assets.py" --output "$LOCKED_BASE"
source "$ROOT/code/prepare_cuda_runtime.sh" "$PYTHON"
"$PYTHON" "$ROOT/code/check_environment.py"
exec "$PYTHON" "$ROOT/code/build_submission.py" \
  --data "$DATA" --base "$LOCKED_BASE" \
  --checkpoint "$ROOT/code/assets/locked/d4_implicit_mf32.npz" --output-dir "$OUT"
