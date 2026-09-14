#!/usr/bin/env bash
# Regenerate the frozen base from official data, then pack it into frozen_base.ckpt.
# Usage: run_generate_base.sh data_B.zip work-dir [gpu] [output-ckpt]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN="$ROOT/code/pipeline"
PYTHON="${PYTHON:-python3}"
DATA="${1:?usage: run_generate_base.sh data_B.zip work-dir [gpu] [output-ckpt]}"
WORK="${2:?usage: run_generate_base.sh data_B.zip work-dir [gpu] [output-ckpt]}"
GPU="${3:-0}"
OUTPUT="${4:-}"
export CUDA_VISIBLE_DEVICES="$GPU" JT_USE_CUDA=1
export use_cutt=0 use_cutlass=0 use_nccl=0 use_mkl=0
export JITTOR_HOME="${JITTOR_HOME:-$(dirname "$WORK")/.jittor-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$(dirname "$WORK")/.pycache}"
export TMPDIR="${TMPDIR:-$(dirname "$WORK")/.tmp}"
export ML_CACHE_ROOT="${ML_CACHE_ROOT:-$(dirname "$WORK")/.runtime}"
mkdir -p "$JITTOR_HOME" "$PYTHONPYCACHEPREFIX" "$TMPDIR"
source "$ROOT/code/prepare_cuda_runtime.sh" "$PYTHON"
"$PYTHON" "$ROOT/code/check_environment.py"
CMD=("$PYTHON" "$GEN/generate_frozen_base.py" --data "$DATA" --work-dir "$WORK" --gpu "$GPU")
if [[ -n "$OUTPUT" ]]; then CMD+=(--output "$OUTPUT"); fi
exec "${CMD[@]}"
