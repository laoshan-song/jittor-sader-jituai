#!/usr/bin/env bash
# Run the complete data_B.zip training and reconstruction chain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
DATA="${1:?usage: run_reproduce.sh data_B.zip work-dir [gpu]}"
WORK="${2:?usage: run_reproduce.sh data_B.zip work-dir [gpu]}"
GPU="${3:-}"

"$PYTHON" "$ROOT/code/audit_package.py" "$ROOT" --route reproduce
if [[ -n "$GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export JT_USE_CUDA=1
export use_cutt=0 use_cutlass=0 use_nccl=0 use_mkl=0
export JITTOR_HOME="${JITTOR_HOME:-$(dirname "$WORK")/.jittor-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$(dirname "$WORK")/.pycache}"
export TMPDIR="${TMPDIR:-$(dirname "$WORK")/.tmp}"
export ML_CACHE_ROOT="${ML_CACHE_ROOT:-$(dirname "$WORK")/.runtime}"
mkdir -p "$JITTOR_HOME" "$PYTHONPYCACHEPREFIX" "$TMPDIR"

source "$ROOT/code/prepare_cuda_runtime.sh" "$PYTHON"
"$PYTHON" "$ROOT/code/check_environment.py"
exec "$PYTHON" "$ROOT/code/pipeline/reproduce_full.py" \
  --data "$DATA" --work-dir "$WORK"
