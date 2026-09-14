#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
DATA="${1:?usage: run_train.sh data_B.zip model-output-dir [gpu]}"
OUT="${2:?usage: run_train.sh data_B.zip model-output-dir [gpu]}"
GPU="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU" JT_USE_CUDA=1
export JITTOR_HOME="${JITTOR_HOME:-$(dirname "$OUT")/.jittor-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$(dirname "$OUT")/.pycache}"
export TMPDIR="${TMPDIR:-$(dirname "$OUT")/.tmp}"
export ML_CACHE_ROOT="${ML_CACHE_ROOT:-$(dirname "$OUT")/.runtime}"
mkdir -p "$JITTOR_HOME" "$PYTHONPYCACHEPREFIX" "$TMPDIR"
source "$ROOT/code/prepare_cuda_runtime.sh" "$PYTHON"
"$PYTHON" "$ROOT/code/check_environment.py"
exec "$PYTHON" "$ROOT/code/raw_training/main.py" --data "$DATA" --output-dir "$OUT"
