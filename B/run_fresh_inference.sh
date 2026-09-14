#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
DATA="${1:?usage: run_fresh_inference.sh data_B.zip model-dir output-dir [gpu]}"
MODELS="${2:?usage: run_fresh_inference.sh data_B.zip model-dir output-dir [gpu]}"
OUT="${3:?usage: run_fresh_inference.sh data_B.zip model-dir output-dir [gpu]}"
GPU="${4:-0}"
export CUDA_VISIBLE_DEVICES="$GPU" JT_USE_CUDA=1
export JITTOR_HOME="${JITTOR_HOME:-$(dirname "$OUT")/.jittor-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$(dirname "$OUT")/.pycache}"
export TMPDIR="${TMPDIR:-$(dirname "$OUT")/.tmp}"
export ML_CACHE_ROOT="${ML_CACHE_ROOT:-$(dirname "$OUT")/.runtime}"
mkdir -p "$JITTOR_HOME" "$PYTHONPYCACHEPREFIX" "$TMPDIR"
source "$ROOT/code/prepare_cuda_runtime.sh" "$PYTHON"
"$PYTHON" "$ROOT/code/check_environment.py"
BASE="$MODELS/base_result.zip"
if [[ ! -f "$BASE" ]]; then
  echo "fresh base missing; rebuilding it from official training data" >&2
  "$PYTHON" "$ROOT/code/raw_training/build_base.py" --data "$DATA" --output "$BASE"
fi
"$PYTHON" "$ROOT/code/build_submission.py" \
  --data "$DATA" --base "$BASE" \
  --checkpoint "$MODELS/d4_implicit_mf32.npz" --output-dir "$OUT" --unlocked
exec "$PYTHON" "$ROOT/code/raw_training/verify_fresh_run.py" \
  --model-dir "$MODELS" --output-dir "$OUT" --output "$OUT/FRESH_RUN_RECEIPT.json"
