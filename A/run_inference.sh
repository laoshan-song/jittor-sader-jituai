#!/usr/bin/env bash
set -euo pipefail

DATA=${1:?"usage: bash run_inference.sh /path/to/data_A.zip /path/to/output_dir"}
OUT=${2:?"usage: bash run_inference.sh /path/to/data_A.zip /path/to/output_dir"}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
export JT_USE_CUDA=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export use_cutt=${use_cutt:-0}
export use_mkl=${use_mkl:-0}

if [ -e "$OUT" ]; then
  echo "refusing to overwrite existing output path: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
export ML_CACHE_ROOT=${ML_CACHE_ROOT:-"$(dirname "$OUT")/.contest1_jittor_cache"}
export HOME="$ML_CACHE_ROOT/home"
export JITTOR_HOME=${JITTOR_HOME:-"$ML_CACHE_ROOT/jittor_home"}
mkdir -p "$ML_CACHE_ROOT" "$HOME" "$JITTOR_HOME"
source "$ROOT/code/tools/prepare_cuda_runtime.sh" "$PYTHON"
"$PYTHON" "$ROOT/code/tools/check_exact_environment.py"

"$PYTHON" "$ROOT/code/build_community_residual_submission.py" \
  --data "$DATA" \
  --baseline "$ROOT/base_result.zip" \
  --checkpoint "$ROOT/models/model_bpr32_prod_community_jittor.npz" \
  --d1-config "$ROOT/experiments/d1_source_support_locked.json" \
  --output "$OUT/result.zip" \
  --audit-output "$OUT/build_audit.json"

EXPECTED=d36facee996b5d45806dd6e1d80f8a48883e505f57c8d8d842b9626a50e8e7ce
ACTUAL=$(sha256sum "$OUT/result.zip" | awk '{print $1}')
test "$ACTUAL" = "$EXPECTED"
echo "reproduced $OUT/result.zip"
