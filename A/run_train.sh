#!/usr/bin/env bash
set -euo pipefail

DATA=${1:?"usage: bash run_train.sh /path/to/data_A.zip /path/to/model_output [--dataset dataset1|dataset2|all] [--cuda]"}
MODELS=${2:?"usage: bash run_train.sh /path/to/data_A.zip /path/to/model_output [--dataset dataset1|dataset2|all] [--cuda]"}
shift 2
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
export JT_USE_CUDA=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export use_cutt=${use_cutt:-0}
export use_mkl=${use_mkl:-0}
export ML_CACHE_ROOT=${ML_CACHE_ROOT:-"$(dirname "$MODELS")/.track1_jittor_cache"}
export HOME="$ML_CACHE_ROOT/home"
export JITTOR_HOME=${JITTOR_HOME:-"$ML_CACHE_ROOT/jittor_home"}
mkdir -p "$ML_CACHE_ROOT" "$HOME" "$JITTOR_HOME"
source "$ROOT/code/tools/prepare_cuda_runtime.sh" "$PYTHON"
"$PYTHON" "$ROOT/code/raw_training/tools/check_raw_pipeline_contract.py"
"$PYTHON" "$ROOT/code/raw_training/tools/check_environment.py"
exec "$PYTHON" "$ROOT/code/raw_training/main.py" train --data "$DATA" --output-models "$MODELS" "$@"
