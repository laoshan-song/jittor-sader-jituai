#!/usr/bin/env bash
set -euo pipefail

DATA=${1:?"usage: bash run_verify.sh /path/to/data_A.zip /path/to/output_dir"}
OUT=${2:?"usage: bash run_verify.sh /path/to/data_A.zip /path/to/output_dir"}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}

cd "$ROOT"
sha256sum -c MANIFEST.sha256
"$PYTHON" code/tools/audit_exact_package.py --root "$ROOT" --data "$DATA"
bash "$ROOT/run_inference.sh" "$DATA" "$OUT"
