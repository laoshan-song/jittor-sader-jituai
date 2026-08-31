#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
"$PYTHON" "$ROOT/code/audit_package.py" "$ROOT"
exec "$ROOT/run_inference.sh" "$@"
