#!/usr/bin/env bash

# Project-local environment for the Jittor competition.
export ML_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The sandboxed workspace cannot write to the real ~/.cache. Jittor also has
# a few unquoted shell commands, so keep its cache in a no-space temp path.
export HOME="/tmp/ml_home"
export JITTOR_HOME="/tmp/ml_jittor_home"
export XDG_CACHE_HOME="/tmp/ml_cache"
export PIP_CACHE_DIR="/tmp/ml_cache/pip"
export use_mpi=0
export nvcc_path=""

mkdir -p "$HOME/.cache/jittor" "$JITTOR_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
if [ -d "$ML_PROJECT_ROOT/.venv" ]; then
  ln -sfn "$ML_PROJECT_ROOT/.venv" /tmp/ml_venv
elif [ -d "$ML_PROJECT_ROOT/../.venv" ]; then
  ln -sfn "$ML_PROJECT_ROOT/../.venv" /tmp/ml_venv
else
  echo "Virtual environment not found. Create it with: python3 -m venv .venv" >&2
  return 1 2>/dev/null || exit 1
fi

export VIRTUAL_ENV="/tmp/ml_venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME
