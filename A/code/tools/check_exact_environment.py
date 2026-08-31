#!/usr/bin/env python3
"""Fail closed before the locked CUDA/Jittor reconstruction writes an output."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from importlib.metadata import version

import jittor as jt
import numba
import numpy as np
import pandas as pd


EXPECTED = {
    "jittor": "1.3.10.0",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "numba": "0.66.0",
    "nvidia-cudnn-cu12": "8.9.7.29",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(sys.version_info[:2] == (3, 10), f"Python 3.10 is required, got {sys.version.split()[0]}")
    require(shutil.which("nvcc") is not None, "nvcc is unavailable after CUDA runtime preparation")
    versions = {
        "jittor": jt.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": numba.__version__,
        "nvidia-cudnn-cu12": version("nvidia-cudnn-cu12"),
    }
    mismatch = {
        name: {"expected": EXPECTED[name], "actual": versions[name]}
        for name in EXPECTED
        if versions[name] != EXPECTED[name]
    }
    require(not mismatch, "pinned dependency versions differ: " + json.dumps(mismatch, sort_keys=True))
    require(bool(jt.has_cuda), "Jittor CUDA is unavailable")
    jt.flags.use_cuda = 1
    probe = jt.array([1.0, 2.0, 3.0], dtype="float32")
    value = float(np.asarray((probe * probe).sum().data).item())
    require(value == 14.0, f"Jittor CUDA arithmetic probe differs: {value}")
    print(
        json.dumps(
            {
                "kind": "contest1_exact_jittor_cuda_environment_v1",
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "versions": versions,
                "has_cuda": bool(jt.has_cuda),
                "use_cuda": int(jt.flags.use_cuda),
                "arithmetic_probe": value,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
