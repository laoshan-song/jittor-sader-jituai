#!/usr/bin/env python3
"""Fail closed before a CUDA/Jittor entry point starts model work."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
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
    require(
        sys.version_info[:2] == (3, 10),
        f"Python 3.10 is required, got {sys.version.split()[0]}",
    )
    release = platform.freedesktop_os_release()
    require(
        release.get("ID") == "ubuntu" and release.get("VERSION_ID") == "22.04",
        f"Ubuntu 22.04 is required, got {release}",
    )
    nvcc = os.environ.get("nvcc_path") or shutil.which("nvcc")
    require(
        bool(nvcc and os.path.isfile(nvcc)),
        "nvcc is unavailable after CUDA preparation",
    )
    versions = {
        "jittor": jt.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": numba.__version__,
        "nvidia-cudnn-cu12": version("nvidia-cudnn-cu12"),
    }
    mismatch = {
        name: {"expected": expected, "actual": versions[name]}
        for name, expected in EXPECTED.items()
        if versions[name] != expected
    }
    require(
        not mismatch,
        "pinned dependency versions differ: "
        + json.dumps(mismatch, sort_keys=True),
    )
    require(bool(jt.has_cuda), "Jittor CUDA is unavailable")
    jt.flags.use_cuda = 1
    probe = jt.array([1.0, 2.0, 3.0], dtype="float32")
    value = float(np.asarray((probe * probe).sum().data).item())
    require(value == 14.0, f"Jittor CUDA arithmetic probe differs: {value}")
    nvcc_version = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    require(
        "release 12.4" in nvcc_version,
        f"CUDA 12.4 is required, got {nvcc_version}",
    )
    nvidia_smi = shutil.which("nvidia-smi")
    require(nvidia_smi is not None, "nvidia-smi is unavailable")
    gpu_names = [
        name.strip()
        for name in subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if name.strip()
    ]
    require(
        any("RTX 4090" in name for name in gpu_names),
        f"NVIDIA RTX 4090 is required, got {gpu_names}",
    )
    print(
        json.dumps(
            {
                "kind": "contest1_b_jittor_cuda_environment_v1",
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "os_release": release,
                "gpu_names": gpu_names,
                "versions": versions,
                "nvcc": nvcc_version.splitlines()[-1],
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
