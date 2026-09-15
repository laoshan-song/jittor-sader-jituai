#!/usr/bin/env python3
"""End-to-end D3/D4 reproduction from the official B-rank archive only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_env(
    work: Path,
    gpu: int | None,
    jittor_home: Path | None,
    cuda_home: Path | None,
) -> dict[str, str]:
    env = os.environ.copy()
    runtime = work / "runtime"
    env.update(
        {
            "JT_USE_CUDA": "1",
            "use_cutt": "0",
            "use_cutlass": "0",
            "use_nccl": "0",
            "use_mkl": "0",
            "HOME": str(runtime / "home"),
            "XDG_CACHE_HOME": str(runtime / "xdg"),
            "JITTOR_HOME": str(jittor_home.resolve() if jittor_home else runtime / "jittor"),
            "TMPDIR": str(runtime / "tmp"),
            "PYTHONPYCACHEPREFIX": str(runtime / "pycache"),
        }
    )
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for key in (
        "HOME",
        "XDG_CACHE_HOME",
        "JITTOR_HOME",
        "TMPDIR",
        "PYTHONPYCACHEPREFIX",
    ):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    if cuda_home:
        cuda = cuda_home.resolve()
        env["CUDA_HOME"] = str(cuda)
        env["PATH"] = os.pathsep.join((str(cuda / "bin"), env.get("PATH", "")))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(cuda / "lib64"), env.get("LD_LIBRARY_PATH", ""))
        )
    return env


def run(command: list[str], env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def source_manifest() -> dict[str, str]:
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != "SOURCE_MANIFEST.json"
    )
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="official data_B.zip")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu",
        type=int,
        help="optional physical GPU index; omitted preserves CUDA visibility",
    )
    parser.add_argument("--jittor-home", type=Path)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--quick", action="store_true", help="integrity and Jittor linkage test")
    args = parser.parse_args()

    data = args.data.resolve()
    work = args.work_dir.resolve()
    if not data.is_file() or sha256(data) != DATA_SHA256:
        raise ValueError("official data_B.zip is missing or its SHA-256 differs")
    if work.exists():
        raise FileExistsError(f"refusing to reuse work directory: {work}")
    work.mkdir(parents=True)
    env = runtime_env(work, args.gpu, args.jittor_home, args.cuda_home)

    audit = subprocess.run(
        [sys.executable, str(ROOT / "audit_source_package.py"), "--root", str(ROOT)],
        cwd=ROOT, env=env, check=True, text=True, capture_output=True,
    )
    (work / "source_audit.json").write_text(audit.stdout, encoding="utf-8")
    python_files = [str(path) for path in ROOT.rglob("*.py")]
    subprocess.run(
        [sys.executable, "-m", "py_compile", *python_files],
        cwd=ROOT,
        env=env,
        check=True,
    )

    linkage = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, jittor as jt; "
                "jt.flags.use_cuda=1; "
                "x=jt.array([1.0,2.0,3.0]); y=(x*x).sum(); jt.sync_all(); "
                "assert abs(float(y.data)-14.0)<1e-6; "
                "print(json.dumps({'jittor':str(jt.__version__),'has_cuda':bool(jt.has_cuda),"
                "'use_cuda':bool(jt.flags.use_cuda),'tensor_result':float(y.data)}))"
            ),
        ],
        cwd=ROOT, env=env, check=True, text=True, capture_output=True,
    )
    (work / "jittor_linkage.json").write_text(linkage.stdout, encoding="utf-8")

    if args.quick:
        command = [
            sys.executable, str(ROOT / "reproduce_third_1.py"),
            "--data", str(data), "--work-dir", str(work / "pipeline_smoke"),
            "--quick",
        ]
        if args.gpu is not None:
            command += ["--gpus", str(args.gpu)]
        if args.jittor_home:
            command += ["--jittor-home", str(args.jittor_home.resolve())]
        if args.cuda_home:
            command += ["--cuda-home", str(args.cuda_home.resolve())]
        run(command, env, work / "logs" / "quick.log")
        decision = "SMOKE_ONLY"
        output = None
    else:
        command = [
            sys.executable, str(ROOT / "reproduce_third_1.py"),
            "--data", str(data), "--work-dir", str(work / "pipeline"),
            "--mf-embedding-dim", "512", "--mf-negative-count", "64",
            "--mf-epochs", "3",
        ]
        if args.gpu is not None:
            command += ["--gpus", str(args.gpu)]
        if args.jittor_home:
            command += ["--jittor-home", str(args.jittor_home.resolve())]
        if args.cuda_home:
            command += ["--cuda-home", str(args.cuda_home.resolve())]
        run(command, env, work / "logs" / "full.log")
        output_path = work / "pipeline" / "result.zip"
        if not output_path.is_file():
            raise FileNotFoundError("pipeline did not produce result.zip")
        decision = "PASS"
        output = {"path": str(output_path), "sha256": sha256(output_path)}

    receipt = {
        "kind": "d3d4_official_jittor_reproduction_v2",
        "decision": decision,
        "official_data_sha256": DATA_SHA256,
        "supervision": ["dataset3/train.csv", "dataset4/train.csv"],
        "uses_teacher_answers": False,
        "uses_answer_derived_cache": False,
        "bundled_data": False,
        "bundled_model_weights": False,
        "cuda_visible_devices": env.get(
            "CUDA_VISIBLE_DEVICES", "first visible device"
        ),
        "source_manifest": source_manifest(),
        "output": output,
    }
    (work / "REPRODUCTION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
