#!/usr/bin/env python3
"""Reproduce c2 from official data, then apply the passed c3 gate and build ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
ROOT = Path(__file__).resolve().parent
C2_ROOT = ROOT / "c2_source"
C2_REPRO = C2_ROOT / "reproduce_c2.py"
C3_GATE = ROOT / "code" / "c2_d3" / "d3_multiscale_craft_gate.py"
C3_BUILD = ROOT / "code" / "c2_d3" / "build_c3_multiscale.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def run_stage(
    resume: bool,
    expected: tuple[Path, ...],
    cleanup: tuple[Path, ...],
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log: Path,
) -> bool:
    if resume and all(path.exists() for path in expected):
        print("SKIP", " ".join(str(path) for path in expected), flush=True)
        return False
    if resume:
        for path in cleanup:
            remove_path(path)
        log.unlink(missing_ok=True)
    run(command, cwd, env, log)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--jittor-home", type=Path)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    data = args.data.resolve()
    work = args.work_dir.resolve()
    if sha256(data) != DATA_SHA256:
        raise ValueError("official data_B.zip hash differs")
    if work.exists() and not args.resume:
        raise FileExistsError(f"refusing work-directory reuse: {work}")
    work.mkdir(parents=True, exist_ok=args.resume)
    c2_work = work / "c2"
    c3_work = work / "c3"
    logs = work / "logs"
    env = os.environ.copy()
    env.update({
        "use_cutt": "0",
        "use_cutlass": "0",
        "use_nccl": "0",
        "use_mkl": "0",
    })
    env["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (
            C2_ROOT / "code",
            C2_ROOT / "code" / "b_rank_a_port",
            C2_ROOT / "code" / "c2_d3",
        )
    )
    gate_cache = (
        args.jittor_home.resolve()
        if args.jittor_home is not None
        else c2_work / "runtime" / "d3" / "jittor"
    )
    env["JITTOR_HOME"] = str(gate_cache)
    env["ML_CACHE_ROOT"] = str(work / "runtime" / "gate")
    if args.cuda_home is not None:
        cuda_home = args.cuda_home.resolve()
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = os.pathsep.join((str(cuda_home / "bin"), env.get("PATH", "")))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(cuda_home / "lib64"), env.get("LD_LIBRARY_PATH", ""))
        )
    if args.gpu is not None:
        env.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    c2_args = [sys.executable, str(C2_REPRO), "--data", str(data),
               "--work-dir", str(c2_work)]
    if args.gpu is not None:
        c2_args += ["--gpu", str(args.gpu)]
    if args.jittor_home is not None:
        c2_args += ["--jittor-home", str(args.jittor_home.resolve())]
    if args.cuda_home is not None:
        c2_args += ["--cuda-home", str(args.cuda_home.resolve())]
    if args.quick:
        c2_args.append("--quick")
    c2_zip = c2_work / "b_rank_d34_c2_source_session.zip"
    c2_manifest = c2_zip.with_suffix(".manifest.json")
    if args.resume:
        c2_args.append("--resume")
    run_stage(
        args.resume,
        (c2_zip, c2_manifest, c2_work / "REPRODUCTION_RECEIPT.json"),
        (),
        c2_args,
        C2_ROOT,
        env,
        logs / "reproduce_c2.log",
    )

    ensemble = c2_work / "reports" / "dataset3_ensemble.json"
    gate = c3_work / "d3_multiscale_gate.json"
    gate_args = [sys.executable, str(C3_GATE), "--data", str(data),
                 "--code", str(C2_ROOT / "code" / "c2_d3"),
                 "--ensemble-report", str(ensemble), "--groups",
                 "2000" if args.quick else "30000", "--batch",
                 "256" if args.quick else "512", "--output", str(gate)]
    run_stage(
        args.resume, (gate,), (gate,), gate_args,
        C2_ROOT / "code" / "c2_d3", env, logs / "gate_c3.log",
    )
    build_gate = gate
    output = c3_work / "b_rank_d34_c3_multiscale.zip"
    build_args = [sys.executable, str(C3_BUILD), "--data", str(data),
                  "--code", str(C2_ROOT / "code" / "c2_d3"), "--base",
                  str(c2_zip), "--base-manifest", str(c2_manifest),
                  "--gate-report", str(build_gate), "--output", str(output)]
    output_manifest = output.with_suffix(".manifest.json")
    built_output = run_stage(
        args.resume, (output, output_manifest), (output, output_manifest),
        build_args, C2_ROOT / "code" / "c2_d3", env, logs / "build_c3.log",
    )
    verify = C2_ROOT / "code" / "verify_v26_submission.py"
    if built_output:
        (logs / "verify_c3.log").unlink(missing_ok=True)
        run([sys.executable, str(verify), "--data", str(data), "--submission", str(output)],
            C2_ROOT, env, logs / "verify_c3.log")
    receipt = {
        "kind": "b_rank_d34_c3_end_to_end_reproduction_receipt_v1",
        "decision": "SMOKE_ONLY" if args.quick else "PASS",
        "data_sha256": DATA_SHA256,
        "c2_submission_sha256": sha256(c2_zip),
        "c3_submission_sha256": sha256(output),
        "gate_report": str(build_gate),
        "fresh_validation_report": None if args.quick else str(gate),
        "quick": bool(args.quick),
    }
    (work / "REPRODUCTION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
