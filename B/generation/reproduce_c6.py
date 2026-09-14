#!/usr/bin/env python3
"""Rebuild c5 from official data, validate the c6 gate, and generate c6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
ONLINE_SHA256 = "627c50a7d0d941ed25af0f5928d2f72a100b087dac8e34b09472895f77fb4a94"
ONLINE_SCORE = 1.3142106240017903
ROOT = Path(__file__).resolve().parent
C2_ROOT = ROOT / "c2_source"
C5_REPRO = ROOT / "reproduce_c5.py"
C6_CODE = ROOT / "code" / "c6"
C6_GATE = C6_CODE / "d3_ring_tie_group_gate.py"
C6_BUILD = C6_CODE / "build_tie_group_submission.py"
VERIFY = C2_ROOT / "code" / "verify_v26_submission.py"
SEEDS = (20260810, 20260824, 20260907)


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
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--jittor-home", type=Path)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    data = args.data.resolve()
    work = args.work_dir.resolve()
    if sha256(data) != DATA_SHA256:
        raise ValueError("official data_B.zip hash differs")
    if work.exists():
        raise FileExistsError(f"refusing work-directory reuse: {work}")
    work.mkdir(parents=True)
    c5_work = work / "c5"
    c6_work = work / "c6"
    logs = work / "logs"
    env = os.environ.copy()
    env.update({
        "use_cutt": "0",
        "use_cutlass": "0",
        "use_nccl": "0",
        "use_mkl": "0",
    })
    env["PYTHONPATH"] = os.pathsep.join(
        str(path)
        for path in (
            C2_ROOT / "code",
            C2_ROOT / "code" / "b_rank_a_port",
            C2_ROOT / "code" / "c2_d3",
            ROOT / "code" / "c2_d3",
            C6_CODE,
        )
    )
    if args.jittor_home is not None:
        env["JITTOR_HOME"] = str(args.jittor_home.resolve())
    if args.cuda_home is not None:
        cuda = args.cuda_home.resolve()
        env["CUDA_HOME"] = str(cuda)
        env["PATH"] = os.pathsep.join((str(cuda / "bin"), env.get("PATH", "")))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(cuda / "lib64"), env.get("LD_LIBRARY_PATH", ""))
        )
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    c5_args = [
        sys.executable,
        str(C5_REPRO),
        "--data",
        str(data),
        "--work-dir",
        str(c5_work),
    ]
    for name, value in (
        ("--gpu", args.gpu),
        ("--jittor-home", args.jittor_home),
        ("--cuda-home", args.cuda_home),
    ):
        if value is not None:
            c5_args.extend((name, str(value)))
    if args.quick:
        c5_args.append("--quick")
    run(c5_args, ROOT, env, logs / "reproduce_c5.log")

    ensemble = c5_work / "prerequisite" / "c2" / "reports" / "dataset3_ensemble.json"
    gate_reports = []
    if not args.quick:
        for seed in SEEDS:
            report = c6_work / f"d3_tie_group_gate_seed{seed}.json"
            run(
                [
                    sys.executable,
                    str(C6_GATE),
                    "--data",
                    str(data),
                    "--code",
                    str(C2_ROOT / "code" / "c2_d3"),
                    "--ensemble-report",
                    str(ensemble),
                    "--groups",
                    "30000",
                    "--batch",
                    "512",
                    "--seed",
                    str(seed),
                    "--mode",
                    "group",
                    "--selection-negative-budget",
                    "0.002",
                    "--output",
                    str(report),
                ],
                C6_CODE,
                env,
                logs / f"gate_c6_seed{seed}.log",
            )
            if json.loads(report.read_text(encoding="utf-8")).get("decision") != "PASS":
                raise ValueError(f"c6 gate failed for seed {seed}")
            gate_reports.append(str(report))

    c5_zip = c5_work / "b_rank_d34_c5_d3_session_ring.zip"
    c5_manifest = c5_zip.with_suffix(".manifest.json")
    output = work / "b_rank_d34_c6_d3_c5_tie_group.zip"
    run(
        [
            sys.executable,
            str(C6_BUILD),
            "--data",
            str(data),
            "--code",
            str(C2_ROOT / "code" / "c2_d3"),
            "--base-c5",
            str(c5_zip),
            "--base-manifest",
            str(c5_manifest),
            "--output",
            str(output),
        ],
        C6_CODE,
        env,
        logs / "build_c6.log",
    )
    run(
        [sys.executable, str(VERIFY), "--data", str(data), "--submission", str(output)],
        C2_ROOT,
        env,
        logs / "verify_c6.log",
    )
    receipt = {
        "kind": "b_rank_d34_c6_end_to_end_reproduction_receipt_v1",
        "decision": "SMOKE_ONLY" if args.quick else "PASS",
        "data_sha256": DATA_SHA256,
        "submission_sha256": sha256(output),
        "historical_online_submission_sha256": ONLINE_SHA256,
        "historical_online_score": ONLINE_SCORE,
        "exact_historical_sha": sha256(output) == ONLINE_SHA256,
        "c5_submission_sha256": sha256(c5_zip),
        "fresh_validation_reports": gate_reports,
        "quick": bool(args.quick),
    }
    (work / "REPRODUCTION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
