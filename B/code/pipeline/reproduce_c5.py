#!/usr/bin/env python3
"""Train from official data, validate c5, build the submission, and verify it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
ONLINE_SHA256 = "3db0d80defb636d387eb73b8666f38e42cd4bdad3d66b2c71a384045b578dfed"
ROOT = Path(__file__).resolve().parent
C2_ROOT = ROOT / "c2_source"
C3_REPRO = ROOT / "reproduce_c3.py"
C5_CODE = ROOT / "code" / "c5"
C5_GATE = C5_CODE / "d3_ring_safe_gate.py"
C5_BUILD = C5_CODE / "build_ring_submission.py"
VERIFY = C2_ROOT / "code" / "verify_v26_submission.py"
EXPECTED_C3_POLICY = {
    "gate": "pair_new",
    "weights": {
        "cross_past_1s": -0.10,
        "session_future_1s": 0.225,
        "session_future_300s": 0.30,
        "session_past_300s": 0.305,
    },
}


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


def same_policy(actual: dict) -> bool:
    if actual.get("gate") != EXPECTED_C3_POLICY["gate"]:
        return False
    weights = actual.get("weights", {})
    expected = EXPECTED_C3_POLICY["weights"]
    return set(weights) == set(expected) and all(
        abs(float(weights[name]) - value) < 1e-8 for name, value in expected.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--jittor-home", type=Path)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--quick", action="store_true", help="small smoke; not score-authorized")
    args = parser.parse_args()
    data = args.data.resolve()
    work = args.work_dir.resolve()
    if sha256(data) != DATA_SHA256:
        raise ValueError("official data_B.zip hash differs")
    if work.exists():
        raise FileExistsError(f"refusing work-directory reuse: {work}")
    work.mkdir(parents=True)
    prerequisite = work / "prerequisite"
    c5_work = work / "c5"
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
        )
    )
    gate_cache = (
        args.jittor_home.resolve()
        if args.jittor_home is not None
        else prerequisite / "c2" / "runtime" / "d3" / "jittor"
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
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    c3_args = [
        sys.executable,
        str(C3_REPRO),
        "--data",
        str(data),
        "--work-dir",
        str(prerequisite),
    ]
    if args.gpu is not None:
        c3_args += ["--gpu", str(args.gpu)]
    if args.jittor_home is not None:
        c3_args += ["--jittor-home", str(args.jittor_home.resolve())]
    if args.cuda_home is not None:
        c3_args += ["--cuda-home", str(args.cuda_home.resolve())]
    if args.quick:
        c3_args += ["--quick"]
    run(c3_args, ROOT, env, logs / "reproduce_c3.log")

    c3_zip = prerequisite / "c3" / "b_rank_d34_c3_multiscale.zip"
    c3_manifest = c3_zip.with_suffix(".manifest.json")
    ensemble = prerequisite / "c2" / "reports" / "dataset3_ensemble.json"
    gate = c5_work / "d3_session_ring_gate.json"
    gate_args = [
        sys.executable,
        str(C5_GATE),
        "--data",
        str(data),
        "--code",
        str(C2_ROOT / "code" / "c2_d3"),
        "--ensemble-report",
        str(ensemble),
        "--groups",
        "5000" if args.quick else "30000",
        "--batch",
        "256" if args.quick else "512",
        "--seed",
        "20260810",
        "--output",
        str(gate),
    ]
    run(gate_args, C5_CODE, env, logs / "gate_c5.log")

    output = work / "b_rank_d34_c5_d3_session_ring.zip"
    build_args = [
        sys.executable,
        str(C5_BUILD),
        "--data",
        str(data),
        "--code",
        str(C2_ROOT / "code" / "c2_d3"),
        "--base",
        str(c3_zip),
        "--base-manifest",
        str(c3_manifest),
        "--gate-report",
        str(gate),
        "--output",
        str(output),
    ]
    run(build_args, C5_CODE, env, logs / "build_c5.log")
    run(
        [sys.executable, str(VERIFY), "--data", str(data), "--submission", str(output)],
        C2_ROOT,
        env,
        logs / "verify_c5.log",
    )

    receipt = {
        "kind": "b_rank_d34_c5_end_to_end_reproduction_receipt_v1",
        "decision": "SMOKE_ONLY" if args.quick else "PASS",
        "data_sha256": DATA_SHA256,
        "submission_sha256": sha256(output),
        "historical_online_submission_sha256": ONLINE_SHA256,
        "exact_historical_sha": sha256(output) == ONLINE_SHA256,
        "c3_submission_sha256": sha256(c3_zip),
        "c5_gate_report": str(gate),
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
