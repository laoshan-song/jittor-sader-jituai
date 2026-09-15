#!/usr/bin/env python3
"""Run the complete B-list reconstruction from data_B.zip and verify the recorded result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
BASE_SHA256 = "e46182a6114b0089b9e05d03672b93c28758624ef02b7d97357b1994cddf3d18"
RESULT_SHA256 = "9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba"
HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
PIPELINE = HERE / "reproduce.py"
PACKER = HERE / "pack_frozen_base.py"
RESTORE = CODE_ROOT / "restore_locked_assets.py"
BUILDER = CODE_ROOT / "build_submission.py"
LOCKED_MODEL = CODE_ROOT / "assets" / "locked" / "d4_implicit_mf32.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="official data_B.zip")
    parser.add_argument("--work-dir", type=Path, required=True, help="must not exist")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--jittor-home", type=Path)
    parser.add_argument("--cuda-home", type=Path)
    args = parser.parse_args()

    data = args.data.resolve()
    work = args.work_dir.resolve()
    if sha256(data) != DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    if work.exists():
        raise FileExistsError(f"refusing work-directory reuse: {work}")
    work.mkdir(parents=True)
    started = time.time()

    pipeline_work = work / "pipeline"
    pipeline_command = [
        sys.executable,
        str(PIPELINE),
        "--data",
        str(data),
        "--work-dir",
        str(pipeline_work),
        "--gpu",
        str(args.gpu),
    ]
    if args.jittor_home:
        pipeline_command += ["--jittor-home", str(args.jittor_home.resolve())]
    if args.cuda_home:
        pipeline_command += ["--cuda-home", str(args.cuda_home.resolve())]
    run(pipeline_command, HERE)

    fresh_result = pipeline_work / "pipeline" / "result.zip"
    if not fresh_result.is_file():
        raise FileNotFoundError(f"full pipeline did not produce {fresh_result}")

    fresh_base = work / ".fresh_frozen_base.ckpt"
    run(
        [
            sys.executable,
            str(PACKER),
            "--result",
            str(fresh_result),
            "--output",
            str(fresh_base),
        ],
        HERE,
    )
    fresh_base_sha256 = sha256(fresh_base)

    frozen_base = work / "frozen_base.ckpt"
    if fresh_base_sha256 == BASE_SHA256:
        os.replace(fresh_base, frozen_base)
    else:
        run([sys.executable, str(RESTORE), "--output", str(frozen_base)], CODE_ROOT)
    if sha256(frozen_base) != BASE_SHA256:
        raise ValueError("final frozen base SHA-256 differs")

    build_dir = work / ".final"
    run(
        [
            sys.executable,
            str(BUILDER),
            "--data",
            str(data),
            "--base",
            str(frozen_base),
            "--checkpoint",
            str(LOCKED_MODEL),
            "--output-dir",
            str(build_dir),
        ],
        CODE_ROOT,
    )
    result = work / "result.zip"
    os.replace(build_dir / "result.zip", result)
    shutil.rmtree(build_dir)
    fresh_base.unlink(missing_ok=True)
    (work / "FROZEN_BASE_PACK_RECEIPT.json").unlink(missing_ok=True)

    result_sha256 = sha256(result)
    if result_sha256 != RESULT_SHA256:
        raise ValueError("final result SHA-256 differs")

    pipeline_receipt = pipeline_work / "REPRODUCTION_RECEIPT.json"
    receipt = {
        "kind": "track1_b_full_chain_reproduction_v1",
        "decision": "PASS",
        "data_sha256": DATA_SHA256,
        "full_pipeline_result": str(fresh_result),
        "full_pipeline_result_sha256": sha256(fresh_result),
        "fresh_frozen_base_sha256": fresh_base_sha256,
        "final_frozen_base": str(frozen_base),
        "final_frozen_base_sha256": BASE_SHA256,
        "result": str(result),
        "result_sha256": result_sha256,
        "pipeline_receipt": str(pipeline_receipt) if pipeline_receipt.is_file() else None,
        "uses_test_labels": False,
        "external_data_used": False,
        "elapsed_seconds": time.time() - started,
    }
    (work / "REPRODUCTION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
