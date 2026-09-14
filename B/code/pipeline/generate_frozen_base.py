#!/usr/bin/env python3
"""Generate frozen_base.ckpt from the official data.

This is the upstream half of the B-list flow that the published package left
implicit. It runs the official-data training/inference pipeline and packs its
score matrices into ``models/frozen_base.ckpt`` so the existing
``code/build_submission.py`` reranker can reproduce the recorded top submission
(online score 1.5240999401892983).

    official data_B.zip
      -> generation/reproduce.py            (end-to-end D3/D4 training -> result.zip)
      -> generation/pack_frozen_base.py     (score matrices -> frozen_base.ckpt)
      -> code/build_submission.py           (frozen base + MF32 residual -> result.zip)

The purpose of this directory is to supply the frozen-base generation chain from
the official data alone. The pipeline records the hashes it produced in
``REPRODUCTION_RECEIPT.json`` next to the base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
# Recorded top submission that this reproduction targets.
TARGET_ONLINE_SCORE = 1.5240999401892983
HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "reproduce.py"
PACKER = HERE / "pack_frozen_base.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="official data_B.zip")
    parser.add_argument("--work-dir", type=Path, required=True, help="must not exist")
    parser.add_argument("--output", type=Path, help="frozen base path (default: <work-dir>/frozen_base.ckpt)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--jittor-home", type=Path)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--quick", action="store_true", help="compile-only smoke; no base is produced")
    args = parser.parse_args()

    data = args.data.resolve()
    work = args.work_dir.resolve()
    if sha256(data) != DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    if work.exists():
        raise FileExistsError(f"refusing work-directory reuse: {work}")
    work.mkdir(parents=True)
    output = (args.output or work / "frozen_base.ckpt").resolve()

    started = time.time()
    pipeline_work = work / "pipeline"
    command = [
        sys.executable, str(PIPELINE),
        "--data", str(data),
        "--work-dir", str(pipeline_work),
        "--gpu", str(args.gpu),
    ]
    if args.jittor_home:
        command += ["--jittor-home", str(args.jittor_home.resolve())]
    if args.cuda_home:
        command += ["--cuda-home", str(args.cuda_home.resolve())]
    if args.quick:
        command.append("--quick")
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)

    if args.quick:
        receipt = {
            "kind": "track1_b_frozen_base_generation_smoke_v1",
            "decision": "SMOKE_ONLY",
            "data_sha256": DATA_SHA256,
            "note": "quick mode compiles the pipeline only and does not build a frozen base",
        }
        (work / "REPRODUCTION_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    # reproduce.py writes the trained submission to <its work-dir>/pipeline/result.zip
    result_zip = pipeline_work / "pipeline" / "result.zip"
    if not result_zip.is_file():
        raise FileNotFoundError(f"pipeline did not produce {result_zip}")

    pack_command = [
        sys.executable, str(PACKER),
        "--result", str(result_zip),
        "--output", str(output),
    ]
    print("RUN", " ".join(pack_command), flush=True)
    subprocess.run(pack_command, cwd=HERE, check=True)

    pipeline_receipt = pipeline_work / "REPRODUCTION_RECEIPT.json"
    receipt = {
        "kind": "track1_b_frozen_base_generation_v1",
        "decision": "PASS_GENERATED_BASE",
        "data_sha256": DATA_SHA256,
        "frozen_base": str(output),
        "frozen_base_sha256": sha256(output),
        "pipeline_result_sha256": sha256(result_zip),
        "pipeline_receipt": str(pipeline_receipt) if pipeline_receipt.is_file() else None,
        "target_online_score": TARGET_ONLINE_SCORE,
        "external_data_used": False,
        "uses_test_labels": False,
        "elapsed_seconds": time.time() - started,
    }
    (work / "REPRODUCTION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
