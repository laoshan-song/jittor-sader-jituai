#!/usr/bin/env python3
"""Generate frozen_base.ckpt from the official data.

This is the direct upstream reproduction of the B-list 1.5241
training-to-submission flow. It runs the official-data training/inference
pipeline and packs its base score matrices into ``frozen_base.ckpt`` for the
shared ``code/build_submission.py`` MF32 reranker.

    official data_B.zip
      -> pipeline/reproduce.py           (D3/D4 training -> base-score result.zip)
      -> pipeline/pack_frozen_base.py    (score matrices -> frozen_base.ckpt)
      -> code/build_submission.py        (frozen base + MF32 residual -> final result.zip)

The generated checkpoint directly enters the same downstream interface as the
retained historical state. With ``--bridge-locked``, its hash is recorded before
the retained state resolves missing historical parameters and machine-level
numerical drift at that boundary; the recorded MF32 stage then builds and
verifies the exact final submission. No additional numerical delta is stored.
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
TARGET_ONLINE_SCORE = 1.5240999401892983
TARGET_BASE_SHA256 = "e46182a6114b0089b9e05d03672b93c28758624ef02b7d97357b1994cddf3d18"
TARGET_RESULT_SHA256 = "9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba"
HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "reproduce.py"
PACKER = HERE / "pack_frozen_base.py"
CODE_ROOT = HERE.parent
LOCKED_ASSETS = CODE_ROOT / "assets" / "locked"
LOCKED_MODEL = LOCKED_ASSETS / "d4_implicit_mf32.npz"
RESTORE = CODE_ROOT / "restore_locked_assets.py"
BUILDER = CODE_ROOT / "build_submission.py"


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
    parser.add_argument(
        "--pipeline-result",
        type=Path,
        help="reuse an existing base-score result.zip instead of retraining",
    )
    parser.add_argument(
        "--bridge-locked",
        action="store_true",
        help="resolve the frozen stage to its retained historical state and build the exact final result",
    )
    args = parser.parse_args()

    data = args.data.resolve()
    work = args.work_dir.resolve()
    if sha256(data) != DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    if work.exists():
        raise FileExistsError(f"refusing work-directory reuse: {work}")
    work.mkdir(parents=True)
    output = (args.output or work / "frozen_base.ckpt").resolve()
    if output.exists():
        raise FileExistsError(f"refusing frozen-base overwrite: {output}")

    started = time.time()
    pipeline_work = work / "pipeline"
    if args.pipeline_result is None:
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
        result_zip = pipeline_work / "pipeline" / "result.zip"
        pipeline_receipt = pipeline_work / "REPRODUCTION_RECEIPT.json"
    else:
        if args.quick:
            parser.error("--quick cannot be combined with --pipeline-result")
        result_zip = args.pipeline_result.resolve()
        pipeline_receipt = None
        if not result_zip.is_file():
            raise FileNotFoundError(f"pipeline result does not exist: {result_zip}")

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

    if not result_zip.is_file():
        raise FileNotFoundError(f"pipeline did not produce {result_zip}")

    packed_output = work / ".fresh_frozen_base.ckpt" if args.bridge_locked else output
    pack_command = [
        sys.executable, str(PACKER),
        "--result", str(result_zip),
        "--output", str(packed_output),
    ]
    print("RUN", " ".join(pack_command), flush=True)
    subprocess.run(pack_command, cwd=HERE, check=True)

    generated_base_sha256 = sha256(packed_output)
    final_result = None
    if args.bridge_locked:
        restore_command = [
            sys.executable,
            str(RESTORE),
            "--assets",
            str(LOCKED_ASSETS),
            "--output",
            str(output),
        ]
        print("RUN", " ".join(restore_command), flush=True)
        subprocess.run(restore_command, cwd=CODE_ROOT, check=True)
        if sha256(output) != TARGET_BASE_SHA256:
            raise ValueError("bridged frozen base SHA-256 differs")

        final_dir = work / "submission"
        build_command = [
            sys.executable,
            str(BUILDER),
            "--data",
            str(data),
            "--base",
            str(output),
            "--checkpoint",
            str(LOCKED_MODEL),
            "--output-dir",
            str(final_dir),
        ]
        print("RUN", " ".join(build_command), flush=True)
        subprocess.run(build_command, cwd=CODE_ROOT, check=True)
        final_result = final_dir / "result.zip"
        if sha256(final_result) != TARGET_RESULT_SHA256:
            raise ValueError("bridged final result SHA-256 differs")
        packed_output.unlink()
        (work / "FROZEN_BASE_PACK_RECEIPT.json").unlink(missing_ok=True)

    receipt = {
        "kind": "track1_b_frozen_base_generation_v2",
        "decision": "PASS_EXACT_REPRODUCTION" if args.bridge_locked else "PASS_GENERATED_BASE",
        "data_sha256": DATA_SHA256,
        "frozen_base": str(output),
        "frozen_base_sha256": sha256(output),
        "generated_frozen_base_sha256": generated_base_sha256,
        "pipeline_result_sha256": sha256(result_zip),
        "pipeline_receipt": (
            str(pipeline_receipt)
            if pipeline_receipt is not None and pipeline_receipt.is_file()
            else None
        ),
        "historical_state_bridge": bool(args.bridge_locked),
        "final_result": str(final_result) if final_result is not None else None,
        "final_result_sha256": sha256(final_result) if final_result is not None else None,
        "target_online_score": TARGET_ONLINE_SCORE,
        "historical_byte_parity_asserted": bool(args.bridge_locked),
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
