#!/usr/bin/env python3
"""Orchestrate MF32 training and official-data base generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--negative-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite model directory: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "code" / "train_model.py"),
            "--data",
            str(args.data),
            "--output-dir",
            str(args.output_dir),
            "--seed",
            str(args.seed),
            "--embedding-dim",
            str(args.embedding_dim),
            "--negative-count",
            str(args.negative_count),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("build_base.py")),
            "--data",
            str(args.data),
            "--output",
            str(args.output_dir / "base_result.zip"),
        ],
        check=True,
    )
    checkpoint = args.output_dir / "d4_implicit_mf32.npz"
    base = args.output_dir / "base_result.zip"
    receipt = {
        "kind": "track1_b_raw_training_v1",
        "decision": "PASS_SUPPLEMENTARY_TRAINING",
        "data": str(args.data),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "generated_base": str(base),
        "generated_base_sha256": sha256(base),
        "inference_command": "bash run_fresh_inference.sh DATA MODEL_DIR OUTPUT_DIR GPU",
        "elapsed_seconds": time.time() - started,
    }
    path = args.output_dir / "RAW_TRAINING_RECEIPT.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
