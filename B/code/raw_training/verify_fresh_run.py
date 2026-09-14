#!/usr/bin/env python3
"""Verify a supplementary official-data training and inference run."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    result = output_dir / "result.zip"
    required = [
        model_dir / "d4_implicit_mf32.npz",
        model_dir / "base_result.zip",
        model_dir / "BASE_RECEIPT.json",
        output_dir / "REPRODUCTION_RECEIPT.json",
        result,
    ]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("fresh training or inference artifact is missing")
    with zipfile.ZipFile(result) as archive:
        if archive.testzip() is not None or archive.namelist() != ["dataset3.csv", "dataset4.csv"]:
            raise ValueError("fresh result ZIP members or CRC differ")
    base = json.loads((model_dir / "BASE_RECEIPT.json").read_text(encoding="utf-8"))
    inference = json.loads((output_dir / "REPRODUCTION_RECEIPT.json").read_text(encoding="utf-8"))
    if base.get("decision") != "PASS_SUPPLEMENTARY_BASE":
        raise ValueError("base receipt has the wrong decision")
    if inference.get("decision") != "PASS_SUPPLEMENTARY_INFERENCE":
        raise ValueError("supplementary inference receipt differs")
    receipt = {
        "kind": "track1_b_supplementary_run_v1",
        "decision": "PASS_SUPPLEMENTARY_RUN",
        "checkpoint_sha256": sha256(model_dir / "d4_implicit_mf32.npz"),
        "base_sha256": sha256(model_dir / "base_result.zip"),
        "result_sha256": sha256(result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
