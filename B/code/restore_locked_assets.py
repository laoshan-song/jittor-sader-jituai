#!/usr/bin/env python3
"""Validate tracked locked assets and restore the split frozen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "locked"
MODEL_NAME = "d4_implicit_mf32.npz"
MODEL_BYTES = 58_167_035
MODEL_SHA256 = "98dc703a0851229f38b43f588b709c1b1aeff98ab60570a1ca61d8e617eb31f4"
BASE_PARTS = tuple(f"frozen_base.ckpt.part{suffix}" for suffix in ("aa", "ab", "ac", "ad"))
BASE_BYTES = 257_814_859
BASE_SHA256 = "e46182a6114b0089b9e05d03672b93c28758624ef02b7d97357b1994cddf3d18"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model(assets: Path) -> Path:
    model = assets / MODEL_NAME
    if not model.is_file() or model.stat().st_size != MODEL_BYTES:
        raise ValueError(f"locked MF32 model is missing or has the wrong size: {model}")
    if sha256(model) != MODEL_SHA256:
        raise ValueError("locked MF32 model SHA-256 differs")
    return model


def restore_base(assets: Path, output: Path) -> bool:
    if output.is_file() and output.stat().st_size == BASE_BYTES and sha256(output) == BASE_SHA256:
        return False

    parts = [assets / name for name in BASE_PARTS]
    missing = [str(path) for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"locked frozen-base parts are missing: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    digest = hashlib.sha256()
    written = 0
    try:
        with temporary.open("xb") as target:
            for part in parts:
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(8 << 20), b""):
                        target.write(block)
                        digest.update(block)
                        written += len(block)
            target.flush()
            os.fsync(target.fileno())
        if written != BASE_BYTES or digest.hexdigest() != BASE_SHA256:
            raise ValueError("restored frozen-base size or SHA-256 differs")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ASSET_DIR)
    parser.add_argument("--model-output", type=Path, help="also copy the locked MF32 checkpoint here")
    args = parser.parse_args()

    assets = args.assets.resolve()
    model = validate_model(assets)
    restored = restore_base(assets, args.output.resolve())
    model_restored = False
    if args.model_output is not None:
        destination = args.model_output.resolve()
        if not (destination.is_file() and destination.stat().st_size == MODEL_BYTES and sha256(destination) == MODEL_SHA256):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                with model.open("rb") as source, temporary.open("xb") as target:
                    for block in iter(lambda: source.read(8 << 20), b""):
                        target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
                if sha256(temporary) != MODEL_SHA256:
                    raise ValueError("restored MF32 checkpoint SHA-256 differs")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            model_restored = True
        model = destination
    print(
        json.dumps(
            {
                "decision": "PASS",
                "frozen_base": str(args.output.resolve()),
                "frozen_base_sha256": BASE_SHA256,
                "model": str(model),
                "model_sha256": MODEL_SHA256,
                "restored": restored,
                "model_restored": model_restored,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
