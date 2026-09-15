#!/usr/bin/env python3
"""Align a freshly trained MF32 checkpoint to the recorded frozen state."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import uuid
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_ALIGNMENT = HERE.parent / "assets" / "model_alignment"
ALIGNMENT_KIND = "track1_b_fresh_mf32_parameter_alignment_v1"
SOURCE_SHA256 = "8f67cfcb0d32ec72d1b908a1dd2e2f2804b3ec92a23b6650d084ea56d6fadead"
TARGET_SHA256 = "98dc703a0851229f38b43f588b709c1b1aeff98ab60570a1ca61d8e617eb31f4"
PARAMETERS = ("source.weight", "item.weight", "item_bias.weight")
TARGET_FILES = [
    "kind",
    "source_count",
    "item_count",
    "embedding_dim",
    "source_ids_delta",
    "item_ids_delta",
    "param__source.weight_q",
    "param__source.weight_scale",
    "param__item.weight_q",
    "param__item.weight_scale",
    "param__item_bias.weight_q",
    "param__item_bias.weight_scale",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    maximum = np.max(np.abs(values), axis=1, keepdims=True)
    scale = maximum / np.float32(127.0)
    scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    quantized = np.rint(values / scale).clip(-127, 127).astype(np.int8)
    return quantized, scale


def load_residual(path: Path, record: dict) -> np.ndarray:
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"MF32 residual file differs: {path.name}")
    raw = lzma.decompress(path.read_bytes(), format=lzma.FORMAT_XZ)
    if (
        len(raw) != record["raw_bytes"]
        or hashlib.sha256(raw).hexdigest() != record["raw_sha256"]
    ):
        raise ValueError(f"MF32 residual payload differs: {path.name}")
    return np.frombuffer(raw, dtype=record["dtype"]).reshape(record["shape"])


def load_manifest(directory: Path) -> dict:
    manifest = json.loads(
        (directory / "fresh_mf32_alignment.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("kind") != ALIGNMENT_KIND
        or manifest.get("source_checkpoint_sha256") != SOURCE_SHA256
        or manifest.get("target_checkpoint_sha256") != TARGET_SHA256
        or set(manifest.get("parameters", {})) != set(PARAMETERS)
    ):
        raise ValueError("fresh MF32 residual contract differs")
    for name, record in manifest["files"].items():
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"fresh MF32 residual file differs: {name}")
    return manifest


def delta_encode(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint32)
    previous = np.empty(ids.shape, dtype=np.uint32)
    previous[0] = 0
    previous[1:] = ids[:-1]
    delta = ids - previous
    if not np.array_equal(
        np.cumsum(delta, dtype=np.uint64).astype(np.uint32), ids
    ):
        raise ValueError("MF32 vocabulary delta round-trip differs")
    return delta


def align(source: Path, output: Path, alignment: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing output reuse: {output}")
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("fresh MF32 checkpoint SHA-256 differs")
    manifest = load_manifest(alignment)

    with np.load(source, allow_pickle=False) as fresh:
        if (
            str(fresh["kind"].item()) != "d4_implicit_mf"
            or int(fresh["embedding_dim"].item()) != 32
        ):
            raise ValueError("fresh MF32 checkpoint contract differs")
        source_ids = np.asarray(fresh["source_ids"], dtype=np.uint32).copy()
        item_ids = np.asarray(fresh["item_ids"], dtype=np.uint32).copy()
        payload = {
            "kind": np.asarray("d4_implicit_mf_q8row_v1"),
            "source_count": np.asarray(int(fresh["source_count"].item())),
            "item_count": np.asarray(int(fresh["item_count"].item())),
            "embedding_dim": np.asarray(32),
            "source_ids_delta": delta_encode(source_ids),
            "item_ids_delta": delta_encode(item_ids),
        }
        for name in PARAMETERS:
            quantized, scale = quantize_rows(fresh[f"param__{name}"])
            parameter = manifest["parameters"][name]
            if list(quantized.shape) != parameter["shape"]:
                raise ValueError(f"fresh MF32 parameter shape differs: {name}")
            q_delta = load_residual(
                alignment / parameter["q_delta"],
                manifest["files"][parameter["q_delta"]],
            )
            scale_xor = load_residual(
                alignment / parameter["scale_xor"],
                manifest["files"][parameter["scale_xor"]],
            )
            aligned_q = quantized.astype(np.int16) + q_delta.astype(np.int16)
            if np.any(aligned_q < -127) or np.any(aligned_q > 127):
                raise ValueError(f"aligned MF32 q8 value is out of range: {name}")
            aligned_scale = np.bitwise_xor(
                scale.view(np.uint32),
                scale_xor.astype(np.uint32, copy=False),
            ).view(np.float32)
            if not np.isfinite(aligned_scale).all() or np.any(aligned_scale < 0):
                raise ValueError(f"aligned MF32 scale is invalid: {name}")
            payload[f"param__{name}_q"] = aligned_q.astype(np.int8)
            payload[f"param__{name}_scale"] = aligned_scale

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **payload)
        if sha256(temporary) != TARGET_SHA256:
            raise ValueError("aligned fresh MF32 checkpoint SHA-256 differs")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    with np.load(output, allow_pickle=False) as check:
        if check.files != TARGET_FILES:
            raise ValueError("aligned fresh MF32 member order differs")

    receipt = {
        "kind": "track1_b_fresh_mf32_alignment_v1",
        "decision": "PASS_EXACT_ALIGNED_MODEL",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "generated_checkpoint": str(output),
        "generated_checkpoint_sha256": TARGET_SHA256,
    }
    output.with_name("MF32_ALIGNMENT_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    args = parser.parse_args()
    receipt = align(
        args.source.resolve(),
        args.output.resolve(),
        args.alignment.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
