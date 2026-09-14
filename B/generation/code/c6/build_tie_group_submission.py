#!/usr/bin/env python3
"""Build c6 from an online or independently reproduced c5 submission."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
C5_SHA256 = "3db0d80defb636d387eb73b8666f38e42cd4bdad3d66b2c71a384045b578dfed"
D4_SHA256 = "11283eb88c87751917cbdaadac80dd1e55a32cec0ac298c222e78e520a42aea1"
ROWS = 157_670
WIDTH = 100
C5_WEIGHTS = np.asarray([0.2625, 0.28, -0.0525], dtype=np.float32)
TIE_SCALE = np.float32(0.2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + np.float32(1e-6)
    )


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    np.exp(values, out=values)
    values /= values.sum(axis=1, keepdims=True)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--base-c5", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("official data hash differs")
    base_hash = sha256(args.base_c5)
    reproduced_base = False
    expected_d4_hash = D4_SHA256
    if base_hash != C5_SHA256:
        if args.base_manifest is None or not args.base_manifest.is_file():
            raise ValueError("base is not online c5; pass its reproduction manifest")
        base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
        expected_d4_hash = base_manifest.get("dataset4", {}).get("csv_sha256")
        if (
            base_manifest.get("kind") != "b_rank_d34_c5_d3_session_ring_submission_v1"
            or base_manifest.get("submission_sha256") != base_hash
            or base_manifest.get("data_sha256") != DATA_SHA256
            or not expected_d4_hash
        ):
            raise ValueError("reproduced c5 manifest differs")
        reproduced_base = True

    sys.path.insert(0, str(args.code.resolve()))
    sys.path.insert(0, str((args.code / "b_rank_a_port").resolve()))
    import d3_cross_source_c2_audit as d3
    import d3_multiscale_craft_gate as multiscale
    import d3_near_time_audit as near
    import run

    run.DATA = str(args.data.resolve())
    train, test = run.read_scene("dataset3")
    src = test.src.to_numpy(np.int64, copy=False)
    time = test.time.to_numpy(np.int64, copy=False)
    candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    index = near.NearTimeIndex(src, time, candidates)
    short_past, short_future = multiscale.directional_support(
        index, src, time, candidates, 900, True
    )
    long_past, long_future = multiscale.directional_support(
        index, src, time, candidates, 86_400, True
    )
    ring_past = long_past - short_past
    ring_future = long_future - short_future
    ring_sum = ring_past + ring_future
    history = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    seen = d3.pair_seen(history, src, candidates)

    ring_features = np.stack(
        (qnorm(np.log1p(ring_past)), qnorm(np.log1p(ring_future)), qnorm(np.log1p(ring_sum)))
    )
    c5_residual = np.tensordot(C5_WEIGHTS, ring_features, axes=(0, 0)).astype(np.float32)
    eligible = np.where(~seen, ring_sum, -1)
    maximum = eligible.max(axis=1)
    unique = ((eligible == maximum[:, None]).sum(axis=1) == 1) & (maximum > 0)
    c5_active = (~seen) & (eligible == maximum[:, None]) & unique[:, None]

    # Reproduce the submitted c6 exactly: reinforce c5 unique winners once more,
    # then lift tied maximum-support groups without changing order inside a group.
    tied = (~seen) & (eligible == maximum[:, None]) & (maximum[:, None] > 0)
    c6_rows = tied.sum(axis=1) >= 2
    c6_active = tied & (~c5_active.any(axis=1))[:, None] & c6_rows[:, None]
    c6_signal = qnorm(np.log1p(ring_sum))
    residual = np.where(c5_active, c5_residual, 0.0)
    residual += np.where(c6_active, TIE_SCALE * c6_signal, 0.0)

    with zipfile.ZipFile(args.base_c5) as source:
        if tuple(source.namelist()) != ("dataset3.csv", "dataset4.csv"):
            raise ValueError("c5 submission members differ")
        if member_sha256(source, "dataset4.csv") != expected_d4_hash:
            raise ValueError("c5 D4 content differs")
        with source.open("dataset3.csv") as handle:
            base = np.loadtxt(handle, delimiter=",", dtype=np.float32)
        if base.shape != (ROWS, WIDTH):
            raise ValueError("c5 D3 shape differs")
        logits = np.log(np.clip(base, np.float32(1e-12), None))
        probability = softmax(logits + residual)
        if not np.isfinite(probability).all() or np.any(probability < 0):
            raise ValueError("candidate probabilities are invalid")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
            ) as destination:
                with destination.open("dataset3.csv", "w", force_zip64=True) as raw_file:
                    with io.TextIOWrapper(raw_file, encoding="ascii", newline="") as text:
                        np.savetxt(text, probability, fmt="%.8f", delimiter=",")
                with source.open("dataset4.csv") as incoming:
                    with destination.open("dataset4.csv", "w", force_zip64=True) as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=8 << 20)
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)

    changed = np.argmax(base, axis=1) != np.argmax(probability, axis=1)
    manifest = {
        "kind": "b_rank_d34_c6_d3_c5_tie_group_submission_v1",
        "submission_sha256": sha256(args.output),
        "data_sha256": DATA_SHA256,
        "base_c5_sha256": base_hash,
        "base_is_reproduced": reproduced_base,
        "policy": {
            "c5_weights": {
                "session_ring_past": 0.2625,
                "session_ring_future": 0.28,
                "session_ring_sum": -0.0525,
            },
            "tie_group_scale": float(TIE_SCALE),
            "c5_unique_rows": int(c5_active.any(axis=1).sum()),
            "c6_tie_rows": int(c6_active.any(axis=1).sum()),
            "c6_tie_cells": int(c6_active.sum()),
        },
        "dataset3": {
            "rows": ROWS,
            "top1_changed_rate": float(changed.mean()),
            "top1_changed_rows": int(changed.sum()),
            "row_sum_max_error": float(np.max(np.abs(probability.sum(axis=1) - 1.0))),
        },
        "dataset4": {
            "rows": 2_322_538,
            "csv_sha256": expected_d4_hash,
            "identical_to_c5": True,
        },
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
