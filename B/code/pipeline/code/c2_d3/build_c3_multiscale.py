#!/usr/bin/env python3
"""Build c3 from c2 using a passed multiscale D3 gate."""

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


C2_SHA256 = "7f223dc558bcab9f8c4dfb2e04ad151a6ad65d06e4e1ef0026910d8fa1abed6e"
D4_SHA256 = "11283eb88c87751917cbdaadac80dd1e55a32cec0ac298c222e78e520a42aea1"
DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
ROWS = {"dataset3": 157_670, "dataset4": 2_322_538}
WIDTH = 100
ALLOWED_FEATURES = {
    f"{prefix}_{direction}_{window}s"
    for prefix in ("session", "cross")
    for direction in ("past", "future")
    for window in (1, 5, 30, 300)
}


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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        help="manifest from an independently reproduced c2 control ZIP",
    )
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("official data hash differs")
    base_hash = sha256(args.base)
    reproduced_base = False
    expected_d4_hash = D4_SHA256
    if base_hash != C2_SHA256:
        if args.base_manifest is None or not args.base_manifest.is_file():
            raise ValueError("base is not online c2; pass a reproduced c2 manifest")
        base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
        expected_d4_hash = base_manifest.get("dataset4", {}).get("csv_sha256")
        if (
            base_manifest.get("kind")
            != "b_rank_d34_c2_d3_source_session_submission_v1"
            or base_manifest.get("submission_sha256") != base_hash
            or base_manifest.get("data_sha256") != DATA_SHA256
            or not expected_d4_hash
        ):
            raise ValueError("reproduced c2 manifest differs")
        reproduced_base = True
    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    policy = gate.get("policy", {})
    weights = policy.get("weights", {})
    if (
        gate.get("kind") != "d3_c2_multiscale_craft_graph_strict_gate_v1"
        or gate.get("decision") != "PASS"
        or not all(gate.get("checks", {}).values())
        or policy.get("gate") not in {"all", "low_margin", "pair_new"}
        or not weights
        or not set(weights) <= ALLOWED_FEATURES
        or any(not np.isfinite(float(value)) for value in weights.values())
    ):
        raise ValueError("multiscale gate is not deployable")

    sys.path.insert(0, str(args.code.resolve()))
    import d3_multiscale_craft_gate as multiscale
    import d3_cross_source_c2_audit as d3
    import d3_near_time_audit as near
    import run

    run.DATA = str(args.data.resolve())
    _train, test = run.read_scene("dataset3")
    src = test.src.to_numpy(np.int64, copy=False)
    time = test.time.to_numpy(np.int64, copy=False)
    candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    index = near.NearTimeIndex(src, time, candidates)
    feature = {}
    for name in weights:
        prefix, direction, window = name.split("_")
        past, future = multiscale.directional_support(
            index, src, time, candidates, int(window[:-1]), prefix == "session"
        )
        support = past if direction == "past" else future
        feature[name] = multiscale.qnorm(np.log1p(support).astype(np.float32))

    with zipfile.ZipFile(args.base) as source:
        if tuple(source.namelist()) != ("dataset3.csv", "dataset4.csv"):
            raise ValueError("c2 submission members differ")
        if member_sha256(source, "dataset4.csv") != expected_d4_hash:
            raise ValueError("c2 D4 content differs")
        with source.open("dataset3.csv") as handle:
            base = np.loadtxt(handle, delimiter=",", dtype=np.float32)
        if base.shape != (ROWS["dataset3"], WIDTH):
            raise ValueError("c2 D3 shape differs")
        logits = np.log(np.clip(base, np.float32(1e-12), None))
        if policy["gate"] == "all":
            active = np.ones_like(logits, dtype=bool)
            active_row_rate = 1.0
        elif policy["gate"] == "pair_new":
            history = _train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
            active = ~d3.pair_seen(history, src, candidates)
            active_row_rate = float(np.mean(np.any(active, axis=1)))
        else:
            ordered = np.sort(logits, axis=1)
            margin = ordered[:, -1] - ordered[:, -2]
            active = np.broadcast_to(
                (margin <= np.quantile(margin, 0.50))[:, None], logits.shape
            )
            active_row_rate = float(np.mean(active[:, 0]))
        residual = np.zeros_like(logits)
        for name, weight in weights.items():
            residual += np.float32(weight) * feature[name]
        candidate = logits + np.where(active, residual, 0.0)
        probability = softmax(candidate)

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
                with destination.open("dataset3.csv", "w", force_zip64=True) as raw:
                    with io.TextIOWrapper(raw, encoding="ascii", newline="\n") as text:
                        np.savetxt(text, probability, fmt="%.8f", delimiter=",")
                with source.open("dataset4.csv") as incoming:
                    with destination.open("dataset4.csv", "w", force_zip64=True) as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=8 << 20)
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)

    changed = np.argmax(base, axis=1) != np.argmax(probability, axis=1)
    with zipfile.ZipFile(args.output) as archive:
        d4_hash = member_sha256(archive, "dataset4.csv")
    if d4_hash != expected_d4_hash:
        raise ValueError("built D4 content differs from c2")
    manifest = {
        "kind": "b_rank_d34_c3_d3_multiscale_target_aware_submission_v1",
        "submission_sha256": sha256(args.output),
        "data_sha256": DATA_SHA256,
        "base_c2_sha256": base_hash,
        "base_is_reproduced": reproduced_base,
        "gate_report": str(args.gate_report.resolve()),
        "gate_report_sha256": sha256(args.gate_report),
        "policy": policy,
        "dataset3": {
            "rows": ROWS["dataset3"],
            "active_row_rate": active_row_rate,
            "top1_changed_rate": float(changed.mean()),
            "top1_changed_rows": int(changed.sum()),
        },
        "dataset4": {
            "rows": ROWS["dataset4"],
            "csv_sha256": d4_hash,
            "identical_to_c2": True,
        },
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
