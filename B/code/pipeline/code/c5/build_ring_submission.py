#!/usr/bin/env python3
"""Apply a passed D3 session-ring gate to an online or reproduced c3 ZIP."""

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
C3_SHA256 = "657cc0dbdbae30cd69fe1c7206800c7161c5cdba560f0cb7a57fe44deeb19788"
D4_SHA256 = "11283eb88c87751917cbdaadac80dd1e55a32cec0ac298c222e78e520a42aea1"
ROWS = {"dataset3": 157_670, "dataset4": 2_322_538}
WIDTH = 100
FEATURES = {"session_ring_past", "session_ring_future", "session_ring_sum"}


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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        help="manifest from an independently reproduced c3 ZIP",
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
    if base_hash != C3_SHA256:
        if args.base_manifest is None or not args.base_manifest.is_file():
            raise ValueError("base is not online c3; pass a reproduced c3 manifest")
        base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
        expected_d4_hash = base_manifest.get("dataset4", {}).get("csv_sha256")
        if (
            base_manifest.get("kind")
            != "b_rank_d34_c3_d3_multiscale_target_aware_submission_v1"
            or base_manifest.get("submission_sha256") != base_hash
            or base_manifest.get("data_sha256") != DATA_SHA256
            or not expected_d4_hash
        ):
            raise ValueError("reproduced c3 manifest differs")
        reproduced_base = True

    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    policy = gate.get("policy", {})
    weights = policy.get("weights", {})
    metrics = gate.get("metrics_vs_c3", {})
    if (
        gate.get("kind") != "d3_c3_session_ring_unique_gate_v1"
        or gate.get("decision") not in {"PASS", "NO_GO"}
        or policy.get("gate") != "pair_new_unique_session_ring_max"
        or set(weights) != FEATURES
        or not all(np.isfinite(float(value)) for value in weights.values())
        or any(split not in metrics for split in ("validation", "confirmation"))
    ):
        raise ValueError("ring gate is not submission-authorized")

    sys.path.insert(0, str(args.code.resolve()))
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
    raw = {
        "session_ring_past": long_past - short_past,
        "session_ring_future": long_future - short_future,
    }
    raw["session_ring_sum"] = raw["session_ring_past"] + raw["session_ring_future"]
    seen = d3.pair_seen(
        train[["src", "dst", "time"]].to_numpy(np.int64, copy=False),
        src,
        candidates,
    )
    eligible = np.where(~seen, raw["session_ring_sum"], -1)
    maximum = eligible.max(axis=1)
    unique = ((eligible == maximum[:, None]).sum(axis=1) == 1) & (maximum > 0)
    active = (~seen) & (eligible == maximum[:, None]) & unique[:, None]
    residual = sum(
        np.float32(weight) * qnorm(np.log1p(raw[name]))
        for name, weight in weights.items()
    )

    with zipfile.ZipFile(args.base) as source:
        if tuple(source.namelist()) != ("dataset3.csv", "dataset4.csv"):
            raise ValueError("c3 submission members differ")
        if member_sha256(source, "dataset4.csv") != expected_d4_hash:
            raise ValueError("c3 D4 content differs")
        with source.open("dataset3.csv") as handle:
            base = np.loadtxt(handle, delimiter=",", dtype=np.float32)
        if base.shape != (ROWS["dataset3"], WIDTH):
            raise ValueError("c3 D3 shape differs")
        logits = np.log(np.clip(base, np.float32(1e-12), None))
        probability = softmax(logits + np.where(active, residual, 0.0))

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
                    with io.TextIOWrapper(raw_file, encoding="ascii", newline="\n") as text:
                        np.savetxt(text, probability, fmt="%.8f", delimiter=",")
                with source.open("dataset4.csv") as incoming:
                    with destination.open("dataset4.csv", "w", force_zip64=True) as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=8 << 20)
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)

    changed = np.argmax(base, axis=1) != np.argmax(probability, axis=1)
    manifest = {
        "kind": "b_rank_d34_c5_d3_session_ring_submission_v1",
        "submission_sha256": sha256(args.output),
        "data_sha256": DATA_SHA256,
        "base_c3_sha256": base_hash,
        "base_is_reproduced": reproduced_base,
        "gate_report": str(args.gate_report.resolve()),
        "gate_report_sha256": sha256(args.gate_report),
        "policy": policy,
        "dataset3": {
            "rows": ROWS["dataset3"],
            "active_row_rate": float(np.mean(unique)),
            "active_cell_rate": float(np.mean(active)),
            "top1_changed_rate": float(changed.mean()),
            "top1_changed_rows": int(changed.sum()),
        },
        "dataset4": {
            "rows": ROWS["dataset4"],
            "csv_sha256": expected_d4_hash,
            "identical_to_c3": True,
        },
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
