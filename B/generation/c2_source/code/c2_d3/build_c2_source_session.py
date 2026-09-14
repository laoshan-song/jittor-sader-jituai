#!/usr/bin/env python3
"""Build c2 by adding the strictly passed D3 source-session residual to c1."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

import d3_cross_source_c2_audit as d3
import d3_near_time_audit as near
from b_rank_a_port import run


ROWS = {"dataset3": 157_670, "dataset4": 2_322_538}
WIDTH = 100
WINDOW_SECONDS = 300
RESIDUAL_WEIGHT = 0.05


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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if d3.sha256(args.data) != d3.DATA_SHA256:
        raise ValueError("official data hash differs")
    base_hash = sha256(args.base)
    base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    if (
        base_manifest.get("submission_sha256") != base_hash
        or base_manifest.get("data_sha256") != d3.DATA_SHA256
    ):
        raise ValueError("base manifest does not match the supplied base ZIP")
    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    expected_policy = {
        "exclude_exact_time_support": True,
        "feature": "source_session",
        "gate": "all",
        "residual_weight": RESIDUAL_WEIGHT,
        "v26_exact_weight": 0.1,
        "window_seconds": WINDOW_SECONDS,
    }
    if (
        gate.get("kind") != "d3_incremental_candidate_support_strict_gate_v1"
        or gate.get("decision") != "PASS"
        or gate.get("fixed_policy") != expected_policy
        or not all(gate.get("checks", {}).values())
    ):
        raise ValueError("D3 source-session gate is not deployable")

    run.DATA = str(args.data.resolve())
    _train, test = run.read_scene("dataset3")
    src = test.src.to_numpy(np.uint32, copy=False)
    time = test.time.to_numpy(np.uint32, copy=False)
    candidates = test.iloc[:, 2:].to_numpy(np.uint32, copy=False)
    index = near.NearTimeIndex(src, time, candidates)
    exact = index.source_support(src, time, candidates, 0)
    support = index.source_support(src, time, candidates, WINDOW_SECONDS) - exact
    if support.min(initial=0) < 0:
        raise ValueError("source-session support became negative")
    residual = d3.qnorm(np.log1p(support).astype(np.float32))

    with zipfile.ZipFile(args.base) as source:
        if tuple(source.namelist()) != ("dataset3.csv", "dataset4.csv"):
            raise ValueError("c1 submission members differ")
        with source.open("dataset3.csv") as handle:
            base = np.loadtxt(handle, delimiter=",", dtype=np.float32)
        if base.shape != (ROWS["dataset3"], WIDTH):
            raise ValueError("c1 Dataset3 shape differs")
        logits = np.log(np.clip(base, np.float32(1e-12), None))
        probability = softmax(logits + np.float32(RESIDUAL_WEIGHT) * residual)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6,
                allowZip64=True,
            ) as destination:
                with destination.open("dataset3.csv", "w", force_zip64=True) as raw:
                    with io.TextIOWrapper(raw, encoding="ascii", newline="\n") as text:
                        np.savetxt(text, probability, fmt="%.8f", delimiter=",")
                with source.open("dataset4.csv") as incoming:
                    with destination.open(
                        "dataset4.csv", "w", force_zip64=True
                    ) as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=8 << 20)
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)

    changed = np.argmax(base, axis=1) != np.argmax(probability, axis=1)
    with zipfile.ZipFile(args.output) as archive:
        d4_sha256 = member_sha256(archive, "dataset4.csv")
    manifest = {
        "kind": "b_rank_d34_c2_d3_source_session_submission_v1",
        "submission_sha256": sha256(args.output),
        "data_sha256": d3.DATA_SHA256,
        "base_sha256": base_hash,
        "base_manifest_sha256": sha256(args.base_manifest),
        "gate_report": str(args.gate_report.resolve()),
        "gate_report_sha256": sha256(args.gate_report),
        "policy": expected_policy,
        "dataset3": {
            "rows": ROWS["dataset3"],
            "signal_cell_rate": float(np.mean(support > 0)),
            "signal_row_rate": float(np.mean(np.any(support > 0, axis=1))),
            "top1_changed_rate": float(np.mean(changed)),
            "top1_changed_rows": int(changed.sum()),
        },
        "dataset4": {
            "rows": ROWS["dataset4"],
            "csv_sha256": d4_sha256,
            "identical_to_base": True,
        },
        "source_sha256": sha256(Path(__file__).resolve()),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
