#!/usr/bin/env python3
"""Generate a B-list candidate base from official training data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
WIDTH = 100
ROWS = {"dataset3": 157_670, "dataset4": 2_322_538}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_member(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def popularity(official: zipfile.ZipFile, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    with official.open(f"{dataset}/train.csv") as handle:
        destination = pd.read_csv(handle, usecols=["dst"], dtype=np.uint32)["dst"].to_numpy()
    ids, counts = np.unique(destination, return_counts=True)
    return ids, np.log1p(counts).astype(np.float32)


def lookup(candidates: np.ndarray, ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    flat = candidates.reshape(-1)
    positions = np.searchsorted(ids, flat)
    inside = positions < len(ids)
    matched = np.zeros(flat.shape, dtype=bool)
    matched[inside] = ids[positions[inside]] == flat[inside]
    output = np.zeros(flat.shape, dtype=np.float32)
    output[matched] = values[positions[matched]]
    return output.reshape(candidates.shape)


def write_scores(
    official: zipfile.ZipFile,
    dataset: str,
    ids: np.ndarray,
    values: np.ndarray,
    output: zipfile.ZipFile,
    chunk_rows: int,
) -> int:
    columns = [f"c{index}" for index in range(1, WIDTH + 1)]
    rows = 0
    with official.open(f"{dataset}/test.csv") as test_file, output.open(
        fixed_member(f"{dataset}.csv"), "w", force_zip64=True
    ) as raw:
        chunks = pd.read_csv(
            test_file,
            usecols=columns,
            dtype=np.uint32,
            chunksize=chunk_rows,
        )
        for test in chunks:
            candidates = test[columns].to_numpy(dtype=np.uint32, copy=False)
            scores = lookup(candidates, ids, values)
            buffer = io.BytesIO()
            np.savetxt(buffer, scores, delimiter=",", fmt="%.8f")
            raw.write(buffer.getvalue())
            rows += len(test)
    if rows != ROWS[dataset]:
        raise ValueError(f"{dataset} row count differs: {rows} != {ROWS[dataset]}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite base: {args.output}")
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with zipfile.ZipFile(args.data) as official, zipfile.ZipFile(
        args.output,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as base:
        d3_ids, d3_values = popularity(official, "dataset3")
        d4_ids, d4_values = popularity(official, "dataset4")
        d3_rows = write_scores(official, "dataset3", d3_ids, d3_values, base, args.chunk_rows)
        d4_rows = write_scores(official, "dataset4", d4_ids, d4_values, base, args.chunk_rows)

    receipt = {
        "kind": "track1_b_official_popularity_base_v1",
        "decision": "PASS_SUPPLEMENTARY_BASE",
        "official_data_sha256": DATA_SHA256,
        "base_sha256": sha256(args.output),
        "base": str(args.output),
        "members": ["dataset3.csv", "dataset4.csv"],
        "rows": {"dataset3": d3_rows, "dataset4": d4_rows},
        "width": WIDTH,
        "source": {
            "dataset3": "log1p(destination_frequency(dataset3/train.csv))",
            "dataset4": "log1p(destination_frequency(dataset4/train.csv))",
        },
        "elapsed_seconds": time.time() - started,
    }
    receipt_path = args.output.with_name("BASE_RECEIPT.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
