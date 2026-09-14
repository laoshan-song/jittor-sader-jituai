#!/usr/bin/env python3
"""Measure how closely a regenerated base matches the original locked base.

Answers the practical question: once the full pipeline regenerates a frozen
base on the server, what fraction of the original (recorded online score
1.5240999401892983) base does it reproduce? Byte-exactness is not expected
because historical checkpoints are missing and Jittor's CUDA operators perturb
low-order bits per machine; this reports the agreement that does survive.

Two inputs are compared. Each may be a frozen_base.ckpt or a result.zip; the
Dataset3/Dataset4 score matrices are decoded with the same codec the reranker
uses, so the comparison is on the real scores.

    python compare_frozen_base.py --a original.ckpt --b regenerated.ckpt
    python compare_frozen_base.py --a original_result.zip --b fresh_result.zip

Reported per dataset:
  * byte_identical         - the two encodings are the same file
  * top1_match_rate        - rows whose top-ranked candidate agrees
  * full_order_match_rate  - rows whose full 100-slot ranking agrees
  * score_correlation      - Pearson correlation of the raw scores
  * score_mae              - mean absolute score error

No GPU, official data, or Jittor is required.
"""

from __future__ import annotations

import argparse
import json
import lzma
import zipfile
from pathlib import Path

import numpy as np


D3_ROWS = 157_670
D4_ROWS = 2_322_538
WIDTH = 100
D3_SCALE = 10_000_000_000
D3_PLANES = 35
Q7_LEVELS = np.float32((1 << 7) - 1)
Q7_D4_MAGIC = b"TRACK1-B-D4-Q7-V1\n"
FROZEN_KIND = "track1_b_frozen_score_q7_d3q35_v1"


# --------------------------------------------------------------------------- #
# Decoders (shared with build_submission.py / pack_frozen_base.py).
# --------------------------------------------------------------------------- #
def _decode_d3_q35(payload: np.ndarray, rows: int) -> np.ndarray:
    count = rows * WIDTH
    plane_bytes = (count + 7) // 8
    encoded = lzma.decompress(np.asarray(payload, dtype=np.uint8).tobytes())
    if len(encoded) != D3_PLANES * plane_bytes:
        raise ValueError("Dataset3 q35 stream size differs")
    planes = np.frombuffer(encoded, dtype=np.uint8).reshape(D3_PLANES, plane_bytes)
    zigzag = np.zeros(count, dtype=np.uint64)
    for bit in range(D3_PLANES):
        bits = np.unpackbits(planes[bit], bitorder="little", count=count)
        zigzag |= bits.astype(np.uint64) << np.uint64(bit)
    fixed = (zigzag >> np.uint64(1)).astype(np.int64) ^ -(zigzag & np.uint64(1)).astype(np.int64)
    return fixed.reshape(rows, WIDTH).astype(np.float64) / D3_SCALE


def _decode_d4_q7(payload: bytes) -> np.ndarray:
    if payload[: len(Q7_D4_MAGIC)] != Q7_D4_MAGIC:
        raise ValueError("packed Dataset4 magic differs")
    body = np.frombuffer(payload[len(Q7_D4_MAGIC):], dtype=np.uint8)
    bits = np.unpackbits(body, bitorder="little")
    shifts = np.arange(7, dtype=np.uint8)
    values = (bits.reshape(-1, 7) << shifts).sum(axis=1, dtype=np.uint16).astype(np.uint8)
    return values.reshape(-1, WIDTH).astype(np.float32) / Q7_LEVELS


def _read_csv_matrix(archive: zipfile.ZipFile, member: str, rows: int) -> np.ndarray:
    with archive.open(member) as handle:
        matrix = np.loadtxt(handle, delimiter=",", dtype=np.float64)
    if matrix.shape != (rows, WIDTH):
        raise ValueError(f"{member} shape {matrix.shape} differs from {(rows, WIDTH)}")
    return matrix


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, bytes, bytes]:
    """Return (d3, d4, d3_raw_bytes, d4_raw_bytes) for a ckpt or result.zip."""
    if path.suffix == ".ckpt" or zipfile.is_zipfile(path) is False:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["kind"].item()) != FROZEN_KIND:
                raise ValueError(f"{path} is not a frozen base checkpoint")
            d3_raw = np.asarray(archive["dataset3_q35_lzma"], dtype=np.uint8)
            d4_raw = np.asarray(archive["dataset4_q7"], dtype=np.uint8).tobytes()
        return _decode_d3_q35(d3_raw, D3_ROWS), _decode_d4_q7(d4_raw), d3_raw.tobytes(), d4_raw
    with zipfile.ZipFile(path) as archive:
        d3 = _read_csv_matrix(archive, "dataset3.csv", D3_ROWS)
        d4 = _read_csv_matrix(archive, "dataset4.csv", D4_ROWS)
        d3_raw = archive.read("dataset3.csv")
        d4_raw = archive.read("dataset4.csv")
    return d3, d4, d3_raw, d4_raw


# --------------------------------------------------------------------------- #
# Agreement metrics.
# --------------------------------------------------------------------------- #
def compare_matrix(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise ValueError(f"score shapes differ: {a.shape} vs {b.shape}")
    rows = len(a)
    top1 = int(np.sum(np.argmax(a, axis=1) == np.argmax(b, axis=1)))
    order_a = np.argsort(-a, axis=1, kind="stable")
    order_b = np.argsort(-b, axis=1, kind="stable")
    full_order = int(np.sum(np.all(order_a == order_b, axis=1)))
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    denom = np.std(af) * np.std(bf)
    correlation = float(np.mean((af - af.mean()) * (bf - bf.mean())) / denom) if denom else 1.0
    return {
        "rows": rows,
        "top1_match_rows": top1,
        "top1_match_rate": top1 / rows,
        "full_order_match_rows": full_order,
        "full_order_match_rate": full_order / rows,
        "score_correlation": correlation,
        "score_mae": float(np.mean(np.abs(af - bf))),
        "score_max_abs_error": float(np.max(np.abs(af - bf))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, required=True, help="original base (ckpt or result.zip)")
    parser.add_argument("--b", type=Path, required=True, help="regenerated base (ckpt or result.zip)")
    parser.add_argument("--output", type=Path, help="write the report JSON here")
    args = parser.parse_args()

    a_d3, a_d4, a_d3_raw, a_d4_raw = load_scores(args.a)
    b_d3, b_d4, b_d3_raw, b_d4_raw = load_scores(args.b)

    report = {
        "kind": "track1_b_frozen_base_agreement_v1",
        "a": str(args.a),
        "b": str(args.b),
        "dataset3": {
            "byte_identical": a_d3_raw == b_d3_raw,
            **compare_matrix(a_d3, b_d3),
        },
        "dataset4": {
            "byte_identical": a_d4_raw == b_d4_raw,
            **compare_matrix(a_d4, b_d4),
        },
    }
    # A single headline number: candidate-rank agreement weighted by row count.
    total_rows = report["dataset3"]["rows"] + report["dataset4"]["rows"]
    report["overall_top1_match_rate"] = (
        report["dataset3"]["top1_match_rows"] + report["dataset4"]["top1_match_rows"]
    ) / total_rows
    report["overall_full_order_match_rate"] = (
        report["dataset3"]["full_order_match_rows"] + report["dataset4"]["full_order_match_rows"]
    ) / total_rows

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
