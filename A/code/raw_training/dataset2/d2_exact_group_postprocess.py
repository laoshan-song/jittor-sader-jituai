#!/usr/bin/env python3
"""Apply a fixed exact-(src,time) candidate-consensus residual to Dataset2.

The learned baseline remains the frozen Jittor ensemble.  This script adds no
learned component: it derives a deterministic candidate-only signal from the
official test matrix.  For each candidate cell, the signal is one when the
same item appears in at least one *other* row with the same ``(src,time)``.
Repeated cells in the current row do not count as cross-row evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np


ROWS = 153420
CANDIDATES = 100
ITEM_KEY_BASE = 110370
WEIGHT = 0.05
FROZEN_RESULT_SHA256 = (
    "4c8fca6a041a94957b04a2df9f958d76098d06ab67093679fea766c364e3a28f"
)
OFFICIAL_TEST_SHA256 = (
    "389e330d6a21317cc1a0a013c878850c2324e916b2392901b3c039384e201372"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + np.float32(1e-6)
    )


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    output = np.exp(values)
    return output / output.sum(axis=1, keepdims=True)


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


def parse_test_csv(payload: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse the fixed official candidate matrix without a dataframe dependency."""
    expected = ["src", "time"] + [f"c{i}" for i in range(1, CANDIDATES + 1)]
    handle = io.BytesIO(payload)
    header = handle.readline().decode("ascii").rstrip("\r\n").split(",")
    if header != expected:
        raise ValueError("unexpected Dataset2 test schema")
    values = np.empty((ROWS, CANDIDATES + 2), dtype=np.int64)
    for row, line in enumerate(handle):
        if row >= ROWS:
            raise ValueError("unexpected Dataset2 test row count")
        parsed = np.fromstring(line.decode("ascii"), sep=",", dtype=np.int64)
        if parsed.shape != (CANDIDATES + 2,):
            raise ValueError(f"unexpected Dataset2 test row width at {row}")
        values[row] = parsed
    if row + 1 != ROWS:
        raise ValueError("unexpected Dataset2 test row count")
    return values[:, 0].copy(), values[:, 1].copy(), values[:, 2:].astype(np.int32, copy=True)


def load_test(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if path.is_dir():
        member = path / "dataset2/test.csv"
        payload = member.read_bytes()
        actual_hash = sha256_bytes(payload)
    else:
        with zipfile.ZipFile(path) as archive:
            payload = archive.read("dataset2/test.csv")
        actual_hash = sha256_bytes(payload)
    if actual_hash != OFFICIAL_TEST_SHA256:
        raise ValueError(
            f"official Dataset2 test hash mismatch: {actual_hash} "
            f"!= {OFFICIAL_TEST_SHA256}"
        )
    src, times, candidates = parse_test_csv(payload)
    if candidates.min() < 1 or candidates.max() >= ITEM_KEY_BASE:
        raise ValueError("candidate IDs outside the pinned official range")
    return src, times, candidates, actual_hash


def exact_other_row_support(
    src: np.ndarray, times: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    """Count distinct other rows in the exact query group containing an item."""
    if candidates.shape != (len(src), CANDIDATES) or times.shape != src.shape:
        raise ValueError("invalid exact-group inputs")
    query_key = (src.astype(np.int64) << 32) | times.astype(np.int64)
    _, group = np.unique(query_key, return_inverse=True)
    group = group.astype(np.int32, copy=False)

    order = np.argsort(candidates, axis=1, kind="stable").astype(
        np.uint8, copy=False
    )
    sorted_candidates = np.take_along_axis(candidates, order, axis=1)
    group_item = (
        group[:, None].astype(np.int64) * ITEM_KEY_BASE + sorted_candidates
    )
    _, inverse = np.unique(group_item, return_inverse=True)
    inverse = inverse.astype(np.int32, copy=False)

    distinct_in_row = np.empty(sorted_candidates.shape, dtype=bool)
    distinct_in_row[:, 0] = True
    distinct_in_row[:, 1:] = (
        sorted_candidates[:, 1:] != sorted_candidates[:, :-1]
    )
    flat_inverse = inverse.ravel()
    presence = np.bincount(
        flat_inverse[distinct_in_row.ravel()], minlength=int(flat_inverse.max()) + 1
    ).astype(np.int16, copy=False)
    support_sorted = (
        presence[inverse].reshape(sorted_candidates.shape) - 1
    ).astype(np.int16, copy=False)
    support = np.empty_like(support_sorted)
    np.put_along_axis(support, order, support_sorted, axis=1)
    if support.min(initial=0) < 0:
        raise AssertionError("negative cross-row support")
    return support


def load_frozen_result(path: Path) -> tuple[bytes, np.ndarray, str]:
    actual_hash = sha256_file(path)
    if actual_hash != FROZEN_RESULT_SHA256:
        raise ValueError(
            f"frozen result hash mismatch: {actual_hash} != {FROZEN_RESULT_SHA256}"
        )
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("frozen result ZIP failed CRC validation")
        if archive.namelist() != ["dataset1.csv", "dataset2.csv"]:
            raise ValueError("unexpected frozen result members")
        dataset1 = archive.read("dataset1.csv")
        dataset2 = archive.read("dataset2.csv")
    probability = np.loadtxt(
        io.BytesIO(dataset2), delimiter=",", dtype=np.float64
    )
    if probability.shape != (ROWS, CANDIDATES):
        raise ValueError(f"unexpected frozen Dataset2 shape: {probability.shape}")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("invalid frozen probabilities")
    return dataset1, probability, actual_hash


def reciprocal_top_overlap(before: np.ndarray, after: np.ndarray, k: int) -> float:
    before_top = np.argpartition(before, -k, axis=1)[:, -k:]
    after_top = np.argpartition(after, -k, axis=1)[:, -k:]
    overlap = np.zeros(len(before), dtype=np.int16)
    for column in range(k):
        overlap += np.any(
            after_top == before_top[:, column, None], axis=1
        ).astype(np.int16)
    return float(np.mean(overlap / k))


def build_candidate(
    probability: np.ndarray, support: np.ndarray, weight: float
) -> tuple[np.ndarray, np.ndarray]:
    if weight != WEIGHT:
        raise ValueError(f"only the validation-frozen weight {WEIGHT} is allowed")
    baseline = qnorm(np.log(np.clip(probability, 1e-12, None)))
    signal = qnorm((support > 0).astype(np.float32))
    logits = baseline + np.float32(weight) * signal
    candidate = softmax(logits)
    if not np.isfinite(candidate).all():
        raise FloatingPointError("candidate probabilities are non-finite")
    return candidate, logits


def diagnostics(
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    candidates: np.ndarray,
    support: np.ndarray,
    baseline_hash: str,
    test_hash: str,
) -> dict[str, object]:
    before_top = np.argmax(baseline_probability, axis=1)
    after_top = np.argmax(candidate_probability, axis=1)
    active = np.any(support > 0, axis=1)
    chosen_supported = support[np.arange(len(support)), after_top] > 0
    return {
        "kind": "dataset2_exact_src_time_candidate_consensus_conservative_v2",
        "config_id": "binary_distinct_other_row_qnorm_weight_0.05",
        "submission_eligible": False,
        "frozen_result_sha256": baseline_hash,
        "official_dataset2_test_sha256": test_hash,
        "weight": WEIGHT,
        "weight_policy": "fixed conservative weight from causal and strong-baseline pressure audits",
        "normalization": "qnorm(log(frozen_probability)) + weight*qnorm(support>0)",
        "support_definition": "distinct other rows in exact (src,time) group containing candidate",
        "rows": int(len(candidates)),
        "candidate_cells": int(candidates.size),
        "active_rows": int(active.sum()),
        "active_row_rate": float(active.mean()),
        "support_cell_rate": float(np.mean(support > 0)),
        "support_ge2_cell_rate": float(np.mean(support >= 2)),
        "top1_changed_rows": int(np.sum(before_top != after_top)),
        "top1_changed_rate": float(np.mean(before_top != after_top)),
        "after_top1_supported_rate": float(chosen_supported.mean()),
        "top5_overlap": reciprocal_top_overlap(
            baseline_probability, candidate_probability, 5
        ),
        "baseline_row_sum_max_error": float(
            np.max(np.abs(baseline_probability.sum(axis=1) - 1.0))
        ),
        "candidate_row_sum_max_error_before_serialization": float(
            np.max(np.abs(candidate_probability.sum(axis=1) - 1.0))
        ),
    }


def serialize_dataset2(probability: np.ndarray) -> bytes:
    return "".join(
        ",".join(f"{value:.8f}" for value in row) + "\n"
        for row in probability
    ).encode("ascii")


def write_result(path: Path, dataset1: bytes, dataset2: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with zipfile.ZipFile(temporary_path, "w") as archive:
            archive.writestr(zip_info("dataset1.csv"), dataset1)
            archive.writestr(zip_info("dataset2.csv"), dataset2)
        with zipfile.ZipFile(temporary_path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("candidate ZIP failed CRC validation")
            if archive.namelist() != ["dataset1.csv", "dataset2.csv"]:
                raise RuntimeError("candidate ZIP members changed")
            if archive.read("dataset1.csv") != dataset1:
                raise RuntimeError("Dataset1 was not preserved byte-for-byte")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only == (args.output is not None):
        parser.error("choose exactly one of --audit-only or --output")
    if args.audit_output.exists():
        raise FileExistsError(f"audit output already exists: {args.audit_output}")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"candidate output already exists: {args.output}")

    dataset1, baseline_probability, baseline_hash = load_frozen_result(
        args.baseline
    )
    src, times, candidates, test_hash = load_test(args.data)
    support = exact_other_row_support(src, times, candidates)
    candidate_probability, _ = build_candidate(
        baseline_probability, support, WEIGHT
    )
    report = diagnostics(
        baseline_probability,
        candidate_probability,
        candidates,
        support,
        baseline_hash,
        test_hash,
    )

    if args.output is not None:
        dataset2 = serialize_dataset2(candidate_probability)
        write_result(args.output, dataset1, dataset2)
        report.update(
            {
                "submission_eligible": True,
                "candidate_result": str(args.output),
                "candidate_result_sha256": sha256_file(args.output),
                "candidate_dataset1_sha256": sha256_bytes(dataset1),
                "candidate_dataset2_sha256": sha256_bytes(dataset2),
            }
        )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
