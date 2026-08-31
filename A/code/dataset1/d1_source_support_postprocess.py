#!/usr/bin/env python3
"""Apply the locked same-source cross-row support rule to Dataset1.

The learned scorer remains the frozen two-checkpoint Jittor ensemble. This
deterministic postprocessor uses only the official Dataset1 test candidate
matrix. For each source, it counts how many query rows contain each candidate.
When one candidate has support at least four and leads the runner-up by at
least two rows, that candidate is promoted to rank one without changing the
relative order of the other 99 candidates.
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


ROWS = 61_051
CANDIDATES = 100
KEY_BASE = 1 << 32
MINIMUM_SUPPORT = 4
MINIMUM_GAP = 2
PROMOTION_LOGIT_MARGIN = 1.0
FROZEN_RESULT_SHA256 = (
    "4c8fca6a041a94957b04a2df9f958d76098d06ab67093679fea766c364e3a28f"
)
FROZEN_DATASET1_SHA256 = (
    "05c91c10d7a5a7f7c9b9b8d75948c8c943da6c4cf7ab41b2ccc84064e71ed335"
)
FROZEN_DATASET2_SHA256 = (
    "425cb07df6996dab717e139db1a9bdf328aa479e64152fc4955a6f91e66944d4"
)
OFFICIAL_TEST_SHA256 = (
    "399b48c09bd380c5552ee8c4286fdf5144484ca7860bc1afeadeacecadce7b5c"
)
LOCKED_CONFIG_SHA256 = (
    "643565eb9f4fed1d20136adcd7890899e946118b38ec136d45cab9be8048daaa"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


def load_rule(path: Path) -> tuple[dict[str, object], str]:
    actual_hash = sha256_file(path)
    if actual_hash != LOCKED_CONFIG_SHA256:
        raise ValueError(
            f"locked rule hash mismatch: {actual_hash} != {LOCKED_CONFIG_SHA256}"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    rule = config.get("rule", {})
    expected = {
        "minimum_winner_support": MINIMUM_SUPPORT,
        "minimum_support_gap": MINIMUM_GAP,
        "action": "promote_unique_support_winner_to_rank_1",
    }
    if rule != expected:
        raise ValueError(f"locked rule mismatch: {rule} != {expected}")
    if config.get("candidate_universe_size") != 23_852:
        raise ValueError("locked candidate universe size changed")
    return config, actual_hash


def parse_test_csv(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse the fixed official candidate matrix without a dataframe dependency."""
    expected = ["src", "time"] + [f"c{i}" for i in range(1, CANDIDATES + 1)]
    handle = io.BytesIO(payload)
    header = handle.readline().decode("ascii").rstrip("\r\n").split(",")
    if header != expected:
        raise ValueError("unexpected Dataset1 test schema")
    values = np.empty((ROWS, CANDIDATES + 2), dtype=np.int64)
    for row, line in enumerate(handle):
        if row >= ROWS:
            raise ValueError("unexpected Dataset1 test row count")
        parsed = np.fromstring(line.decode("ascii"), sep=",", dtype=np.int64)
        if parsed.shape != (CANDIDATES + 2,):
            raise ValueError(f"unexpected Dataset1 test row width at {row}")
        values[row] = parsed
    if row + 1 != ROWS:
        raise ValueError("unexpected Dataset1 test row count")
    return values[:, 0].copy(), values[:, 2:].astype(np.int32, copy=True)


def load_test(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    member = path / "dataset1/test.csv" if path.is_dir() else None
    if member is not None:
        payload = member.read_bytes()
        actual_hash = sha256_bytes(payload)
    else:
        with zipfile.ZipFile(path) as archive:
            payload = archive.read("dataset1/test.csv")
        actual_hash = sha256_bytes(payload)
    if actual_hash != OFFICIAL_TEST_SHA256:
        raise ValueError(
            f"official Dataset1 test hash mismatch: {actual_hash} "
            f"!= {OFFICIAL_TEST_SHA256}"
        )
    source, candidates = parse_test_csv(payload)
    if source.min() < 0 or source.max() >= (1 << 31):
        raise ValueError("source IDs outside the pinned key range")
    if candidates.min() < 0 or candidates.max() >= (1 << 31):
        raise ValueError("candidate IDs outside the pinned key range")
    ordered = np.sort(candidates, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError("official Dataset1 candidates must be unique within each row")
    if np.unique(candidates).size != 23_852:
        raise ValueError("official Dataset1 candidate universe changed")
    return source, candidates, actual_hash


def source_support(
    source: np.ndarray, candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cell support, winning column, maximum support and winner gap."""
    if candidates.shape != (len(source), CANDIDATES):
        raise ValueError("invalid Dataset1 source-support inputs")
    if np.any(np.sort(candidates, axis=1)[:, 1:] == np.sort(
        candidates, axis=1
    )[:, :-1]):
        raise ValueError("source support requires unique candidates per row")
    keys = source[:, None].astype(np.int64) * KEY_BASE + candidates.astype(
        np.int64, copy=False
    )
    _, inverse, counts = np.unique(
        keys.ravel(), return_inverse=True, return_counts=True
    )
    support = counts[inverse].reshape(candidates.shape).astype(np.int32)
    winner = np.argmax(support, axis=1).astype(np.int32)
    top_two = np.partition(support, -2, axis=1)[:, -2:]
    maximum = top_two[:, 1].astype(np.int32, copy=False)
    gap = (top_two[:, 1] - top_two[:, 0]).astype(np.int32, copy=False)
    return support, winner, maximum, gap


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    output = np.exp(shifted)
    return output / output.sum(axis=1, keepdims=True)


def build_candidate(
    probability: np.ndarray,
    winner: np.ndarray,
    maximum: np.ndarray,
    gap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if probability.shape != (len(winner), CANDIDATES):
        raise ValueError("invalid Dataset1 frozen probability shape")
    if maximum.shape != winner.shape or gap.shape != winner.shape:
        raise ValueError("invalid Dataset1 support summary shape")
    baseline_top = np.argmax(probability, axis=1)
    eligible = (maximum >= MINIMUM_SUPPORT) & (gap >= MINIMUM_GAP)
    changed = eligible & (winner != baseline_top)
    candidate = probability.copy()
    rows = np.flatnonzero(changed)
    if len(rows):
        logits = np.log(np.clip(probability[rows], 1e-300, None))
        logits[np.arange(len(rows)), winner[rows]] = (
            logits.max(axis=1) + PROMOTION_LOGIT_MARGIN
        )
        candidate[rows] = softmax(logits)
    if not np.isfinite(candidate).all() or np.any(candidate < 0.0):
        raise FloatingPointError("candidate probabilities are invalid")
    if not np.array_equal(np.argmax(candidate, axis=1)[changed], winner[changed]):
        raise AssertionError("support winner was not promoted to rank one")
    return candidate, changed


def load_frozen_result(path: Path) -> tuple[bytes, bytes, np.ndarray, str]:
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
    if sha256_bytes(dataset1) != FROZEN_DATASET1_SHA256:
        raise ValueError("frozen Dataset1 member hash mismatch")
    if sha256_bytes(dataset2) != FROZEN_DATASET2_SHA256:
        raise ValueError("frozen Dataset2 member hash mismatch")
    probability = np.loadtxt(
        io.BytesIO(dataset1), delimiter=",", dtype=np.float64
    )
    if probability.shape != (ROWS, CANDIDATES):
        raise ValueError(f"unexpected frozen Dataset1 shape: {probability.shape}")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("invalid frozen Dataset1 probabilities")
    return dataset1, dataset2, probability, actual_hash


def serialize_probability(probability: np.ndarray) -> bytes:
    return "".join(
        ",".join(f"{value:.8f}" for value in row) + "\n"
        for row in probability
    ).encode("ascii")


def relative_nonwinner_order_is_preserved(
    before: np.ndarray,
    after: np.ndarray,
    winner: np.ndarray,
    changed: np.ndarray,
) -> bool:
    for row in np.flatnonzero(changed):
        column = int(winner[row])
        before_order = np.argsort(-before[row], kind="stable")
        after_order = np.argsort(-after[row], kind="stable")
        before_order = before_order[before_order != column]
        after_order = after_order[after_order != column]
        if not np.array_equal(before_order, after_order):
            return False
    return True


def top_overlap(before: np.ndarray, after: np.ndarray, k: int) -> float:
    before_top = np.argpartition(before, -k, axis=1)[:, -k:]
    after_top = np.argpartition(after, -k, axis=1)[:, -k:]
    overlap = np.zeros(len(before), dtype=np.int16)
    for column in range(k):
        overlap += np.any(
            after_top == before_top[:, column, None], axis=1
        ).astype(np.int16)
    return float(np.mean(overlap / k))


def diagnostics(
    baseline: np.ndarray,
    candidate: np.ndarray,
    source: np.ndarray,
    support: np.ndarray,
    winner: np.ndarray,
    maximum: np.ndarray,
    gap: np.ndarray,
    changed: np.ndarray,
    baseline_hash: str,
    test_hash: str,
    config_hash: str,
) -> dict[str, object]:
    eligible = (maximum >= MINIMUM_SUPPORT) & (gap >= MINIMUM_GAP)
    rows = np.arange(len(source))
    winner_probability = baseline[rows, winner]
    winner_rank = 1 + (baseline > winner_probability[:, None]).sum(axis=1)
    return {
        "kind": "dataset1_same_source_cross_row_candidate_support_v1",
        "config_id": "support_ge4_unique_gap_ge2_hard_promote",
        "submission_eligible": False,
        "frozen_result_sha256": baseline_hash,
        "official_dataset1_test_sha256": test_hash,
        "locked_rule_sha256": config_hash,
        "rows": int(len(source)),
        "candidate_cells": int(support.size),
        "sources": int(np.unique(source).size),
        "minimum_support": MINIMUM_SUPPORT,
        "minimum_gap": MINIMUM_GAP,
        "promotion_logit_margin": PROMOTION_LOGIT_MARGIN,
        "eligible_rows": int(eligible.sum()),
        "eligible_row_rate": float(eligible.mean()),
        "eligible_sources": int(np.unique(source[eligible]).size),
        "top1_changed_rows": int(changed.sum()),
        "top1_changed_rate": float(changed.mean()),
        "eligible_already_top1_rows": int((eligible & ~changed).sum()),
        "changed_winner_support_min": int(maximum[changed].min())
        if np.any(changed)
        else 0,
        "changed_winner_support_mean": float(maximum[changed].mean()),
        "changed_support_gap_min": int(gap[changed].min())
        if np.any(changed)
        else 0,
        "changed_support_gap_mean": float(gap[changed].mean()),
        "changed_winner_baseline_rank_mean": float(winner_rank[changed].mean()),
        "changed_winner_baseline_rank_max": int(winner_rank[changed].max(initial=0)),
        "top5_overlap": top_overlap(baseline, candidate, 5),
        "nonwinner_relative_order_preserved": relative_nonwinner_order_is_preserved(
            baseline, candidate, winner, changed
        ),
        "unchanged_rows_value_identical": bool(
            np.array_equal(baseline[~changed], candidate[~changed])
        ),
        "baseline_row_sum_max_error": float(
            np.max(np.abs(baseline.sum(axis=1) - 1.0))
        ),
        "candidate_row_sum_max_error_before_serialization": float(
            np.max(np.abs(candidate.sum(axis=1) - 1.0))
        ),
    }


def serialized_row_differences(before: bytes, after: bytes) -> np.ndarray:
    before_rows = before.splitlines()
    after_rows = after.splitlines()
    if len(before_rows) != ROWS or len(after_rows) != ROWS:
        raise ValueError("serialized Dataset1 row count changed")
    return np.fromiter(
        (left != right for left, right in zip(before_rows, after_rows)),
        dtype=bool,
        count=ROWS,
    )


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
            if archive.read("dataset2.csv") != dataset2:
                raise RuntimeError("Dataset2 was not preserved byte-for-byte")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
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

    _, config_hash = load_rule(args.config)
    source, candidates, test_hash = load_test(args.data)
    frozen_dataset1, frozen_dataset2, baseline, baseline_hash = (
        load_frozen_result(args.baseline)
    )
    if serialize_probability(baseline) != frozen_dataset1:
        raise ValueError("frozen Dataset1 is not canonical eight-decimal CSV")

    support, winner, maximum, gap = source_support(source, candidates)
    candidate, changed = build_candidate(baseline, winner, maximum, gap)
    candidate_dataset1 = serialize_probability(candidate)
    serialized_changed = serialized_row_differences(
        frozen_dataset1, candidate_dataset1
    )
    if not np.array_equal(serialized_changed, changed):
        raise AssertionError("serialized changes differ from promoted-row mask")

    report = diagnostics(
        baseline,
        candidate,
        source,
        support,
        winner,
        maximum,
        gap,
        changed,
        baseline_hash,
        test_hash,
        config_hash,
    )
    serialized = np.loadtxt(
        io.BytesIO(candidate_dataset1), delimiter=",", dtype=np.float64
    )
    report.update(
        {
            "serialized_changed_rows": int(serialized_changed.sum()),
            "candidate_dataset1_sha256": sha256_bytes(candidate_dataset1),
            "preserved_dataset2_sha256": sha256_bytes(frozen_dataset2),
            "candidate_row_sum_max_error_after_serialization": float(
                np.max(np.abs(serialized.sum(axis=1) - 1.0))
            ),
            "serialized_finite": bool(np.isfinite(serialized).all()),
            "serialized_nonnegative": bool(np.all(serialized >= 0.0)),
        }
    )
    if report["candidate_row_sum_max_error_after_serialization"] > 1e-6:
        raise ValueError("serialized Dataset1 row sum exceeds submission tolerance")

    if args.output is not None:
        write_result(args.output, candidate_dataset1, frozen_dataset2)
        report.update(
            {
                "submission_eligible": True,
                "candidate_result": str(args.output),
                "candidate_result_sha256": sha256_file(args.output),
            }
        )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
