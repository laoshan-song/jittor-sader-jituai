#!/usr/bin/env python3
"""Apply the frozen causal RP3 policy to ruc3 Dataset4 and build a submission."""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from numba import njit, prange

import d4_rp3beta_gate as rp3


ROWS = {"dataset3.csv": 157_670, "dataset4.csv": 2_322_538}
WIDTH = 100
ALPHA = 0.01
BETA = 1.0


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


def test_chunks(data: Path, chunk_rows: int):
    with zipfile.ZipFile(data) as archive, archive.open("dataset4/test.csv") as raw:
        with io.TextIOWrapper(raw, encoding="ascii", newline="") as text:
            header = text.readline().strip()
            expected = "src,time," + ",".join(f"c{i}" for i in range(1, WIDTH + 1))
            if header != expected:
                raise ValueError("official Dataset4 test header differs")
            while True:
                lines = list(itertools.islice(text, chunk_rows))
                if not lines:
                    return
                payload = ",".join(line.strip() for line in lines)
                values = np.fromstring(payload, dtype=np.uint64, sep=",")
                if values.size != len(lines) * (WIDTH + 2):
                    raise ValueError("official Dataset4 test row differs")
                values = values.reshape((-1, WIDTH + 2))
                yield (
                    values[:, 0].astype(np.uint32),
                    values[:, 1].astype(np.uint32),
                    values[:, 2:].astype(np.uint32),
                )


@njit(parallel=True, cache=True)
def pair_seen(
    source: np.ndarray,
    candidates: np.ndarray,
    source_base: int,
    indptr: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    output = np.zeros(candidates.shape, dtype=np.bool_)
    for row in prange(len(source)):
        query = int(source[row]) - source_base
        if query < 0 or query + 1 >= len(indptr):
            continue
        start = indptr[query]
        stop = indptr[query + 1]
        for column in range(candidates.shape[1]):
            target = int(candidates[row, column])
            left = start
            right = stop
            while left < right:
                middle = (left + right) // 2
                if indices[middle] < target:
                    left = middle + 1
                else:
                    right = middle
            output[row, column] = left < stop and indices[left] == target
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--max-candidate-degree", type=float, default=4096.0)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite candidate output")
    rp3.set_num_threads(args.threads)
    started = time.time()

    raw_source = np.load(args.data_cache / "src.npy", mmap_mode="r")
    raw_item = np.load(args.data_cache / "dst.npy", mmap_mode="r")
    raw_time = np.load(args.data_cache / "time.npy", mmap_mode="r")
    cutoff = int(raw_time[-1]) + 1
    forward, reverse, source_base, events = rp3.graph(
        raw_source, raw_item, raw_time, cutoff
    )
    print(json.dumps({"graph_events": events, "unique_edges": len(forward.indices)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    total = changed = active = duplicate_count = pair_seen_errors = 0
    maximum_row_sum_error = 0.0
    try:
        with zipfile.ZipFile(args.base) as base, zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
        ) as destination:
            if tuple(base.namelist()) != tuple(ROWS):
                raise ValueError("base submission members differ")
            with base.open("dataset3.csv") as incoming, destination.open(
                "dataset3.csv", "w", force_zip64=True
            ) as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=8 << 20)

            with base.open("dataset4.csv") as base_raw, destination.open(
                "dataset4.csv", "w", force_zip64=True
            ) as output_raw:
                with io.TextIOWrapper(base_raw, encoding="ascii") as base_text, io.TextIOWrapper(
                    output_raw, encoding="ascii", newline="\n"
                ) as output_text:
                    for source, _query_time, candidates in test_chunks(args.data, args.chunk):
                        baseline_probability = np.loadtxt(
                            itertools.islice(base_text, len(source)),
                            delimiter=",",
                            dtype=np.float64,
                        )
                        if baseline_probability.shape != (len(source), WIDTH):
                            raise ValueError("base Dataset4 rows ended early")
                        raw, degree = rp3.score_context(
                            forward, reverse, source_base, source, candidates,
                            args.max_candidate_degree,
                        )
                        residual = rp3.feature(raw, degree, 1, BETA)
                        duplicate = np.any(
                            np.diff(np.sort(candidates, axis=1), axis=1) == 0,
                            axis=1,
                        )
                        residual[duplicate] = 0.0
                        seen = pair_seen(
                            source, candidates, source_base,
                            forward.indptr, forward.indices,
                        )
                        baseline_slots = rp3.strict_slots(baseline_probability)
                        candidate_slots = rp3.candidate_score(
                            baseline_slots, residual, seen, ALPHA
                        )
                        baseline_rank = np.argsort(
                            np.argsort(-baseline_slots, axis=1, kind="stable"),
                            axis=1,
                            kind="stable",
                        )
                        candidate_rank = np.argsort(
                            np.argsort(-candidate_slots, axis=1, kind="stable"),
                            axis=1,
                            kind="stable",
                        )
                        pair_seen_errors += int(np.count_nonzero(
                            baseline_rank[seen] != candidate_rank[seen]
                        ))
                        new_order = np.argsort(-candidate_slots, axis=1, kind="stable")
                        old_order = np.argsort(-baseline_probability, axis=1, kind="stable")
                        sorted_probability = np.take_along_axis(
                            baseline_probability, old_order, axis=1
                        )
                        probability = np.empty_like(baseline_probability)
                        rows = np.broadcast_to(
                            np.arange(len(source))[:, None], probability.shape
                        )
                        probability[rows, new_order] = sorted_probability
                        maximum_row_sum_error = max(
                            maximum_row_sum_error,
                            float(np.max(np.abs(probability.sum(axis=1) - 1.0))),
                        )
                        changed += int(np.count_nonzero(
                            np.argmax(baseline_probability, axis=1)
                            != np.argmax(probability, axis=1)
                        ))
                        active += int(np.count_nonzero(np.any(raw != 0, axis=(1, 2))))
                        duplicate_count += int(duplicate.sum())
                        total += len(source)
                        np.savetxt(output_text, probability, fmt="%.8f", delimiter=",")
                        if total % (args.chunk * 20) == 0:
                            print(json.dumps({
                                "rows": total,
                                "changed_top1_rate": changed / total,
                                "elapsed_seconds": time.time() - started,
                            }), flush=True)
                    if base_text.readline():
                        raise ValueError("base Dataset4 has extra rows")
        if total != ROWS["dataset4.csv"]:
            raise ValueError(f"official Dataset4 row count differs: {total}")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    with zipfile.ZipFile(args.base) as base, zipfile.ZipFile(args.output) as output:
        d3_unchanged = member_sha256(base, "dataset3.csv") == member_sha256(
            output, "dataset3.csv"
        )
        members = output.namelist()
    checks = {
        "members_exact": members == list(ROWS),
        "row_count_exact": total == ROWS["dataset4.csv"],
        "dataset3_unchanged": d3_unchanged,
        "pair_seen_ranks_unchanged": pair_seen_errors == 0,
        "row_sums_within_5e_7": maximum_row_sum_error <= 5e-7,
        "top1_change_rate_between_0_10_and_0_35": 0.10 <= changed / total <= 0.35,
    }
    report = {
        "kind": "d4_causal_rp3beta_submission_build_v1",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "base_sha256": sha256(args.base),
        "output_sha256": sha256(args.output),
        "policy": {
            "path": "source-item-source-candidate",
            "alpha": ALPHA,
            "beta": BETA,
            "max_candidate_degree": args.max_candidate_degree,
            "pair_seen_rank_frozen": True,
            "duplicate_rows_frozen": True,
        },
        "graph": {"events": events, "unique_edges": len(forward.indices), "cutoff": cutoff},
        "test": {
            "rows": total,
            "active_row_rate": active / total,
            "duplicate_row_rate": duplicate_count / total,
            "top1_changed": changed,
            "top1_changed_rate": changed / total,
            "pair_seen_rank_errors": pair_seen_errors,
            "maximum_row_sum_error": maximum_row_sum_error,
        },
        "checks": checks,
        "elapsed_seconds": time.time() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["decision"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
