#!/usr/bin/env python3
"""Audit D3 cross-source support just outside v26's exact timestamp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import d3_cross_source_c2_audit as d3
from b_rank_a_port import ensemble_core as core


WINDOWS = (1, 5, 30)
MIN_LABEL_RATE = 0.005
MIN_ENRICHMENT = 20.0


def distinct_occurrences(candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(candidates, axis=1, kind="stable")
    distinct = np.empty(ordered.shape, dtype=bool)
    distinct[:, 0] = True
    distinct[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    rows = np.repeat(
        np.arange(len(ordered), dtype=np.int32),
        distinct.sum(axis=1, dtype=np.int32),
    )
    return rows, ordered[distinct].astype(np.uint32, copy=False)


class NearTimeIndex:
    def __init__(
        self, source: np.ndarray, time: np.ndarray, candidates: np.ndarray
    ) -> None:
        rows, items = distinct_occurrences(candidates)
        occurrence_time = np.asarray(time[rows], dtype=np.uint64)
        occurrence_source = np.asarray(source[rows], dtype=np.uint64)
        self.total_keys = np.sort(
            (items.astype(np.uint64) << np.uint64(32)) | occurrence_time,
            kind="stable",
        )
        source_item = (
            occurrence_source << np.uint64(32)
        ) | items.astype(np.uint64)
        self.source_item_ids, inverse = np.unique(source_item, return_inverse=True)
        self.own_keys = np.sort(
            (inverse.astype(np.uint64) << np.uint64(32)) | occurrence_time,
            kind="stable",
        )

    def source_support(
        self,
        source: np.ndarray,
        time: np.ndarray,
        candidates: np.ndarray,
        window: int,
    ) -> np.ndarray:
        query_time = np.asarray(time, dtype=np.uint64)[:, None]
        source_item = (
            np.asarray(source, dtype=np.uint64)[:, None] << np.uint64(32)
        ) | np.asarray(candidates, dtype=np.uint64)
        group = np.searchsorted(self.source_item_ids, source_item)
        known = group < len(self.source_item_ids)
        known[known] &= self.source_item_ids[group[known]] == source_item[known]
        group[~known] = len(self.source_item_ids)
        own_base = group.astype(np.uint64) << np.uint64(32)
        lower_time = np.maximum(
            query_time.astype(np.int64) - int(window), 0
        ).astype(np.uint64)
        upper_time = query_time + np.uint64(window)
        return (
            np.searchsorted(self.own_keys, own_base | upper_time, side="right")
            - np.searchsorted(self.own_keys, own_base | lower_time, side="left")
        ).astype(np.int32, copy=False)

    def support(
        self,
        source: np.ndarray,
        time: np.ndarray,
        candidates: np.ndarray,
        window: int,
    ) -> np.ndarray:
        query_time = np.asarray(time, dtype=np.uint64)[:, None]
        item_base = np.asarray(candidates, dtype=np.uint64) << np.uint64(32)
        lower_time = np.maximum(
            query_time.astype(np.int64) - int(window), 0
        ).astype(np.uint64)
        upper_time = query_time + np.uint64(window)
        total = np.searchsorted(
            self.total_keys, item_base | upper_time, side="right"
        ) - np.searchsorted(self.total_keys, item_base | lower_time, side="left")

        source_item = (
            np.asarray(source, dtype=np.uint64)[:, None] << np.uint64(32)
        ) | np.asarray(candidates, dtype=np.uint64)
        group = np.searchsorted(self.source_item_ids, source_item)
        known = group < len(self.source_item_ids)
        known[known] &= self.source_item_ids[group[known]] == source_item[known]
        group[~known] = len(self.source_item_ids)
        own_base = group.astype(np.uint64) << np.uint64(32)
        own = np.searchsorted(
            self.own_keys, own_base | upper_time, side="right"
        ) - np.searchsorted(self.own_keys, own_base | lower_time, side="left")
        result = (total - own).astype(np.int32, copy=False)
        if result.min(initial=0) < 0:
            raise ValueError("cross-source support became negative")
        return result


def audit(values: np.ndarray, labels: np.ndarray) -> dict:
    positive = values[np.arange(len(labels)), labels]
    cell_rate = float(np.mean(values > 0))
    label_rate = float(np.mean(positive > 0))
    return {
        "candidate_cell_rate": cell_rate,
        "label_rate": label_rate,
        "label_enrichment": label_rate
        / max(cell_rate, np.finfo(np.float64).eps),
        "positive_label_rows": int(np.sum(positive > 0)),
        "positive_query_rows": int(np.sum(np.any(values > 0, axis=1))),
        "mean_positive_support": float(positive[positive > 0].mean())
        if np.any(positive > 0)
        else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--groups", type=int, default=30_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if d3.sha256(args.data) != d3.DATA_SHA256:
        raise ValueError("official data hash differs")
    (
        _train,
        _test,
        _max_node,
        _use_src_freq,
        pool,
        _freq,
        _src_freq,
        _initial_history,
        segments,
    ) = core.scene_data(args.data, "dataset3")
    results = {}
    for ordinal, name in enumerate(("validation", "confirmation"), start=1):
        rows = min(int(args.groups), len(segments[name]))
        seed = int(args.seed) + ordinal
        src, time, candidates, labels = core.sample_segment(
            segments, pool, name, rows, seed
        )
        pool_src, pool_time, pool_candidates, _ = core.sample_segment(
            segments, pool, name, len(segments[name]), seed + 10_000
        )
        index = NearTimeIndex(pool_src, pool_time, pool_candidates)
        exact = index.support(src, time, candidates, 0)
        reference = d3.cross_source_support(
            pool_src, pool_time, pool_candidates, src, time, candidates
        )
        if not np.array_equal(exact, reference):
            raise ValueError("window index does not reproduce v26 exact support")
        results[name] = {"rows": rows, "exact": audit(exact, labels), "windows": {}}
        for window in WINDOWS:
            support = index.support(src, time, candidates, window)
            incremental = support - exact
            if incremental.min(initial=0) < 0:
                raise ValueError("window support is below exact support")
            results[name]["windows"][str(window)] = audit(incremental, labels)
        print(json.dumps({name: results[name]}, sort_keys=True), flush=True)

    passing = []
    for window in WINDOWS:
        key = str(window)
        if all(
            results[split]["windows"][key]["label_rate"] >= MIN_LABEL_RATE
            and results[split]["windows"][key]["label_enrichment"]
            >= MIN_ENRICHMENT
            for split in ("validation", "confirmation")
        ):
            passing.append(window)
    output = {
        "kind": "d3_near_time_incremental_support_audit_v1",
        "decision": "PROMISING" if passing else "NO_GO",
        "criteria": {
            "minimum_label_rate": MIN_LABEL_RATE,
            "minimum_label_enrichment": MIN_ENRICHMENT,
            "must_pass_validation_and_confirmation": True,
        },
        "passing_windows_seconds": passing,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return 0 if passing else 3


if __name__ == "__main__":
    raise SystemExit(main())
