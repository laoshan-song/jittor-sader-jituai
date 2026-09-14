#!/usr/bin/env python3
"""Strict D4 gate for causal three-hop RP3beta collaborative propagation."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads


STRATEGIES = ("history", "test_pool")


def load(root: Path, strategy: str, split: str, name: str) -> np.ndarray:
    return np.load(root / f"{strategy}__{split}__{name}.npy", mmap_mode="r")


def reciprocal_ranks(score: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positive = score[np.arange(len(labels)), labels]
    columns = np.arange(score.shape[1])[None, :]
    rank = 1 + (score > positive[:, None]).sum(axis=1)
    rank += ((score == positive[:, None]) & (columns < labels[:, None])).sum(axis=1)
    return 1.0 / rank


def strict_slots(score: np.ndarray) -> np.ndarray:
    order = np.argsort(-score, axis=1, kind="stable")
    rows = np.broadcast_to(np.arange(len(score))[:, None], order.shape)
    values = np.linspace(1.0, 0.0, score.shape[1], dtype=np.float32)
    output = np.empty_like(score, dtype=np.float32)
    output[rows, order] = values
    return output


def candidate_score(
    baseline: np.ndarray, residual: np.ndarray, seen: np.ndarray, alpha: float
) -> np.ndarray:
    slots = strict_slots(baseline)
    pair_new = ~seen
    key = baseline + np.float32(alpha) * residual
    order = np.argsort(np.where(pair_new, -key, np.inf), axis=1, kind="stable")
    destinations = np.sort(np.where(pair_new, slots, -np.inf), axis=1)[:, ::-1]
    active = np.arange(score_width := baseline.shape[1])[None, :] < pair_new.sum(axis=1)[:, None]
    rows = np.broadcast_to(np.arange(len(baseline))[:, None], (len(baseline), score_width))
    output = slots.copy()
    output[rows[active], order[active]] = destinations[active]
    return output


def qnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / (values.std(axis=1, keepdims=True) + 1e-6)


def paired(before: np.ndarray, after: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    delta = reciprocal_ranks(after, labels) - reciprocal_ranks(before, labels)
    return {
        "delta": float(delta.mean()),
        "delta_se": float(delta.std(ddof=1) / math.sqrt(len(delta))),
        "positive_row_rate": float(np.mean(delta > 0)),
        "negative_row_rate": float(np.mean(delta < 0)),
        "top1_changed_rate": float(np.mean(np.argmax(before, axis=1) != np.argmax(after, axis=1))),
    }


@dataclass
class CSR:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    degree: np.ndarray


def graph(
    raw_source: np.ndarray, raw_item: np.ndarray, raw_time: np.ndarray, cutoff: int
) -> tuple[CSR, CSR, int, int]:
    stop = int(np.searchsorted(raw_time, cutoff, side="left"))
    source = np.asarray(raw_source[:stop], dtype=np.int64)
    item = np.asarray(raw_item[:stop], dtype=np.int64)
    source_base = int(source.min())
    source -= source_base
    order = np.lexsort((item, source))
    sorted_source = source[order]
    sorted_item = item[order]
    unique = np.empty(stop, dtype=bool)
    unique[0] = True
    unique[1:] = (sorted_source[1:] != sorted_source[:-1]) | (
        sorted_item[1:] != sorted_item[:-1]
    )
    starts = np.flatnonzero(unique)
    frequency = np.diff(np.append(starts, stop)).astype(np.float32)
    source_pair = sorted_source[starts]
    item_pair = sorted_item[starts]
    del order, sorted_source, sorted_item, source, item, unique, starts

    source_rows = int(source_pair.max()) + 1
    item_rows = int(item_pair.max()) + 1
    source_indptr = np.zeros(source_rows + 1, dtype=np.int64)
    source_indptr[1:] = np.bincount(source_pair, minlength=source_rows)
    np.cumsum(source_indptr, out=source_indptr)
    source_degree = np.bincount(
        source_pair, weights=frequency, minlength=source_rows
    ).astype(np.float32)
    forward = CSR(
        source_indptr,
        item_pair.astype(np.int32, copy=False),
        frequency,
        source_degree,
    )

    reverse_order = np.lexsort((source_pair, item_pair))
    reverse_item = item_pair[reverse_order]
    item_indptr = np.zeros(item_rows + 1, dtype=np.int64)
    item_indptr[1:] = np.bincount(reverse_item, minlength=item_rows)
    np.cumsum(item_indptr, out=item_indptr)
    item_degree = np.bincount(
        item_pair, weights=frequency, minlength=item_rows
    ).astype(np.float32)
    reverse = CSR(
        item_indptr,
        source_pair[reverse_order].astype(np.int32, copy=False),
        frequency[reverse_order],
        item_degree,
    )
    return forward, reverse, source_base, stop


@njit(parallel=True, cache=True)
def three_hop(
    source: np.ndarray,
    candidates: np.ndarray,
    source_base: int,
    source_indptr: np.ndarray,
    source_indices: np.ndarray,
    source_data: np.ndarray,
    item_indptr: np.ndarray,
    item_indices: np.ndarray,
    item_data: np.ndarray,
    source_degree: np.ndarray,
    item_degree: np.ndarray,
    max_candidate_degree: float,
) -> np.ndarray:
    rows, width = candidates.shape
    output = np.zeros((rows, width, 2), dtype=np.float32)
    for row in prange(rows):
        query = int(source[row]) - source_base
        if query < 0 or query >= len(source_degree) or source_degree[query] <= 0:
            continue
        query_start = source_indptr[query]
        query_stop = source_indptr[query + 1]
        for column in range(width):
            candidate = int(candidates[row, column])
            if candidate < 0 or candidate >= len(item_degree):
                continue
            candidate_degree = item_degree[candidate]
            if candidate_degree <= 0 or candidate_degree > max_candidate_degree:
                continue
            count_score = 0.0
            walk_score = 0.0
            for edge in range(item_indptr[candidate], item_indptr[candidate + 1]):
                neighbour = item_indices[edge]
                if neighbour == query:
                    continue
                candidate_frequency = item_data[edge]
                left = query_start
                right = source_indptr[neighbour]
                right_stop = source_indptr[neighbour + 1]
                neighbour_degree = source_degree[neighbour]
                while left < query_stop and right < right_stop:
                    query_item = source_indices[left]
                    neighbour_item = source_indices[right]
                    if query_item < neighbour_item:
                        left += 1
                    elif query_item > neighbour_item:
                        right += 1
                    else:
                        frequency = source_data[left] * source_data[right] * candidate_frequency
                        count_score += frequency
                        walk_score += frequency / (item_degree[query_item] * neighbour_degree)
                        left += 1
                        right += 1
            output[row, column, 0] = math.log1p(count_score)
            output[row, column, 1] = walk_score
    return output


def score_context(
    matrix: CSR,
    reverse: CSR,
    source_base: int,
    source: np.ndarray,
    candidates: np.ndarray,
    max_candidate_degree: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw = three_hop(
        np.asarray(source, dtype=np.uint32),
        np.asarray(candidates, dtype=np.uint32),
        source_base,
        matrix.indptr,
        matrix.indices,
        matrix.data,
        reverse.indptr,
        reverse.indices,
        reverse.data,
        matrix.degree,
        reverse.degree,
        max_candidate_degree,
    )
    candidate_degree = np.zeros(candidates.shape, dtype=np.float32)
    known = candidates < len(reverse.degree)
    candidate_degree[known] = reverse.degree[np.asarray(candidates[known], dtype=np.int64)]
    return raw, candidate_degree


def feature(raw: np.ndarray, degree: np.ndarray, kind: int, beta: float) -> np.ndarray:
    denominator = np.maximum(degree, 1.0) ** np.float32(beta)
    return qnorm(raw[:, :, kind] / denominator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--identity-cache", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-cutoff", type=int, default=1512131937)
    parser.add_argument("--confirmation-cutoff", type=int, default=1512135046)
    parser.add_argument("--selection-rows", type=int, default=5000)
    parser.add_argument("--holdout-rows", type=int, default=5000)
    parser.add_argument("--confirmation-rows", type=int, default=10000)
    parser.add_argument("--max-candidate-degree", type=float, default=4096.0)
    parser.add_argument("--threads", type=int, default=48)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    set_num_threads(args.threads)

    raw_source = np.load(args.data_cache / "src.npy", mmap_mode="r")
    raw_item = np.load(args.data_cache / "dst.npy", mmap_mode="r")
    raw_time = np.load(args.data_cache / "time.npy", mmap_mode="r")
    prepared: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    limits = {
        "validation": args.selection_rows + args.holdout_rows,
        "confirmation": args.confirmation_rows,
    }
    for split, cutoff in (
        ("validation", args.validation_cutoff),
        ("confirmation", args.confirmation_cutoff),
    ):
        matrix, reverse, source_base, events = graph(raw_source, raw_item, raw_time, cutoff)
        print(json.dumps({
            "graph": split,
            "events": events,
            "unique_edges": len(matrix.indices),
            "shape": [len(matrix.degree), len(reverse.degree)],
        }), flush=True)
        for strategy in STRATEGIES:
            limit = limits[split]
            source = load(args.identity_cache, strategy, split, "src")[:limit]
            candidates = load(args.identity_cache, strategy, split, "candidates")[:limit]
            raw, degree = score_context(
                matrix, reverse, source_base, source, candidates, args.max_candidate_degree
            )
            labels = load(args.replay_root / strategy, strategy, split, "labels")[:limit]
            seen = load(args.replay_root / strategy, strategy, split, "seen")[:limit]
            baseline = np.load(
                args.baseline_cache / f"{strategy}__{split}.npy", mmap_mode="r"
            )[:limit]
            prepared[(strategy, split)] = {
                "raw": raw,
                "degree": degree,
                "labels": np.asarray(labels),
                "seen": np.asarray(seen),
                "baseline": np.asarray(baseline),
                "duplicate": np.any(
                    np.diff(np.sort(candidates, axis=1), axis=1) == 0, axis=1
                ),
            }
            print(json.dumps({
                "scored": [strategy, split],
                "rows": limit,
                "nonzero_row_rate": float(np.mean(np.any(raw != 0, axis=(1, 2)))),
                "nonzero_cell_rate": float(np.mean(np.any(raw != 0, axis=2))),
            }), flush=True)
        del matrix, reverse
        gc.collect()

    selection = []
    part = slice(0, args.selection_rows)
    for kind, name in ((0, "path_count"), (1, "rp3")):
        for beta in (0.0, 0.25, 0.5, 1.0):
            for alpha in (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16):
                deltas = []
                for strategy in STRATEGIES:
                    values = prepared[(strategy, "validation")]
                    residual = feature(values["raw"][part], values["degree"][part], kind, beta)
                    residual[values["duplicate"][part]] = 0.0
                    candidate = candidate_score(
                        values["baseline"][part], residual, values["seen"][part], alpha
                    )
                    deltas.append(
                        reciprocal_ranks(candidate, values["labels"][part])
                        - reciprocal_ranks(values["baseline"][part], values["labels"][part])
                    )
                joined = np.concatenate(deltas)
                selection.append({
                    "kind": name,
                    "kind_index": kind,
                    "beta": beta,
                    "alpha": alpha,
                    "delta": float(joined.mean()),
                    "delta_se": float(joined.std(ddof=1) / math.sqrt(len(joined))),
                })
    selected = max(selection, key=lambda value: (value["delta"], -value["alpha"]))

    evaluation: dict[str, dict[str, dict[str, float]]] = {}
    for strategy in STRATEGIES:
        for output_split, cache_split, rows in (
            ("holdout", "validation", slice(args.selection_rows, args.selection_rows + args.holdout_rows)),
            ("confirmation", "confirmation", slice(0, args.confirmation_rows)),
        ):
            values = prepared[(strategy, cache_split)]
            residual = feature(
                values["raw"][rows], values["degree"][rows],
                int(selected["kind_index"]), float(selected["beta"]),
            )
            residual[values["duplicate"][rows]] = 0.0
            candidate = candidate_score(
                values["baseline"][rows], residual, values["seen"][rows],
                float(selected["alpha"]),
            )
            evaluation.setdefault(strategy, {})[output_split] = paired(
                values["baseline"][rows], candidate, values["labels"][rows]
            )

    checks = {
        "selection_delta_at_least_0_01": selected["delta"] >= 0.01,
        "both_holdouts_positive": all(evaluation[name]["holdout"]["delta"] > 0 for name in STRATEGIES),
        "both_confirmations_at_least_0_01": all(evaluation[name]["confirmation"]["delta"] >= 0.01 for name in STRATEGIES),
        "both_confirmations_above_two_se": all(
            evaluation[name]["confirmation"]["delta"] > 2 * evaluation[name]["confirmation"]["delta_se"]
            for name in STRATEGIES
        ),
    }
    report = {
        "kind": "d4_causal_rp3beta_strict_gate_v1",
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "selected": selected,
        "top_selection": sorted(selection, key=lambda value: value["delta"], reverse=True)[:12],
        "evaluation": evaluation,
        "checks": checks,
        "confirmation_excluded_from_selection": True,
        "data_contract": "three-hop graph contains only official train edges before each fixed split cutoff",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["decision"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
