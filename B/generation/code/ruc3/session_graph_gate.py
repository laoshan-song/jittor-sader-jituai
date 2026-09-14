#!/usr/bin/env python3
"""Strict D4 multi-window session co-visitation hard-negative gate."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np

from b_rank import pairnew_transformer_jittor as pairnew, replay_score_cache


POLICY_ALPHAS = (0.005, 0.01, 0.025, 0.05, 0.10, 0.20)
TOP_NS = (20, 50, 100)
HISTORY_SIZE = 8
GRAPH_LAGS = 4
GRAPH_WINDOW = 1800


def identity(root: Path, strategy: str, split: str, name: str) -> np.ndarray:
    return np.load(root / f"{strategy}__{split}__{name}.npy", mmap_mode="r")


def rank_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    rows = np.arange(len(scores))[:, None]
    output = np.empty_like(order, dtype=np.int16)
    output[rows, order] = np.arange(scores.shape[1], dtype=np.int16)
    return output


def paired(before: np.ndarray, after: np.ndarray, labels: np.ndarray, active: np.ndarray) -> dict:
    left = pairnew._reciprocal_ranks(before, labels)
    right = pairnew._reciprocal_ranks(after, labels)
    delta = right - left
    return {
        "delta": float(delta.mean()),
        "delta_se": float(delta.std(ddof=1) / math.sqrt(len(delta))),
        "positive_row_rate": float(np.mean(delta > 0)),
        "negative_row_rate": float(np.mean(delta < 0)),
        "top1_changed_rate": float(np.mean(np.argmax(before, axis=1) != np.argmax(after, axis=1))),
        "active_row_rate": float(active.mean()),
    }


def build_graph(
    source_all: np.ndarray,
    item_all: np.ndarray,
    time_all: np.ndarray,
    cutoff: int,
) -> dict[str, np.ndarray]:
    started = time.monotonic()
    stop = int(np.searchsorted(time_all, cutoff, side="left"))
    source = np.asarray(source_all[:stop], dtype=np.uint32)
    item = np.asarray(item_all[:stop], dtype=np.uint32)
    timestamp = np.asarray(time_all[:stop], dtype=np.uint32)
    order = np.argsort(source, kind="stable")
    source = source[order]
    item = item[order]
    timestamp = timestamp[order]
    parts = []
    for lag in range(1, GRAPH_LAGS + 1):
        same = source[lag:] == source[:-lag]
        gap = timestamp[lag:].astype(np.int64) - timestamp[:-lag].astype(np.int64)
        keep = same & (gap >= 0) & (gap <= GRAPH_WINDOW)
        left = item[:-lag][keep].astype(np.uint64)
        right = item[lag:][keep].astype(np.uint64)
        parts.append((left << np.uint64(32)) | right)
        parts.append((right << np.uint64(32)) | left)
    raw = np.concatenate(parts)
    del parts
    keys, counts = np.unique(raw, return_counts=True)
    del raw
    maximum = int(item.max(initial=0))
    frequency = np.bincount(item, minlength=maximum + 1).astype(np.uint32)
    starts = np.r_[0, np.flatnonzero(source[1:] != source[:-1]) + 1]
    source_ids = source[starts].copy()
    ends = np.r_[starts[1:], len(source)]
    print(json.dumps({
        "graph_cutoff": cutoff,
        "history_edges": stop,
        "directed_pairs": int(counts.sum()),
        "unique_pairs": len(keys),
        "seconds": time.monotonic() - started,
    }), flush=True)
    return {
        "keys": keys,
        "counts": counts.astype(np.uint32),
        "frequency": frequency,
        "source": source,
        "item": item,
        "time": timestamp,
        "source_ids": source_ids,
        "starts": starts.astype(np.int64),
        "ends": ends.astype(np.int64),
    }


def recent_history(graph: dict[str, np.ndarray], source: np.ndarray, timestamp: np.ndarray) -> np.ndarray:
    output = np.zeros((len(source), HISTORY_SIZE), dtype=np.uint32)
    positions = np.searchsorted(graph["source_ids"], source)
    inside = positions < len(graph["source_ids"])
    matched = np.zeros(len(source), dtype=bool)
    matched[inside] = graph["source_ids"][positions[inside]] == source[inside]
    for row in np.flatnonzero(matched):
        group = int(positions[row])
        start, end = int(graph["starts"][group]), int(graph["ends"][group])
        stop = start + int(np.searchsorted(graph["time"][start:end], timestamp[row], side="left"))
        take = min(HISTORY_SIZE, stop - start)
        if take:
            output[row, -take:] = graph["item"][stop - take:stop]
    return output


def lookup_counts(keys: np.ndarray, graph_keys: np.ndarray, counts: np.ndarray) -> np.ndarray:
    flat = keys.reshape(-1)
    positions = np.searchsorted(graph_keys, flat)
    inside = positions < len(graph_keys)
    found = np.zeros(len(flat), dtype=bool)
    found[inside] = graph_keys[positions[inside]] == flat[inside]
    output = np.zeros(len(flat), dtype=np.float32)
    output[found] = counts[positions[found]].astype(np.float32)
    return output.reshape(keys.shape)


def graph_features(
    graph: dict[str, np.ndarray],
    source: np.ndarray,
    timestamp: np.ndarray,
    candidates: np.ndarray,
    chunk: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    history = recent_history(graph, source, timestamp)
    raw_sum = np.zeros(candidates.shape, dtype=np.float32)
    raw_max = np.zeros(candidates.shape, dtype=np.float32)
    norm_sum = np.zeros(candidates.shape, dtype=np.float32)
    norm_max = np.zeros(candidates.shape, dtype=np.float32)
    frequency = graph["frequency"]
    for start in range(0, len(source), chunk):
        stop = min(len(source), start + chunk)
        candidate = np.asarray(candidates[start:stop], dtype=np.uint32)
        candidate_frequency = np.zeros(candidate.shape, dtype=np.float32)
        known_candidate = candidate < len(frequency)
        candidate_frequency[known_candidate] = frequency[candidate[known_candidate]]
        for index in range(HISTORY_SIZE):
            previous = history[start:stop, index]
            valid = previous > 0
            if not np.any(valid):
                continue
            query = (previous[:, None].astype(np.uint64) << np.uint64(32)) | candidate.astype(np.uint64)
            count = lookup_counts(query, graph["keys"], graph["counts"])
            decay = np.float32(0.72 ** (HISTORY_SIZE - index - 1))
            raw = np.log1p(count)
            previous_frequency = np.zeros(len(previous), dtype=np.float32)
            known_previous = previous < len(frequency)
            previous_frequency[known_previous] = frequency[previous[known_previous]]
            cosine = count / np.sqrt(
                np.maximum(previous_frequency[:, None], 1.0)
                * np.maximum(candidate_frequency, 1.0)
            )
            normalized = np.log1p(10_000.0 * cosine)
            raw_sum[start:stop] += decay * raw
            raw_max[start:stop] = np.maximum(raw_max[start:stop], raw)
            norm_sum[start:stop] += decay * normalized
            norm_max[start:stop] = np.maximum(norm_max[start:stop], normalized)
    active = raw_max.max(axis=1) > 0
    features = {
        "raw_sum": pairnew._qnorm(raw_sum),
        "raw_max": pairnew._qnorm(raw_max),
        "norm_sum": pairnew._qnorm(norm_sum),
        "norm_max": pairnew._qnorm(norm_max),
        "hit": (raw_max > 0.0).astype(np.float32),
    }
    features["hybrid"] = pairnew._qnorm(features["raw_sum"] + features["norm_max"])
    return features, active


def choose_policy(prepared: dict, rows: slice) -> tuple[dict, list[dict]]:
    trace = []
    for feature_name in prepared["history"]["features"]:
        for top_n in TOP_NS:
            for alpha in POLICY_ALPHAS:
                metrics = []
                for strategy in ("history", "test_pool"):
                    values = prepared[strategy]
                    baseline = values["baseline"][rows]
                    rank = values["rank"][rows]
                    residual = values["features"][feature_name][rows].copy()
                    residual[rank >= top_n] = 0.0
                    candidate = pairnew._candidate_score(
                        baseline, residual, values["seen"][rows], alpha
                    )
                    metrics.append(paired(
                        baseline, candidate, values["labels"][rows], values["active"][rows]
                    ))
                delta = float(np.mean([value["delta"] for value in metrics]))
                trace.append({
                    "feature": feature_name,
                    "top_n": top_n,
                    "alpha": alpha,
                    "delta": delta,
                    "history_delta": metrics[0]["delta"],
                    "test_pool_delta": metrics[1]["delta"],
                    "max_negative_rate": max(value["negative_row_rate"] for value in metrics),
                })
    selected = max(trace, key=lambda value: (value["delta"], -value["max_negative_rate"]))
    return selected, trace


def evaluate(prepared: dict, policy: dict, rows: slice) -> dict:
    output = {}
    for strategy, values in prepared.items():
        baseline = values["baseline"][rows]
        rank = values["rank"][rows]
        residual = values["features"][policy["feature"]][rows].copy()
        residual[rank >= int(policy["top_n"])] = 0.0
        candidate = pairnew._candidate_score(
            baseline, residual, values["seen"][rows], float(policy["alpha"])
        )
        output[strategy] = paired(
            baseline, candidate, values["labels"][rows], values["active"][rows]
        )
    return output


def prepare_split(
    scored: dict,
    identity_root: Path,
    baseline_root: Path,
    graph: dict[str, np.ndarray],
    split: str,
    rows: int,
    chunk: int,
) -> dict:
    output = {}
    for strategy in ("history", "test_pool"):
        _, labels, seen, _, _ = scored[(strategy, split)]
        source = identity(identity_root, strategy, split, "src")[:rows]
        timestamp = identity(identity_root, strategy, split, "time")[:rows]
        candidates = identity(identity_root, strategy, split, "candidates")[:rows]
        baseline = np.load(
            baseline_root / f"{strategy}__{split}.npy", mmap_mode="r"
        )[:rows]
        features, active = graph_features(
            graph, source, timestamp, candidates, chunk
        )
        for values in features.values():
            values[seen[:rows]] = 0.0
        output[strategy] = {
            "labels": labels[:rows],
            "seen": seen[:rows],
            "baseline": baseline,
            "rank": rank_positions(baseline),
            "features": features,
            "active": active,
        }
        print(json.dumps({
            "prepared": [strategy, split],
            "rows": rows,
            "active_row_rate": float(active.mean()),
        }), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--replay-cache", type=Path, action="append", required=True)
    parser.add_argument("--identity-cache", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=30_000)
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args()
    scored, _, manifests = replay_score_cache.load(args.replay_cache, verify=False)
    source_all = np.load(args.train_cache / "src.npy", mmap_mode="r")
    item_all = np.load(args.train_cache / "dst.npy", mmap_mode="r")
    time_all = np.load(args.train_cache / "time.npy", mmap_mode="r")
    plan = manifests[0]["metadata"]["group_metadata"]["history"]["plan"]

    validation_graph = build_graph(
        source_all, item_all, time_all, int(plan["cutoffs"]["valid"])
    )
    validation = prepare_split(
        scored, args.identity_cache, args.baseline_cache, validation_graph,
        "validation", int(args.rows), int(args.chunk)
    )
    selected, trace = choose_policy(validation, slice(0, 5_000))
    smoke = evaluate(validation, selected, slice(5_000, 10_000))
    smoke_pass = all(
        value["delta"] >= 0.003 and value["delta"] > 2.0 * value["delta_se"]
        for value in smoke.values()
    )
    report = {
        "kind": "d4_multiwindow_session_graph_hard_negative_gate_v1",
        "decision": "SMOKE_PASS" if smoke_pass else "NO_GO_SMOKE",
        "selected": selected,
        "smoke": smoke,
        "selection_trace": trace,
        "graph": {"lags": GRAPH_LAGS, "window_seconds": GRAPH_WINDOW, "history_size": HISTORY_SIZE},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not smoke_pass:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 3

    holdout = evaluate(validation, selected, slice(5_000, int(args.rows)))
    del validation, validation_graph
    gc.collect()
    confirmation_graph = build_graph(
        source_all, item_all, time_all, int(plan["cutoffs"]["confirm"])
    )
    confirmation_values = prepare_split(
        scored, args.identity_cache, args.baseline_cache, confirmation_graph,
        "confirmation", int(args.rows), int(args.chunk)
    )
    confirmation = evaluate(confirmation_values, selected, slice(None))
    checks = {
        "both_holdouts_at_least_0_01": all(value["delta"] >= 0.01 for value in holdout.values()),
        "both_confirmations_at_least_0_01": all(value["delta"] >= 0.01 for value in confirmation.values()),
        "both_confirmations_above_two_se": all(
            value["delta"] > 2.0 * value["delta_se"] for value in confirmation.values()
        ),
    }
    report.update(
        decision="PASS" if all(checks.values()) else "NO_GO_30K",
        holdout=holdout,
        confirmation=confirmation,
        checks=checks,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
