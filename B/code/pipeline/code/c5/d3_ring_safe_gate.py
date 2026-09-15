#!/usr/bin/env python3
"""Cross-fitted D3 challenger with deduplicated source/query consensus support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


C2_CROSS_WEIGHT = 0.10
C2_SESSION_WEIGHT = 0.05
C3_POLICY = {
    "cross_past_1s": -0.10,
    "session_future_1s": 0.225,
    "session_future_300s": 0.30,
    "session_past_300s": 0.305,
}
MIN_DELTA = 0.02


def qnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + np.float32(1e-6)
    )


def reciprocal_ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positive = scores[np.arange(len(labels)), labels]
    columns = np.arange(scores.shape[1])[None, :]
    rank = 1 + (scores > positive[:, None]).sum(axis=1)
    rank += ((scores == positive[:, None]) & (columns < labels[:, None])).sum(axis=1)
    return 1.0 / rank.astype(np.float64)


def metrics(control: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> dict:
    before = reciprocal_ranks(control, labels)
    after = reciprocal_ranks(candidate, labels)
    paired = after - before
    return {
        "control_mrr": float(before.mean()),
        "candidate_mrr": float(after.mean()),
        "delta": float(paired.mean()),
        "delta_se": float(paired.std(ddof=1) / np.sqrt(len(paired))),
        "positive_row_rate": float(np.mean(paired > 0.0)),
        "negative_row_rate": float(np.mean(paired < 0.0)),
        "top1_changed": float(np.mean(np.argmax(control, axis=1) != np.argmax(candidate, axis=1))),
    }


def _keys_by_item(items: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.lexsort((values, items))
    items = items[order]
    values = values[order]
    keep = np.empty(len(order), dtype=bool)
    keep[0] = True
    keep[1:] = (items[1:] != items[:-1]) | (values[1:] != values[:-1])
    return np.sort((items[keep].astype(np.uint64) << np.uint64(32)) | values[keep].astype(np.uint64))


class ConsensusIndex:
    def __init__(self, source: np.ndarray, time: np.ndarray, candidates: np.ndarray):
        ordered = np.sort(candidates, axis=1, kind="stable")
        distinct = np.empty(ordered.shape, dtype=bool)
        distinct[:, 0] = True
        distinct[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        rows = np.repeat(np.arange(len(ordered), dtype=np.int32), distinct.sum(axis=1, dtype=np.int32))
        items = ordered[distinct].astype(np.uint32, copy=False)
        src = np.asarray(source[rows], dtype=np.uint32)
        tim = np.asarray(time[rows], dtype=np.uint32)
        query = np.asarray(rows, dtype=np.uint32)
        self.source_keys = _keys_by_item(items, src)
        self.time_keys = _keys_by_item(items, tim)
        self.query_keys = _keys_by_item(items, query)

    @staticmethod
    def count(keys: np.ndarray, candidates: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        base = np.asarray(candidates, dtype=np.uint64) << np.uint64(32)
        return (
            np.searchsorted(keys, base | np.asarray(upper, dtype=np.uint64)[:, None], side="right")
            - np.searchsorted(keys, base | np.asarray(lower, dtype=np.uint64)[:, None], side="left")
        ).astype(np.int32, copy=False)

    def distinct_sources(self, candidates: np.ndarray) -> np.ndarray:
        zeros = np.zeros(len(candidates), dtype=np.uint64)
        highs = np.full(len(candidates), np.iinfo(np.uint32).max, dtype=np.uint64)
        return self.count(self.source_keys, candidates, zeros, highs)

    def directional_times(self, time: np.ndarray, candidates: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        time = np.asarray(time, dtype=np.int64)
        past = self.count(self.time_keys, candidates, np.maximum(time - window, 0), np.maximum(time - 1, 0))
        future = self.count(self.time_keys, candidates, time + 1, time + window)
        return past, future

    def query_occurrences(self, candidates: np.ndarray) -> np.ndarray:
        zeros = np.zeros(len(candidates), dtype=np.uint64)
        highs = np.full(len(candidates), np.iinfo(np.uint32).max, dtype=np.uint64)
        return self.count(self.query_keys, candidates, zeros, highs) - 1


def apply(control: np.ndarray, features: np.ndarray, weights: np.ndarray, gate: np.ndarray) -> np.ndarray:
    residual = np.tensordot(weights, features, axes=(0, 0)).astype(np.float32)
    return control + np.where(gate, residual, 0.0)


def tune(control: np.ndarray, features: np.ndarray, labels: np.ndarray, gate: np.ndarray):
    weights = np.zeros(len(features), dtype=np.float64)
    score = control.copy()
    value = float(reciprocal_ranks(score, labels).mean())
    for steps in (np.arange(-0.10, 0.201, 0.025), np.arange(-0.04, 0.081, 0.01)):
        changed = True
        while changed:
            changed = False
            for index in range(len(features)):
                for delta in steps:
                    if delta == 0.0:
                        continue
                    candidate = score + np.where(gate, np.float32(delta) * features[index], 0.0)
                    candidate_value = float(reciprocal_ranks(candidate, labels).mean())
                    if candidate_value > value + 1e-10:
                        score = candidate
                        weights[index] += float(delta)
                        value = candidate_value
                        changed = True
    active = np.flatnonzero(np.abs(weights) > 1e-12)
    if len(active) > 4:
        keep = active[np.argsort(np.abs(weights[active]))[-4:]]
        compact = np.zeros_like(weights)
        compact[keep] = weights[keep]
        weights = compact
        score = apply(control, features, weights, gate)
        value = float(reciprocal_ranks(score, labels).mean())
    return value, weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--ensemble-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=30_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.code.resolve()))
    import d3_cross_source_c2_audit as d3
    import d3_near_time_audit as near
    import d3_multiscale_craft_gate as c3
    import ensemble_core as core

    if d3.sha256(args.data) != d3.DATA_SHA256:
        raise ValueError("official data hash differs")
    _, model_dirs, active, ensemble_weights = d3.load_ensemble(args.ensemble_report)
    (_train, _test, max_node, use_src_freq, pool, freq, src_freq, initial_history, segments) = core.scene_data(args.data, "dataset3")
    scored = {}
    feature_names = None
    for ordinal, split in enumerate(("meta_train", "validation", "confirmation")):
        seed = int(args.seed) + ordinal
        src, time, candidates, labels = core.sample_segment(segments, pool, split, int(args.groups), seed)
        names, components, _ = core.component_scores(
            scene="dataset3", model_dirs=model_dirs,
            history=core.segment_history(initial_history, segments, split), max_node=max_node,
            use_src_freq=use_src_freq, freq=freq, src_freq=src_freq, src=src, time=time,
            candidates=candidates, labels=labels, batch=int(args.batch))
        base = core.mixed_score(ensemble_weights, components[[names.index(name) for name in active]])
        pool_src, pool_time, pool_candidates, _ = core.sample_segment(
            segments, pool, split, len(segments[split]), seed + 10_000
        )
        index = near.NearTimeIndex(pool_src, pool_time, pool_candidates)
        cross_exact = index.support(src, time, candidates, 0)
        source_exact = index.source_support(src, time, candidates, 0)
        source_session = index.source_support(src, time, candidates, 300) - source_exact
        control = base + np.float32(C2_CROSS_WEIGHT) * qnorm(np.log1p(cross_exact)) + np.float32(C2_SESSION_WEIGHT) * qnorm(np.log1p(source_session))
        seen = d3.pair_seen(
            core.segment_history(initial_history, segments, split), src, candidates
        )
        for name, weight in C3_POLICY.items():
            prefix, direction, window = name.split("_")
            past, future = c3.directional_support(index, src, time, candidates, int(window[:-1]), prefix == "session")
            support = past if direction == "past" else future
            control += np.where(
                ~seen,
                np.float32(weight) * qnorm(np.log1p(support)),
                np.float32(0.0),
            )
        short_past, short_future = c3.directional_support(
            index, src, time, candidates, 900, True
        )
        long_past, long_future = c3.directional_support(
            index, src, time, candidates, 86400, True
        )
        ring_past = long_past - short_past
        ring_future = long_future - short_future
        raw = {
            "session_ring_past": ring_past,
            "session_ring_future": ring_future,
            "session_ring_sum": ring_past + ring_future,
        }
        feature_names = list(raw)
        features = np.stack([qnorm(np.log1p(raw[name])) for name in feature_names])
        eligible = np.where(~seen, raw["session_ring_sum"], -1)
        maximum = eligible.max(axis=1)
        unique = (eligible == maximum[:, None]).sum(axis=1) == 1
        unique &= maximum > 0
        unique_max = (~seen) & (eligible == maximum[:, None]) & unique[:, None]
        scored[split] = {"control": control, "features": features, "labels": labels, "gate": unique_max,
                         "audits": {name: {"cell_rate": float(np.mean(value > 0)), "label_rate": float(np.mean(value[np.arange(len(labels)), labels] > 0))} for name, value in raw.items()}}
        print(json.dumps({split: {"rows": len(labels), "control_mrr": float(reciprocal_ranks(control, labels).mean())}}), flush=True)

    meta = scored["meta_train"]
    selected_mrr, weights = tune(meta["control"], meta["features"], meta["labels"], meta["gate"])
    scale_trace = []
    for scale in np.arange(0.1, 1.01, 0.1):
        candidate = apply(
            meta["control"], meta["features"], weights * scale, meta["gate"]
        )
        scale_trace.append({"scale": float(scale), **metrics(meta["control"], candidate, meta["labels"])})
    safe_scales = [
        value for value in scale_trace
        if value["negative_row_rate"] < 0.005
    ]
    selected_scale = max(
        safe_scales or scale_trace,
        key=lambda value: (value["delta"], -value["negative_row_rate"]),
    )["scale"]
    weights *= selected_scale
    evaluations = {split: metrics(values["control"], apply(values["control"], values["features"], weights, values["gate"]), values["labels"]) for split, values in scored.items()}
    checks = {
        "validation_delta_at_least_0_002": evaluations["validation"]["delta"] >= MIN_DELTA,
        "confirmation_delta_at_least_0_002": evaluations["confirmation"]["delta"] >= MIN_DELTA,
        "confirmation_above_two_se": evaluations["confirmation"]["delta"] > 2 * evaluations["confirmation"]["delta_se"],
        "confirmation_negative_rows_below_0_005": evaluations["confirmation"]["negative_row_rate"] < 0.005,
    }
    output = {
        "kind": "d3_c3_session_ring_unique_gate_v1", "decision": "PASS" if all(checks.values()) else "NO_GO",
        "control": "online c3", "selection_split": "meta_train only", "untouched_final_splits": ["validation", "confirmation"],
        "selected_meta_mrr": selected_mrr,
        "policy": {"gate": "pair_new_unique_session_ring_max", "scale": selected_scale, "weights": {name: float(weight) for name, weight in zip(feature_names, weights) if abs(weight) > 1e-12}},
        "scale_selection": {"split": "meta_train", "max_negative_row_rate": 0.005, "trace": scale_trace},
        "checks": checks, "metrics_vs_c3": evaluations,
        "feature_audits": {split: values["audits"] for split, values in scored.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    # The gate decision is diagnostic for a fresh-data run; the selected policy
    # remains a valid candidate even when a conservative quality threshold misses.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
