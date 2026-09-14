#!/usr/bin/env python3
"""Strict D3 multiscale/graph residual gate against the real c2 formula."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


C2_CROSS_WEIGHT = 0.10
C2_SESSION_WEIGHT = 0.05
MIN_DELTA = 0.003


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
        "top1_changed": float(
            np.mean(np.argmax(control, axis=1) != np.argmax(candidate, axis=1))
        ),
    }


def _range_support(
    index,
    source: np.ndarray,
    time: np.ndarray,
    candidates: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    same_source: bool,
) -> np.ndarray:
    lower = np.asarray(lower, dtype=np.uint64)[:, None]
    upper = np.asarray(upper, dtype=np.uint64)[:, None]
    items = np.asarray(candidates, dtype=np.uint64)
    item_base = items << np.uint64(32)
    total = np.searchsorted(index.total_keys, item_base | upper, side="right")
    total -= np.searchsorted(index.total_keys, item_base | lower, side="left")

    source_item = (
        np.asarray(source, dtype=np.uint64)[:, None] << np.uint64(32)
    ) | items
    group = np.searchsorted(index.source_item_ids, source_item)
    known = group < len(index.source_item_ids)
    known[known] &= index.source_item_ids[group[known]] == source_item[known]
    group[~known] = len(index.source_item_ids)
    own_base = group.astype(np.uint64) << np.uint64(32)
    own = np.searchsorted(index.own_keys, own_base | upper, side="right")
    own -= np.searchsorted(index.own_keys, own_base | lower, side="left")
    output = own if same_source else total - own
    if output.min(initial=0) < 0:
        raise ValueError("negative range support")
    return output.astype(np.int32, copy=False)


def directional_support(index, source, time, candidates, window, same_source):
    time = np.asarray(time, dtype=np.int64)
    past = _range_support(
        index,
        source,
        time,
        candidates,
        np.maximum(time - int(window), 0),
        np.maximum(time - 1, 0),
        same_source,
    )
    future = _range_support(
        index,
        source,
        time,
        candidates,
        time + 1,
        time + int(window),
        same_source,
    )
    return past, future


def build_features(index, src, time, candidates, components, names):
    feature_names: list[str] = []
    values: list[np.ndarray] = []
    raw: dict[str, np.ndarray] = {}
    for prefix, same_source in (("session", True), ("cross", False)):
        for window in (1, 5, 30, 300):
            past, future = directional_support(
                index, src, time, candidates, window, same_source
            )
            for direction, support in (("past", past), ("future", future)):
                name = f"{prefix}_{direction}_{window}s"
                raw[name] = support
                feature_names.append(name)
                values.append(qnorm(np.log1p(support).astype(np.float32)))
    prop_indices = [i for i, name in enumerate(names) if name.endswith(":prop")]
    if prop_indices:
        feature_names.append("craft_graph_prop_mean")
        values.append(qnorm(components[prop_indices].mean(axis=0)))
    return feature_names, np.stack(values), raw


def apply(control, features, weights, gate):
    residual = np.tensordot(weights, features, axes=(0, 0)).astype(np.float32)
    return control + np.where(gate, residual, 0.0)


def tune(control, features, labels, gates):
    best = (float(reciprocal_ranks(control, labels).mean()), np.zeros(len(features)), "all")
    for gate_name, gate in gates.items():
        weights = np.zeros(len(features), dtype=np.float64)
        score = control.copy()
        value = float(reciprocal_ranks(score, labels).mean())
        for steps in (
            np.arange(-0.10, 0.201, 0.025),
            np.arange(-0.04, 0.081, 0.01),
        ):
            changed = True
            while changed:
                changed = False
                for index in range(len(features)):
                    for delta in steps:
                        if delta == 0.0:
                            continue
                        candidate = score + np.where(
                            gate, np.float32(delta) * features[index], 0.0
                        )
                        candidate_value = float(reciprocal_ranks(candidate, labels).mean())
                        if candidate_value > value + 1e-10:
                            score = candidate
                            weights[index] += float(delta)
                            value = candidate_value
                            changed = True
        active = np.flatnonzero(np.abs(weights) > 1e-12)
        # Keep the residual compact: refit only the four strongest coordinates.
        if len(active) > 4:
            keep = active[np.argsort(np.abs(weights[active]))[-4:]]
            compact = np.zeros_like(weights)
            compact[keep] = weights[keep]
            weights = compact
            score = apply(control, features, weights, gate)
            value = float(reciprocal_ranks(score, labels).mean())
        candidate = (value, weights, gate_name)
        if candidate[0] > best[0] + 1e-10:
            best = candidate
    return best


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
    import ensemble_core as core

    if d3.sha256(args.data) != d3.DATA_SHA256:
        raise ValueError("official data hash differs")
    _, model_dirs, active, ensemble_weights = d3.load_ensemble(args.ensemble_report)
    (
        _train, _test, max_node, use_src_freq, pool, freq, src_freq,
        initial_history, segments,
    ) = core.scene_data(args.data, "dataset3")
    scored = {}
    for ordinal, split in enumerate(("meta_train", "validation", "confirmation")):
        seed = int(args.seed) + ordinal
        src, time, candidates, labels = core.sample_segment(
            segments, pool, split, int(args.groups), seed
        )
        names, components, _ = core.component_scores(
            scene="dataset3",
            model_dirs=model_dirs,
            history=core.segment_history(initial_history, segments, split),
            max_node=max_node,
            use_src_freq=use_src_freq,
            freq=freq,
            src_freq=src_freq,
            src=src,
            time=time,
            candidates=candidates,
            labels=labels,
            batch=int(args.batch),
        )
        base = core.mixed_score(
            ensemble_weights, components[[names.index(name) for name in active]]
        )
        pool_src, pool_time, pool_candidates, _ = core.sample_segment(
            segments, pool, split, len(segments[split]), seed + 10_000
        )
        index = near.NearTimeIndex(pool_src, pool_time, pool_candidates)
        cross_exact = index.support(src, time, candidates, 0)
        source_exact = index.source_support(src, time, candidates, 0)
        source_session = (
            index.source_support(src, time, candidates, 300) - source_exact
        )
        control = (
            base
            + np.float32(C2_CROSS_WEIGHT)
            * qnorm(np.log1p(cross_exact).astype(np.float32))
            + np.float32(C2_SESSION_WEIGHT)
            * qnorm(np.log1p(source_session).astype(np.float32))
        )
        feature_names, features, raw = build_features(
            index, src, time, candidates, components, names
        )
        history = core.segment_history(initial_history, segments, split)
        seen = d3.pair_seen(history, src, candidates)
        margin = np.sort(control, axis=1)[:, -1] - np.sort(control, axis=1)[:, -2]
        scored[split] = {
            "control": control,
            "features": features,
            "labels": labels,
            "gates": {
                "all": np.ones(control.shape, dtype=bool),
                "pair_new": ~seen,
                "row_no_seen": np.broadcast_to(
                    (~np.any(seen, axis=1))[:, None], control.shape
                ),
                "low_margin": np.broadcast_to(
                    (margin <= np.quantile(margin, 0.50))[:, None], control.shape
                ),
            },
            "audits": {
                name: {
                    "cell_rate": float(np.mean(values > 0)),
                    "label_rate": float(
                        np.mean(values[np.arange(len(labels)), labels] > 0)
                    ),
                }
                for name, values in raw.items()
            },
        }
        print(json.dumps({split: {"rows": len(labels), "control_mrr": float(reciprocal_ranks(control, labels).mean())}}), flush=True)

    meta = scored["meta_train"]
    selected_mrr, weights, gate_name = tune(
        meta["control"], meta["features"], meta["labels"], meta["gates"]
    )
    evaluations = {}
    for split, values in scored.items():
        candidate = apply(
            values["control"], values["features"], weights, values["gates"][gate_name]
        )
        evaluations[split] = metrics(values["control"], candidate, values["labels"])
    checks = {
        "validation_delta_at_least_0_003": evaluations["validation"]["delta"] >= MIN_DELTA,
        "confirmation_delta_at_least_0_003": evaluations["confirmation"]["delta"] >= MIN_DELTA,
        "confirmation_above_one_se": evaluations["confirmation"]["delta"] > evaluations["confirmation"]["delta_se"],
        "confirmation_negative_rows_below_0_01": evaluations["confirmation"]["negative_row_rate"] < 0.01,
    }
    output = {
        "kind": "d3_c2_multiscale_craft_graph_strict_gate_v1",
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "control": "real c2 = ensemble + 0.10 cross-source exact + 0.05 same-source +/-300s excluding exact",
        "selection_split": "meta_train only",
        "untouched_final_splits": ["validation", "confirmation"],
        "selected_meta_mrr": selected_mrr,
        "policy": {
            "gate": gate_name,
            "weights": {
                name: float(weight)
                for name, weight in zip(feature_names, weights)
                if abs(weight) > 1e-12
            },
        },
        "checks": checks,
        "metrics_vs_c2": evaluations,
        "feature_audits": {split: values["audits"] for split, values in scored.items()},
        "active_ensemble_components": active,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return 0 if output["decision"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
