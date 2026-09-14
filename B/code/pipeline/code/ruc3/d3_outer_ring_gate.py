#!/usr/bin/env python3
"""Evaluate disjoint 1-7d and 7-30d D3 session bands on top of online c5."""

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
C5_POLICY = np.asarray([0.2625, 0.28, -0.0525], dtype=np.float32)
WINDOWS = (86_400, 604_800, 2_592_000)


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


def unique_maximum(raw_sum: np.ndarray, seen: np.ndarray) -> np.ndarray:
    eligible = np.where(~seen, raw_sum, -1)
    maximum = eligible.max(axis=1)
    unique = ((eligible == maximum[:, None]).sum(axis=1) == 1) & (maximum > 0)
    return (~seen) & (eligible == maximum[:, None]) & unique[:, None]


def apply(
    control: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray,
    gate: np.ndarray,
) -> np.ndarray:
    residual = np.tensordot(weights, features, axes=(0, 0)).astype(np.float32)
    return control + np.where(gate, residual, 0.0)


def tune(
    control: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    gate: np.ndarray,
) -> tuple[float, np.ndarray]:
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
    if len(active) > 4:
        keep = active[np.argsort(np.abs(weights[active]))[-4:]]
        compact = np.zeros_like(weights)
        compact[keep] = weights[keep]
        weights = compact
        value = float(
            reciprocal_ranks(apply(control, features, weights, gate), labels).mean()
        )
    return value, weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--ensemble-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=5_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.code.resolve()))
    import d3_cross_source_c2_audit as d3
    import d3_multiscale_craft_gate as multiscale
    import d3_near_time_audit as near
    from b_rank_a_port import ensemble_core as core

    if d3.sha256(args.data) != d3.DATA_SHA256:
        raise ValueError("official data hash differs")
    _, model_dirs, active, ensemble_weights = d3.load_ensemble(args.ensemble_report)
    (
        _train,
        _test,
        max_node,
        use_src_freq,
        pool,
        freq,
        src_freq,
        initial_history,
        segments,
    ) = core.scene_data(args.data, "dataset3")
    scored = {}
    feature_names = None
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
        source_session = index.source_support(src, time, candidates, 300) - source_exact
        control = base + np.float32(C2_CROSS_WEIGHT) * qnorm(np.log1p(cross_exact))
        control += np.float32(C2_SESSION_WEIGHT) * qnorm(np.log1p(source_session))
        history = core.segment_history(initial_history, segments, split)
        seen = d3.pair_seen(history, src, candidates)
        for name, weight in C3_POLICY.items():
            prefix, direction, window = name.split("_")
            past, future = multiscale.directional_support(
                index, src, time, candidates, int(window[:-1]), prefix == "session"
            )
            support = past if direction == "past" else future
            control += np.where(
                ~seen, np.float32(weight) * qnorm(np.log1p(support)), 0.0
            )

        directional = {
            window: multiscale.directional_support(
                index, src, time, candidates, window, True
            )
            for window in WINDOWS
        }
        inner_past, inner_future = multiscale.directional_support(
            index, src, time, candidates, 900, True
        )
        day_past, day_future = directional[86_400]
        c5_raw = np.stack(
            (day_past - inner_past, day_future - inner_future), axis=0
        )
        c5_features = np.stack(
            (
                qnorm(np.log1p(c5_raw[0])),
                qnorm(np.log1p(c5_raw[1])),
                qnorm(np.log1p(c5_raw.sum(axis=0))),
            )
        )
        c5_gate = unique_maximum(c5_raw.sum(axis=0), seen)
        c5_control = apply(control, c5_features, C5_POLICY, c5_gate)

        week_past, week_future = directional[604_800]
        month_past, month_future = directional[2_592_000]
        raw = {
            "day7_past": week_past - day_past,
            "day7_future": week_future - day_future,
            "day30_past": month_past - week_past,
            "day30_future": month_future - week_future,
        }
        raw["day7_sum"] = raw["day7_past"] + raw["day7_future"]
        raw["day30_sum"] = raw["day30_past"] + raw["day30_future"]
        raw["outer_sum"] = raw["day7_sum"] + raw["day30_sum"]
        feature_names = list(raw)
        features = np.stack([qnorm(np.log1p(raw[name])) for name in feature_names])
        gate = unique_maximum(raw["outer_sum"], seen)
        scored[split] = {
            "control": c5_control,
            "features": features,
            "labels": labels,
            "gate": gate,
            "audits": {
                "active_row_rate": float(gate.any(axis=1).mean()),
                **{
                    name: {
                        "cell_rate": float(np.mean(value > 0)),
                        "label_rate": float(
                            np.mean(value[np.arange(len(labels)), labels] > 0)
                        ),
                    }
                    for name, value in raw.items()
                },
            },
        }
        print(
            json.dumps(
                {
                    split: {
                        "rows": len(labels),
                        "c5_mrr": float(reciprocal_ranks(c5_control, labels).mean()),
                        "outer_active_row_rate": float(gate.any(axis=1).mean()),
                    }
                }
            ),
            flush=True,
        )

    meta = scored["meta_train"]
    selected_mrr, weights = tune(
        meta["control"], meta["features"], meta["labels"], meta["gate"]
    )
    evaluations = {
        split: metrics(
            values["control"],
            apply(values["control"], values["features"], weights, values["gate"]),
            values["labels"],
        )
        for split, values in scored.items()
    }
    checks = {
        "validation_delta_at_least_0_003": evaluations["validation"]["delta"] >= 0.003,
        "confirmation_delta_at_least_0_003": evaluations["confirmation"]["delta"] >= 0.003,
        "confirmation_above_two_se": evaluations["confirmation"]["delta"]
        > 2 * evaluations["confirmation"]["delta_se"],
        "confirmation_negative_rows_below_0_005": evaluations["confirmation"][
            "negative_row_rate"
        ]
        < 0.005,
    }
    output = {
        "kind": "d3_c5_outer_session_bands_gate_v1",
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "control": "online c5 frozen policy",
        "selection_split": "meta_train only",
        "untouched_final_splits": ["validation", "confirmation"],
        "selected_meta_mrr": selected_mrr,
        "policy": {
            "gate": "pair_new_unique_outer_session_max",
            "weights": {
                name: float(weight)
                for name, weight in zip(feature_names, weights)
                if abs(weight) > 1e-12
            },
        },
        "checks": checks,
        "metrics_vs_c5": evaluations,
        "feature_audits": {
            split: values["audits"] for split, values in scored.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return 0 if output["decision"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
