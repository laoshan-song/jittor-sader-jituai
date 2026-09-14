#!/usr/bin/env python3
"""Break c5-excluded D3 ring-support ties with the frozen directional residual."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import d3_outer_ring_gate as common


SCALES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.70)
FRACTIONS = (0.25, 0.50, 0.75, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--ensemble-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=5_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--mode", choices=("winner", "group"), default="winner")
    parser.add_argument("--selection-negative-budget", type=float, default=0.005)
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
    scene = core.scene_data(args.data, "dataset3")
    _, _, max_node, use_src_freq, pool, freq, src_freq, history0, segments = scene
    scored = {}
    for ordinal, split in enumerate(("meta_train", "validation", "confirmation")):
        seed = int(args.seed) + ordinal
        src, time, candidates, labels = core.sample_segment(
            segments, pool, split, int(args.groups), seed
        )
        history = core.segment_history(history0, segments, split)
        names, components, _ = core.component_scores(
            scene="dataset3", model_dirs=model_dirs, history=history,
            max_node=max_node, use_src_freq=use_src_freq, freq=freq,
            src_freq=src_freq, src=src, time=time, candidates=candidates,
            labels=labels, batch=int(args.batch),
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
        c3 = base + np.float32(common.C2_CROSS_WEIGHT) * common.qnorm(np.log1p(cross_exact))
        c3 += np.float32(common.C2_SESSION_WEIGHT) * common.qnorm(np.log1p(source_session))
        seen = d3.pair_seen(history, src, candidates)
        for name, weight in common.C3_POLICY.items():
            prefix, direction, window = name.split("_")
            past, future = multiscale.directional_support(
                index, src, time, candidates, int(window[:-1]), prefix == "session"
            )
            support = past if direction == "past" else future
            c3 += np.where(~seen, np.float32(weight) * common.qnorm(np.log1p(support)), 0.0)

        short_past, short_future = multiscale.directional_support(
            index, src, time, candidates, 900, True
        )
        long_past, long_future = multiscale.directional_support(
            index, src, time, candidates, 86_400, True
        )
        ring_past = long_past - short_past
        ring_future = long_future - short_future
        ring_sum = ring_past + ring_future
        features = np.stack((
            common.qnorm(np.log1p(ring_past)),
            common.qnorm(np.log1p(ring_future)),
            common.qnorm(np.log1p(ring_sum)),
        ))
        residual = np.tensordot(common.C5_POLICY, features, axes=(0, 0)).astype(np.float32)
        unique_gate = common.unique_maximum(ring_sum, seen)
        c5 = c3 + np.where(unique_gate, residual, 0.0)

        eligible = np.where(~seen, ring_sum, -1)
        maximum = eligible.max(axis=1)
        tied = (~seen) & (eligible == maximum[:, None])
        tie_count = tied.sum(axis=1)
        tie_rows = (maximum > 0) & (tie_count >= 2) & ~unique_gate.any(axis=1)
        if args.mode == "group":
            # The tied set has strong recall but weak single-candidate ordering.
            # Preserve c3 ordering inside it and only lift the set as a unit.
            winner = tied
            valid = tie_rows
            gap = maximum.astype(np.float32) / tie_count.clip(min=1)
            residual = common.qnorm(np.log1p(ring_sum))
        else:
            directional = np.where(tied, residual, -np.inf)
            ordered = np.sort(directional, axis=1)
            gap = ordered[:, -1] - ordered[:, -2]
            winner = tied & (directional == ordered[:, -1, None])
            valid = tie_rows & np.isfinite(gap) & ((winner.sum(axis=1)) == 1)
            winner &= valid[:, None]
        scored[split] = {
            "control": c5, "labels": labels, "residual": residual,
            "winner": winner, "gap": gap, "valid": valid,
            "tie_count": tie_count,
        }
        print(json.dumps({split: {
            "rows": len(labels),
            "c5_mrr": float(common.reciprocal_ranks(c5, labels).mean()),
            "tie_row_rate": float(tie_rows.mean()),
            "directional_unique_tie_row_rate": float(valid.mean()),
            "label_in_tied_max_rate": float(np.mean(tied[np.arange(len(labels)), labels])),
        }}), flush=True)

    meta = scored["meta_train"]
    gaps = meta["gap"][meta["valid"]]
    trace = []
    for fraction in FRACTIONS:
        threshold = float(np.quantile(gaps, 1.0 - fraction, method="higher"))
        rows = meta["valid"] & (meta["gap"] >= threshold)
        gate = meta["winner"] & rows[:, None]
        for scale in SCALES:
            candidate = meta["control"] + np.where(
                gate, np.float32(scale) * meta["residual"], 0.0
            )
            trace.append({
                "fraction": fraction, "threshold": threshold, "scale": scale,
                "active_row_rate": float(rows.mean()),
                **common.metrics(meta["control"], candidate, meta["labels"]),
            })
    safe = [
        item
        for item in trace
        if item["negative_row_rate"] < float(args.selection_negative_budget)
    ]
    policy = max(safe or trace, key=lambda item: (item["delta"], -item["negative_row_rate"]))
    evaluations = {}
    for split, values in scored.items():
        rows = values["valid"] & (values["gap"] >= float(policy["threshold"]))
        gate = values["winner"] & rows[:, None]
        candidate = values["control"] + np.where(
            gate, np.float32(policy["scale"]) * values["residual"], 0.0
        )
        evaluations[split] = {
            **common.metrics(values["control"], candidate, values["labels"]),
            "active_row_rate": float(rows.mean()),
        }
    checks = {
        "validation_delta_at_least_0_003": evaluations["validation"]["delta"] >= 0.003,
        "confirmation_delta_at_least_0_003": evaluations["confirmation"]["delta"] >= 0.003,
        "confirmation_above_two_se": evaluations["confirmation"]["delta"] > 2 * evaluations["confirmation"]["delta_se"],
        "confirmation_negative_rows_below_0_005": evaluations["confirmation"]["negative_row_rate"] < 0.005,
    }
    report = {
        "kind": f"d3_c5_ring_tie_{args.mode}_gate_v1",
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "control": "online c5 frozen policy",
        "selection_split": "meta_train only",
        "mode": args.mode,
        "selection_negative_budget": float(args.selection_negative_budget),
        "policy": {key: policy[key] for key in ("fraction", "threshold", "scale", "active_row_rate")},
        "metrics_vs_c5": evaluations,
        "checks": checks,
        "selection_trace": trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["decision"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
