#!/usr/bin/env python3
"""Strict D3 MRR gate for +/-5s support incremental to v26 exact-time support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import d3_cross_source_c2_audit as d3
import d3_near_time_audit as near
from b_rank_a_port import ensemble_core as core


V26_WEIGHT = 0.10
MIN_DELTA = 0.003


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ensemble-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--groups", type=int, default=30_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument(
        "--feature", choices=("cross_source", "source_session"),
        default="cross_source",
    )
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--residual-weight", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if d3.sha256(args.data) != d3.DATA_SHA256:
        raise ValueError("official data hash differs")
    report, model_dirs, active, weights = d3.load_ensemble(args.ensemble_report)
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
    evaluations = {}
    audits = {}
    for ordinal, name in enumerate(("validation", "confirmation"), start=1):
        seed = int(args.seed) + ordinal
        src, time, candidates, labels = core.sample_segment(
            segments, pool, name, int(args.groups), seed
        )
        names, components, _ = core.component_scores(
            scene="dataset3",
            model_dirs=model_dirs,
            history=core.segment_history(initial_history, segments, name),
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
        indices = [names.index(component) for component in active]
        control = core.mixed_score(weights, components[indices])
        pool_src, pool_time, pool_candidates, _ = core.sample_segment(
            segments, pool, name, len(segments[name]), seed + 10_000
        )
        index = near.NearTimeIndex(pool_src, pool_time, pool_candidates)
        support = (
            index.support
            if args.feature == "cross_source"
            else index.source_support
        )
        exact = support(src, time, candidates, 0)
        expanded = support(src, time, candidates, int(args.window))
        incremental = expanded - exact
        if incremental.min(initial=0) < 0:
            raise ValueError("near-time support is below exact-time support")
        history = core.segment_history(initial_history, segments, name)
        seen = d3.pair_seen(history, src, candidates)
        v26 = d3.apply_policy(
            control, d3.qnorm(np.log1p(exact).astype(np.float32)), seen,
            V26_WEIGHT, "all",
        )
        candidate = d3.apply_policy(
            v26, d3.qnorm(np.log1p(incremental).astype(np.float32)), seen,
            float(args.residual_weight), "all",
        )
        evaluations[name] = d3.metrics(v26, candidate, labels)
        audits[name] = near.audit(incremental, labels)
        print(
            json.dumps(
                {name: {"metrics_vs_v26": evaluations[name], "audit": audits[name]}},
                sort_keys=True,
            ),
            flush=True,
        )
    checks = {
        "validation_delta_at_least_0_003": evaluations["validation"]["delta"]
        >= MIN_DELTA,
        "confirmation_delta_at_least_0_003": evaluations["confirmation"]["delta"]
        >= MIN_DELTA,
        "confirmation_positive_after_one_se": evaluations["confirmation"]["delta"]
        > evaluations["confirmation"]["delta_se"],
        "confirmation_negative_rows_below_0_005": evaluations["confirmation"][
            "negative_row_rate"
        ] < 0.005,
    }
    output = {
        "kind": "d3_incremental_candidate_support_strict_gate_v1",
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "fixed_policy": {
            "feature": args.feature,
            "window_seconds": int(args.window),
            "exclude_exact_time_support": True,
            "v26_exact_weight": V26_WEIGHT,
            "residual_weight": float(args.residual_weight),
            "gate": "all",
        },
        "checks": checks,
        "metrics_vs_v26": evaluations,
        "feature_audits": audits,
        "ensemble_report": {
            "path": str(args.ensemble_report.resolve()),
            "sha256": d3.sha256(args.ensemble_report),
        },
        "active_components": active,
        "source_sha256": d3.sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return 0 if output["decision"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
