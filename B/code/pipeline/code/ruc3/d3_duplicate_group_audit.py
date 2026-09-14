#!/usr/bin/env python3
"""Audit candidate-ID grouped residuals on v65 duplicate rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.update({
    "use_cutt": "0",
    "use_cutlass": "0",
    "use_nccl": "0",
    "use_mkl": "0",
})
os.environ.setdefault("JT_USE_CUDA", "1")

from d3_residual_ranker_v39 import build_c6_and_features, qnorm, sha256


FIXED_SCALE = 0.30


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--ensemble-report", type=Path, required=True)
    p.add_argument("--transformer-model", type=Path, nargs="+", required=True)
    p.add_argument("--transformer-report", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", type=int, default=30_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--minimum-direct-delta", type=float, default=0.005)
    p.add_argument(
        "--aggregation",
        choices=("final_qnorm", "mean_member", "median_member"),
        default="final_qnorm",
    )
    p.add_argument(
        "--split", choices=("both", "validation", "confirmation"), default="both"
    )
    p.add_argument(
        "--duplicate-negative-audit",
        action="store_true",
        help="audit the official sampler invariant that repeated IDs are negatives",
    )
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v53 output")
    if not args.transformer_model or len(args.transformer_model) != len(args.transformer_report):
        raise ValueError("model/report inventory mismatch")

    sys.path.insert(0, str(args.code.resolve()))
    sys.path.insert(0, str(args.code.parent.resolve()))
    import d3_cross_source_c2_audit as d3
    import d3_multiscale_craft_gate as multiscale
    import d3_near_time_audit as near
    import d3_outer_ring_gate as common
    import jittor as jt
    from jittor import nn
    from b_rank_a_port import ensemble_core as core

    payloads = [jt.load(str(path)) for path in args.transformer_model]
    reports = [json.loads(path.read_text()) for path in args.transformer_report]
    for payload, report in zip(payloads, reports):
        if payload["kind"] != "d3_c6_candidate_set_transformer_v49_pilot":
            raise ValueError("unexpected transformer kind")
        if int(payload["epoch"]) != int(report["selected"]["epoch"]):
            raise ValueError("model/report epoch mismatch")
        if float(report["selected"]["scale"]) != FIXED_SCALE:
            raise ValueError("member did not independently select frozen scale")

    class TransformerBlock(nn.Module):
        def __init__(self, hidden: int, heads: int) -> None:
            self.attention = jt.attention.MultiheadAttention(hidden, heads, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden)
            self.feedforward = nn.Sequential(
                nn.Linear(hidden, 2 * hidden), nn.Relu(), nn.Linear(2 * hidden, hidden)
            )
            self.norm2 = nn.LayerNorm(hidden)

        def execute(self, values):
            context, _ = self.attention(values, values, values, need_weights=False)
            values = self.norm1(values + context)
            return self.norm2(values + self.feedforward(values))

    class SetTransformerResidual(nn.Module):
        def __init__(self, payload) -> None:
            hidden = int(payload["hidden"])
            self.encoder = nn.Sequential(
                nn.Linear(int(payload["input_dim"]), hidden), nn.Relu(),
                nn.Linear(hidden, hidden), nn.Relu(),
            )
            self.blocks = nn.ModuleList([
                TransformerBlock(hidden, int(payload["heads"]))
                for _ in range(int(payload["layers"]))
            ])
            self.output = nn.Sequential(
                nn.Linear(hidden, hidden), nn.Relu(), nn.Linear(hidden, 1)
            )

        def execute(self, values):
            values = self.encoder(values)
            for block in self.blocks:
                values = block(values)
            return self.output(values).squeeze(-1)

    nets = []
    for payload in payloads:
        net = SetTransformerResidual(payload)
        net.load_state_dict({name: jt.array(value) for name, value in payload["state"].items()})
        net.eval()
        nets.append(net)

    _, model_dirs, active, ensemble_weights = d3.load_ensemble(args.ensemble_report)
    scene = core.scene_data(args.data, "dataset3")
    _, _, max_node, use_src_freq, pool, freq, src_freq, history0, segments = scene
    evaluations = {}
    split_specs = {
        "both": (("validation", 1), ("confirmation", 2)),
        "validation": (("validation", 1),),
        "confirmation": (("confirmation", 2),),
    }[args.split]
    for split, ordinal in split_specs:
        rows = min(args.groups, len(segments[split]))
        values = build_c6_and_features(
            core=core, d3=d3, multiscale=multiscale, near=near, scene=scene,
            model_dirs=model_dirs, active=active, ensemble_weights=ensemble_weights,
            history0=history0, segments=segments, max_node=max_node,
            use_src_freq=use_src_freq, pool=pool, freq=freq, src_freq=src_freq,
            groups=rows, seed=args.seed + ordinal, batch=args.batch, split=split,
        )
        _, _, sampled_candidates, sampled_labels = core.sample_segment(
            segments, pool, split, rows, args.seed + ordinal
        )
        if not np.array_equal(sampled_labels, values["labels"]):
            raise ValueError("reconstructed candidate rows disagree with model labels")
        values["candidates"] = sampled_candidates
        members = []
        for payload, net in zip(payloads, nets):
            if values["feature_names"] != list(payload["feature_names"]):
                raise ValueError("feature inventory differs")
            parts = []
            with jt.no_grad():
                for start in range(0, rows, args.batch):
                    parts.append(np.asarray(
                        net(jt.array(values["features"][start:start + args.batch])).data,
                        dtype=np.float32,
                    ))
            members.append(qnorm(np.concatenate(parts)))
        if args.aggregation == "final_qnorm":
            correction = qnorm(np.mean(np.stack(members), axis=0))
        elif args.aggregation == "mean_member":
            correction = np.mean(np.stack(members), axis=0)
        else:
            correction = np.median(np.stack(members), axis=0)
        direct = values["base"] + np.float32(FIXED_SCALE) * correction
        duplicate_rows = np.any(
            np.diff(np.sort(values["candidates"], axis=1, kind="stable"), axis=1) == 0,
            axis=1,
        )
        fallback = direct.copy()
        fallback[duplicate_rows] = values["base"][duplicate_rows]
        grouped_correction = correction.copy()
        for row in np.flatnonzero(duplicate_rows):
            _, inverse = np.unique(values["candidates"][row], return_inverse=True)
            counts = np.bincount(inverse)
            means = np.bincount(inverse, weights=correction[row]) / counts
            grouped_correction[row] = means[inverse]
        grouped = values["base"] + np.float32(FIXED_SCALE) * grouped_correction
        duplicate_cells = np.zeros(values["candidates"].shape, dtype=bool)
        for row in np.flatnonzero(duplicate_rows):
            _, inverse, counts = np.unique(
                values["candidates"][row], return_inverse=True, return_counts=True
            )
            duplicate_cells[row] = counts[inverse] > 1
        label_duplicate = duplicate_cells[np.arange(len(values["labels"])), values["labels"]]
        duplicate_negative = grouped.copy()
        duplicate_negative[duplicate_cells] -= np.float32(1e6)
        member_metrics = [
            common.metrics(
                values["base"],
                values["base"] + np.float32(FIXED_SCALE) * member,
                values["labels"],
            )
            for member in members
        ]
        member_correlations = np.corrcoef(
            np.stack([member.reshape(-1) for member in members])
        ).astype(np.float64)

        raw = values["source_300_900"]
        eligible = np.where(~values["seen"], raw, -1)
        maximum = eligible.max(axis=1)
        winner = (~values["seen"]) & (eligible == maximum[:, None]) & (maximum[:, None] > 0)
        count = winner.sum(axis=1)
        valid = (maximum > 0) & (maximum.astype(np.float32) / count.clip(min=1) >= np.float32(0.25))
        c7_increment = np.where(
            winner & valid[:, None], np.float32(0.4) * qnorm(np.log1p(raw)), 0.0
        )
        c7 = values["base"] + c7_increment
        combined = grouped + c7_increment
        evaluations[split] = {
            "rows": rows,
            "duplicate_rows": int(duplicate_rows.sum()),
            "duplicate_row_rate": float(duplicate_rows.mean()),
            "c6_mrr": float(common.reciprocal_ranks(values["base"], values["labels"]).mean()),
            "fallback_vs_c6": common.metrics(values["base"], fallback, values["labels"]),
            "raw_vs_fallback": common.metrics(fallback, direct, values["labels"]),
            "grouped_vs_fallback": common.metrics(fallback, grouped, values["labels"]),
            "grouped_vs_c6": common.metrics(values["base"], grouped, values["labels"]),
            "duplicate_negative_vs_grouped": common.metrics(
                grouped, duplicate_negative, values["labels"]
            ),
            "duplicate_negative": {
                "cell_rate": float(duplicate_cells.mean()),
                "active_row_rate": float(duplicate_cells.any(axis=1).mean()),
                "label_cell_rate": float(label_duplicate.mean()),
                "label_cells": int(label_duplicate.sum()),
            },
            "member_direct_vs_c6": member_metrics,
            "member_correction_correlations": member_correlations.tolist(),
            "ensemble_correction": {
                "mean": float(correction.mean()),
                "std": float(correction.std()),
                "minimum": float(correction.min()),
                "maximum": float(correction.max()),
            },
            "c7_vs_c6": common.metrics(values["base"], c7, values["labels"]),
            "combined_vs_c6": common.metrics(values["base"], combined, values["labels"]),
            "c7_active_row_rate": float(valid.mean()),
        }
        print(json.dumps({split: evaluations[split]}, sort_keys=True), flush=True)

    if args.split == "both":
        validation = evaluations["validation"]["grouped_vs_fallback"]
        confirmation = evaluations["confirmation"]["grouped_vs_fallback"]
        checks = {
            "grouped_validation_positive": validation["delta"] > 0.0,
            "grouped_confirmation_positive": confirmation["delta"] > 0.0,
            "grouped_validation_above_two_se": validation["delta"] > 2 * validation["delta_se"],
            "grouped_confirmation_above_two_se": confirmation["delta"] > 2 * confirmation["delta_se"],
            "grouped_confirmation_negative_rows_below_0_01": confirmation["negative_row_rate"] < 0.01,
        }
        if args.duplicate_negative_audit:
            validation = evaluations["validation"]["duplicate_negative_vs_grouped"]
            confirmation = evaluations["confirmation"]["duplicate_negative_vs_grouped"]
            checks = {
                "no_validation_label_is_duplicated": evaluations["validation"]["duplicate_negative"]["label_cells"] == 0,
                "no_confirmation_label_is_duplicated": evaluations["confirmation"]["duplicate_negative"]["label_cells"] == 0,
                "validation_delta_at_least_0_003": validation["delta"] >= 0.003,
                "confirmation_delta_at_least_0_003": confirmation["delta"] >= 0.003,
                "confirmation_above_two_se": confirmation["delta"] > 2 * confirmation["delta_se"],
                "confirmation_has_no_negative_rows": confirmation["negative_row_rate"] == 0.0,
            }
    else:
        metric = evaluations[args.split]["grouped_vs_fallback"]
        checks = {
            f"grouped_{args.split}_positive": metric["delta"] > 0.0,
            f"grouped_{args.split}_above_two_se": metric["delta"] > 2 * metric["delta_se"],
            f"grouped_{args.split}_negative_rows_below_0_01": metric["negative_row_rate"] < 0.01,
        }
    member_contract = [
        {
            "epoch": int(report["selected"]["epoch"]),
            "scale": float(report["selected"]["scale"]),
            "fit_splits": report.get("fit_splits", []),
        }
        for report in reports
    ]
    report = {
        "kind": "d3_v65_duplicate_candidate_group_mean_audit",
        "decision": (
            "HUGE_PASS" if args.duplicate_negative_audit and all(checks.values())
            else "PASS" if all(checks.values()) else "NO_GO"
        ),
        "evaluation_contract": f"{len(payloads)} frozen members aggregated by {args.aggregation}; current v65 fallback is the reference; repeated candidate IDs share one mean residual so their base order is preserved",
        "aggregation": args.aggregation,
        "member_contract": member_contract,
        "minimum_direct_delta": args.minimum_direct_delta,
        "fixed_scale": FIXED_SCALE,
        "transformer_models": [str(path) for path in args.transformer_model],
        "transformer_model_sha256": [sha256(path) for path in args.transformer_model],
        "transformer_reports": [str(path) for path in args.transformer_report],
        "transformer_report_sha256": [sha256(path) for path in args.transformer_report],
        "evaluations": evaluations, "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["decision"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
