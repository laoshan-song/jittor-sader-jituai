#!/usr/bin/env python3
"""Pilot a candidate-set Transformer residual on top of frozen D3 c6.

Only meta_train is loaded.  A deterministic 60/40 row split is used for
fitting versus epoch/scale selection.  Validation and confirmation remain
untouched until the architecture and hyperparameters are frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

from d3_residual_ranker_v39 import (
    EXPECTED_DATA_SHA256,
    build_c6_and_features,
    qnorm,
    sha256,
)


SCALE_GRID = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--ensemble-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=30_000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--predict-batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--base-logit-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--freeze-final-epoch", action="store_true")
    parser.add_argument(
        "--split-strategy", choices=("hash", "temporal"), default="hash"
    )
    parser.add_argument("--temporal-fit-fraction", type=float, default=0.60)
    parser.add_argument("--full-fit", action="store_true")
    parser.add_argument("--fixed-scale", type=float)
    parser.add_argument(
        "--fit-splits", nargs="+", choices=("meta_train", "validation", "confirmation"),
        default=("meta_train",),
    )
    args = parser.parse_args()
    if args.output.exists() or args.model_output.exists():
        raise FileExistsError("refusing to overwrite v49 output")
    if sha256(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("official data hash differs")
    if args.hidden % args.heads:
        raise ValueError("hidden must be divisible by heads")
    if not 0.50 <= args.temporal_fit_fraction <= 0.90:
        raise ValueError("temporal fit fraction must be between 0.50 and 0.90")

    sys.path.insert(0, str(args.code.resolve()))
    sys.path.insert(0, str(args.code.parent.resolve()))
    import d3_cross_source_c2_audit as d3
    import d3_multiscale_craft_gate as multiscale
    import d3_near_time_audit as near
    import d3_outer_ring_gate as common
    import jittor as jt
    from jittor import nn
    from b_rank_a_port import ensemble_core as core

    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    class TransformerBlock(nn.Module):
        def __init__(self, hidden: int) -> None:
            self.attention = jt.attention.MultiheadAttention(
                hidden, args.heads, batch_first=True
            )
            self.norm1 = nn.LayerNorm(hidden)
            self.feedforward = nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                nn.Relu(),
                nn.Linear(2 * hidden, hidden),
            )
            self.norm2 = nn.LayerNorm(hidden)

        def execute(self, values):
            context, _ = self.attention(
                values, values, values, need_weights=False
            )
            values = self.norm1(values + context)
            return self.norm2(values + self.feedforward(values))

    class SetTransformerResidual(nn.Module):
        def __init__(self, feature_count: int, hidden: int) -> None:
            self.encoder = nn.Sequential(
                nn.Linear(feature_count, hidden),
                nn.Relu(),
                nn.Linear(hidden, hidden),
                nn.Relu(),
            )
            self.blocks = nn.ModuleList(
                [TransformerBlock(hidden) for _ in range(args.layers)]
            )
            self.output = nn.Sequential(
                nn.Linear(hidden, hidden), nn.Relu(), nn.Linear(hidden, 1)
            )

        def execute(self, values):
            values = self.encoder(values)
            for block in self.blocks:
                values = block(values)
            return self.output(values).squeeze(-1)

    def normalize_jt(values):
        centered = values - values.mean(dim=1, keepdims=True)
        return centered / jt.sqrt(
            (centered * centered).mean(dim=1, keepdims=True) + 1e-6
        )

    _, model_dirs, active, ensemble_weights = d3.load_ensemble(
        args.ensemble_report
    )
    scene = core.scene_data(args.data, "dataset3")
    _, _, max_node, use_src_freq, pool, freq, src_freq, history0, segments = scene
    split_ordinals = {"meta_train": 0, "validation": 1, "confirmation": 2}
    split_values = [
        build_c6_and_features(
            core=core,
            d3=d3,
            multiscale=multiscale,
            near=near,
            scene=scene,
            model_dirs=model_dirs,
            active=active,
            ensemble_weights=ensemble_weights,
            history0=history0,
            segments=segments,
            max_node=max_node,
            use_src_freq=use_src_freq,
            pool=pool,
            freq=freq,
            src_freq=src_freq,
            groups=min(args.groups, len(segments[split])),
            seed=args.seed + split_ordinals[split],
            batch=max(args.predict_batch, 128),
            split=split,
        )
        for split in args.fit_splits
    ]
    if any(
        part["feature_names"] != split_values[0]["feature_names"]
        or part["component_names"] != split_values[0]["component_names"]
        for part in split_values[1:]
    ):
        raise ValueError("split feature inventory differs")
    values = {
        key: np.concatenate([part[key] for part in split_values], axis=0)
        for key in ("src", "time", "labels", "base", "seen", "source_300_900", "features")
    }
    values["feature_names"] = split_values[0]["feature_names"]
    values["component_names"] = split_values[0]["component_names"]
    row_hash = (
        values["src"].astype(np.int64) * np.int64(1_000_003)
        + values["time"].astype(np.int64)
    )
    if args.full_fit:
        fit_mask = np.ones(len(row_hash), dtype=bool)
    elif args.split_strategy == "hash":
        fit_mask = (row_hash % np.int64(10)) < 6
    else:
        chronological = np.argsort(values["time"], kind="stable")
        fit_mask = np.zeros(len(chronological), dtype=bool)
        fit_mask[
            chronological[: int(args.temporal_fit_fraction * len(chronological))]
        ] = True
    audit_mask = np.ones(len(fit_mask), dtype=bool) if args.full_fit else ~fit_mask
    base_rr = common.reciprocal_ranks(values["base"], values["labels"])
    anchor_mask = ((row_hash // np.int64(10)) % np.int64(4)) == 0
    optimize_mask = fit_mask & ((base_rr < 1.0) | anchor_mask)
    fit_ids = np.flatnonzero(optimize_mask)
    audit_ids = np.flatnonzero(audit_mask)
    if len(fit_ids) < 1_000 or len(audit_ids) < 1_000:
        raise ValueError("unexpectedly small internal split")
    print(json.dumps({
        "features_ready": True,
        "rows": int(len(values["labels"])),
        "feature_dim": int(values["features"].shape[-1]),
        "fit_rows": int(len(fit_ids)),
        "audit_rows": int(len(audit_ids)),
        "c6_mrr": float(base_rr.mean()),
    }, sort_keys=True), flush=True)

    net = SetTransformerResidual(values["features"].shape[-1], args.hidden)
    optimizer = jt.optim.AdamW(
        net.parameters(), lr=args.learning_rate, weight_decay=2e-5
    )
    rng = np.random.default_rng(args.seed)

    def predict(row_ids: np.ndarray) -> np.ndarray:
        output = []
        net.eval()
        with jt.no_grad():
            for start in range(0, len(row_ids), args.predict_batch):
                ids = row_ids[start : start + args.predict_batch]
                output.append(np.asarray(
                    net(jt.array(values["features"][ids])).data,
                    dtype=np.float32,
                ))
        net.train()
        return qnorm(np.concatenate(output))

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(fit_ids)
        losses = []
        for start in range(0, len(order), args.batch):
            ids = order[start : start + args.batch]
            raw = net(jt.array(values["features"][ids]))
            correction = normalize_jt(raw)
            logits = (
                np.float32(args.base_logit_scale)
                * jt.array(values["base"][ids])
                + correction
            )
            loss = nn.cross_entropy_loss(logits, jt.array(values["labels"][ids]))
            loss += np.float32(1e-4) * (raw * raw).mean()
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.data).item()))

        correction = predict(audit_ids)
        choices = []
        scale_grid = (
            (float(args.fixed_scale),)
            if args.fixed_scale is not None else SCALE_GRID
        )
        for scale in scale_grid:
            candidate = (
                values["base"][audit_ids] + np.float32(scale) * correction
            )
            metrics = common.metrics(
                values["base"][audit_ids], candidate, values["labels"][audit_ids]
            )
            choices.append(
                (metrics["delta"], -metrics["negative_row_rate"], -scale,
                 float(scale), metrics)
            )
        _, _, _, selected_scale, metrics = max(choices)
        record = {
            "epoch": int(epoch),
            "loss": float(np.mean(losses)),
            "scale": selected_scale,
            "internal_audit": metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        key = (metrics["delta"], -metrics["negative_row_rate"], -epoch)
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "epoch": int(epoch),
                "scale": selected_scale,
                "metrics": metrics,
                "state": {
                    name: np.asarray(value.data).copy()
                    for name, value in net.state_dict().items()
                },
            }

    if args.freeze_final_epoch or args.full_fit:
        final = history[-1]
        best = {
            "key": (
                final["internal_audit"]["delta"],
                -final["internal_audit"]["negative_row_rate"],
                -final["epoch"],
            ),
            "epoch": int(final["epoch"]),
            "scale": float(final["scale"]),
            "metrics": final["internal_audit"],
            "state": {
                name: np.asarray(value.data).copy()
                for name, value in net.state_dict().items()
            },
        }

    checks = {
        "internal_delta_at_least_0_008": best["metrics"]["delta"] >= 0.008,
        "internal_above_two_se": (
            best["metrics"]["delta"] > 2 * best["metrics"]["delta_se"]
        ),
    }
    decision = (
        "TRAINED_FROZEN" if args.full_fit
        else "PROMISING" if all(checks.values()) else "NO_GO"
    )
    payload = {
        "kind": "d3_c6_candidate_set_transformer_v49_pilot",
        "input_dim": int(values["features"].shape[-1]),
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "heads": int(args.heads),
        "epoch": int(best["epoch"]),
        "scale": float(best["scale"]),
        "base_logit_scale": float(args.base_logit_scale),
        "feature_names": values["feature_names"],
        "component_names": values["component_names"],
        "state": best["state"],
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    jt.save(payload, str(args.model_output))
    report = {
        "kind": payload["kind"],
        "decision": decision,
        "selection_contract": (
            (f"fit on all labels from {list(args.fit_splits)} using epoch and scale frozen by the "
             "earlier temporal audit; in-sample metrics are diagnostic only; "
             "no later split was loaded")
            if args.full_fit else
            (f"fit on earliest {args.temporal_fit_fraction:.0%} of {list(args.fit_splits)} "
             f"by query time; select epoch and scale on later "
             f"{1.0 - args.temporal_fit_fraction:.0%}; no later split loaded")
            if args.split_strategy == "temporal" else
            ("fit on deterministic meta_train 60%; select epoch and scale on "
             "disjoint meta_train 40%; validation and confirmation not loaded")
        ),
        "data_sha256": EXPECTED_DATA_SHA256,
        "ensemble_report": str(args.ensemble_report),
        "ensemble_report_sha256": sha256(args.ensemble_report),
        "config": vars(args) | {
            "data": str(args.data),
            "code": str(args.code),
            "ensemble_report": str(args.ensemble_report),
            "output": str(args.output),
            "model_output": str(args.model_output),
        },
        "rows": {
            "total": int(len(values["labels"])),
            "fit_partition": int(fit_mask.sum()),
            "optimized": int(optimize_mask.sum()),
            "internal_audit": int(audit_mask.sum()),
        },
        "fit_splits": list(args.fit_splits),
        "feature_names": values["feature_names"],
        "selected": {
            "epoch": int(best["epoch"]),
            "scale": float(best["scale"]),
            "internal_audit": best["metrics"],
        },
        "history": history,
        "checks": checks,
        "model_sha256": sha256(args.model_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if decision in {"PROMISING", "TRAINED_FROZEN"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
