#!/usr/bin/env python3
"""Train a leakage-controlled candidate residual ranker on top of frozen D3 c6.

The pilot stage uses only meta_train.  A deterministic 60/40 row split is used
for fitting versus epoch/scale selection; validation and confirmation are never
loaded.  A later formal stage must freeze the selected hyperparameters before
touching those two holdouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

os.environ.update({
    "use_cutt": "0",
    "use_cutlass": "0",
    "use_nccl": "0",
    "use_mkl": "0",
})
os.environ.setdefault("JT_USE_CUDA", "1")


EXPECTED_DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
SCALE_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + np.float32(1e-6)
    )


def rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.float32)
    rows = np.arange(len(values))[:, None]
    ranks[rows, order] = np.linspace(
        0.0, 1.0, values.shape[1], dtype=np.float32
    )[None, :]
    return ranks


def component_features(components: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    names = []
    output = []
    for index in range(len(components)):
        score = np.asarray(components[index], dtype=np.float32)
        normalized = qnorm(score)
        names.extend(
            (
                f"component_{index}_qnorm",
                f"component_{index}_rank",
                f"component_{index}_winner_gap",
            )
        )
        output.extend(
            (
                normalized,
                rank_percentile(score),
                normalized - normalized.max(axis=1, keepdims=True),
            )
        )
    return names, output


def add_structural(
    feature_names: list[str],
    feature_values: list[np.ndarray],
    name: str,
    raw: np.ndarray,
) -> None:
    feature_names.append(name)
    feature_values.append(qnorm(np.log1p(np.maximum(raw, 0))))


def build_c6_and_features(
    *,
    core,
    d3,
    multiscale,
    near,
    scene,
    model_dirs,
    active,
    ensemble_weights,
    history0,
    segments,
    max_node,
    use_src_freq,
    pool,
    freq,
    src_freq,
    groups: int,
    seed: int,
    batch: int,
    split: str = "meta_train",
) -> dict:
    import d3_outer_ring_gate as common

    src, query_time, candidates, labels = core.sample_segment(
        segments, pool, split, groups, seed
    )
    history = core.segment_history(history0, segments, split)
    component_names, components, _ = core.component_scores(
        scene="dataset3",
        model_dirs=model_dirs,
        history=history,
        max_node=max_node,
        use_src_freq=use_src_freq,
        freq=freq,
        src_freq=src_freq,
        src=src,
        time=query_time,
        candidates=candidates,
        labels=labels,
        batch=batch,
    )
    base = core.mixed_score(
        ensemble_weights,
        components[[component_names.index(name) for name in active]],
    ).astype(np.float32)

    pool_src, pool_time, pool_candidates, _ = core.sample_segment(
        segments,
        pool,
        split,
        len(segments[split]),
        seed + 10_000,
    )
    index = near.NearTimeIndex(pool_src, pool_time, pool_candidates)
    cross_exact = index.support(src, query_time, candidates, 0)
    source_exact = index.source_support(src, query_time, candidates, 0)
    source_300_past, source_300_future = multiscale.directional_support(
        index, src, query_time, candidates, 300, True
    )
    # directional_support excludes the exact timestamp, so past + future is
    # exactly source_support(window=300) - source_support(window=0), matching
    # the frozen c2/c6 implementation.
    source_session = source_300_past + source_300_future
    reference_source_session = (
        index.source_support(src, query_time, candidates, 300) - source_exact
    )
    if not np.array_equal(source_session, reference_source_session):
        raise ValueError("directional session support differs from frozen c2")

    c3 = base + np.float32(common.C2_CROSS_WEIGHT) * qnorm(np.log1p(cross_exact))
    c3 += np.float32(common.C2_SESSION_WEIGHT) * qnorm(np.log1p(source_session))
    seen = d3.pair_seen(history, src, candidates)
    for policy_name, weight in common.C3_POLICY.items():
        prefix, direction, window = policy_name.split("_")
        past, future = multiscale.directional_support(
            index,
            src,
            query_time,
            candidates,
            int(window[:-1]),
            prefix == "session",
        )
        support = past if direction == "past" else future
        c3 += np.where(
            ~seen,
            np.float32(weight) * qnorm(np.log1p(support)),
            0.0,
        )

    source_900_past, source_900_future = multiscale.directional_support(
        index, src, query_time, candidates, 900, True
    )
    source_86400_past, source_86400_future = multiscale.directional_support(
        index, src, query_time, candidates, 86_400, True
    )
    ring_past = source_86400_past - source_900_past
    ring_future = source_86400_future - source_900_future
    ring_sum = ring_past + ring_future
    directional = np.tensordot(
        common.C5_POLICY,
        np.stack((qnorm(np.log1p(ring_past)), qnorm(np.log1p(ring_future)), qnorm(np.log1p(ring_sum)))),
        axes=(0, 0),
    ).astype(np.float32)
    unique_gate = common.unique_maximum(ring_sum, seen)
    c6 = c3 + np.where(unique_gate, directional, 0.0)
    eligible = np.where(~seen, ring_sum, -1)
    maximum = eligible.max(axis=1)
    tied = (~seen) & (eligible == maximum[:, None])
    tie_count = tied.sum(axis=1)
    tie_rows = (maximum > 0) & (tie_count >= 2) & ~unique_gate.any(axis=1)
    c6 += np.where(unique_gate, directional, 0.0)
    c6 += np.where(
        tied & tie_rows[:, None],
        np.float32(0.2) * qnorm(np.log1p(ring_sum)),
        0.0,
    )

    feature_names, feature_values = component_features(components)
    feature_names.extend(("c6_qnorm", "c6_rank", "c6_winner_gap", "pair_seen"))
    c6_normalized = qnorm(c6)
    feature_values.extend(
        (
            c6_normalized,
            rank_percentile(c6),
            c6_normalized - c6_normalized.max(axis=1, keepdims=True),
            seen.astype(np.float32),
        )
    )
    add_structural(feature_names, feature_values, "source_0_300", source_session)
    add_structural(
        feature_names,
        feature_values,
        "source_300_900",
        source_900_past + source_900_future - source_300_past - source_300_future,
    )
    source_300_900 = (
        source_900_past + source_900_future - source_300_past - source_300_future
    )
    source_3600_past, source_3600_future = multiscale.directional_support(
        index, src, query_time, candidates, 3_600, True
    )
    add_structural(
        feature_names,
        feature_values,
        "source_900_3600",
        source_3600_past + source_3600_future - source_900_past - source_900_future,
    )
    source_21600_past, source_21600_future = multiscale.directional_support(
        index, src, query_time, candidates, 21_600, True
    )
    add_structural(
        feature_names,
        feature_values,
        "source_3600_21600",
        source_21600_past + source_21600_future - source_3600_past - source_3600_future,
    )
    add_structural(
        feature_names,
        feature_values,
        "source_21600_86400",
        source_86400_past + source_86400_future - source_21600_past - source_21600_future,
    )
    add_structural(feature_names, feature_values, "cross_exact", cross_exact)
    cross_300_past, cross_300_future = multiscale.directional_support(
        index, src, query_time, candidates, 300, False
    )
    cross_900_past, cross_900_future = multiscale.directional_support(
        index, src, query_time, candidates, 900, False
    )
    add_structural(
        feature_names,
        feature_values,
        "cross_300_900",
        cross_900_past + cross_900_future - cross_300_past - cross_300_future,
    )
    cross_3600_past, cross_3600_future = multiscale.directional_support(
        index, src, query_time, candidates, 3_600, False
    )
    add_structural(
        feature_names,
        feature_values,
        "cross_900_3600",
        cross_3600_past + cross_3600_future - cross_900_past - cross_900_future,
    )

    features = np.stack(feature_values, axis=2).astype(np.float32, copy=False)
    return {
        "src": src,
        "time": query_time,
        "labels": labels,
        "base": c6.astype(np.float32, copy=False),
        "seen": seen,
        "source_300_900": source_300_900,
        "features": features,
        "feature_names": feature_names,
        "component_names": component_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--ensemble-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=30_000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--base-logit-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--freeze-final-epoch", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.model_output.exists():
        raise FileExistsError("refusing to overwrite v39 output")
    if sha256(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("official data hash differs")

    sys.path.insert(0, str(args.code.resolve()))
    sys.path.insert(0, str(args.code.parent.resolve()))
    import d3_cross_source_c2_audit as d3
    import d3_multiscale_craft_gate as multiscale
    import d3_near_time_audit as near
    import jittor as jt
    from jittor import nn
    from b_rank_a_port import ensemble_core as core
    import d3_outer_ring_gate as common

    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    _, model_dirs, active, ensemble_weights = d3.load_ensemble(args.ensemble_report)
    scene = core.scene_data(args.data, "dataset3")
    _, _, max_node, use_src_freq, pool, freq, src_freq, history0, segments = scene
    values = build_c6_and_features(
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
        groups=args.groups,
        seed=args.seed,
        batch=args.batch,
    )
    print(
        json.dumps(
            {
                "features_ready": True,
                "rows": len(values["labels"]),
                "feature_dim": values["features"].shape[-1],
                "c6_mrr": float(common.reciprocal_ranks(values["base"], values["labels"]).mean()),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    class ResidualRanker(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, 1)

        def execute(self, x):
            return self.fc3(nn.relu(self.fc2(nn.relu(self.fc1(x)))))

    def predict(net, row_ids: np.ndarray) -> np.ndarray:
        output = []
        net.eval()
        with jt.no_grad():
            for start in range(0, len(row_ids), args.batch):
                ids = row_ids[start : start + args.batch]
                part = values["features"][ids]
                correction = net(
                    jt.array(part.reshape(-1, part.shape[-1]))
                ).reshape((len(ids), 100))
                output.append(np.asarray(correction.data, dtype=np.float32))
        net.train()
        return np.concatenate(output)

    row_hash = (
        values["src"].astype(np.int64) * np.int64(1_000_003)
        + values["time"].astype(np.int64)
    )
    fit_mask = (row_hash % np.int64(10)) < 6
    audit_mask = ~fit_mask
    base_ranks = 1.0 / common.reciprocal_ranks(values["base"], values["labels"])
    anchor_mask = ((row_hash // np.int64(10)) % np.int64(4)) == 0
    optimize_mask = fit_mask & ((base_ranks > 1.0) | anchor_mask)
    fit_ids = np.flatnonzero(optimize_mask)
    audit_ids = np.flatnonzero(audit_mask)
    minimum_fit_rows = max(50, min(1_000, int(args.groups) // 10))
    minimum_audit_rows = max(100, min(1_000, int(args.groups) // 5))
    if len(fit_ids) < minimum_fit_rows or len(audit_ids) < minimum_audit_rows:
        raise ValueError("unexpectedly small internal split")

    net = ResidualRanker(values["features"].shape[-1], args.hidden)
    optimizer = nn.Adam(net.parameters(), lr=args.learning_rate, weight_decay=2e-5)
    rng = np.random.default_rng(args.seed)
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(fit_ids)
        losses = []
        net.train()
        for start in range(0, len(order), args.batch):
            ids = order[start : start + args.batch]
            part = values["features"][ids]
            correction = net(
                jt.array(part.reshape(-1, part.shape[-1]))
            ).reshape((len(ids), 100))
            logits = (
                np.float32(args.base_logit_scale) * jt.array(values["base"][ids])
                + correction
            )
            loss = nn.cross_entropy_loss(logits, jt.array(values["labels"][ids]))
            loss += np.float32(1e-4) * (correction * correction).mean()
            optimizer.step(loss)
            losses.append(float(loss.data[0]))

        raw_correction = predict(net, audit_ids)
        correction = qnorm(raw_correction)
        choices = []
        for scale in SCALE_GRID:
            candidate = values["base"][audit_ids] + np.float32(scale) * correction
            metrics = common.metrics(
                values["base"][audit_ids],
                candidate,
                values["labels"][audit_ids],
            )
            choices.append((metrics["delta"], -metrics["negative_row_rate"], -scale, scale, metrics))
        _, _, _, scale, metrics = max(choices)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "scale": float(scale),
            "internal_audit": metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (metrics["delta"], -metrics["negative_row_rate"], -epoch)
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "epoch": epoch,
                "scale": float(scale),
                "metrics": metrics,
                "state": {
                    name: np.asarray(value.data).copy()
                    for name, value in net.state_dict().items()
                },
            }

    if args.freeze_final_epoch:
        final_row = history[-1]
        best = {
            "key": (
                final_row["internal_audit"]["delta"],
                -final_row["internal_audit"]["negative_row_rate"],
                -final_row["epoch"],
            ),
            "epoch": int(final_row["epoch"]),
            "scale": float(final_row["scale"]),
            "metrics": final_row["internal_audit"],
            "state": {
                name: np.asarray(value.data).copy()
                for name, value in net.state_dict().items()
            },
        }

    checks = {
        "internal_delta_at_least_0_005": best["metrics"]["delta"] >= 0.005,
        "internal_above_two_se": best["metrics"]["delta"] > 2 * best["metrics"]["delta_se"],
        "internal_negative_rows_below_0_01": best["metrics"]["negative_row_rate"] < 0.01,
    }
    decision = "PROMISING" if all(checks.values()) else "NO_GO"
    model_payload = {
        "kind": "d3_c6_candidate_residual_ranker_v39_pilot",
        "input_dim": int(values["features"].shape[-1]),
        "hidden": int(args.hidden),
        "epoch": int(best["epoch"]),
        "scale": float(best["scale"]),
        "base_logit_scale": float(args.base_logit_scale),
        "feature_names": values["feature_names"],
        "component_names": values["component_names"],
        "state": best["state"],
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    jt.save(model_payload, str(args.model_output))
    report = {
        "kind": "d3_c6_candidate_residual_ranker_v39_pilot",
        "decision": decision,
        "selection_contract": "fit on deterministic 60% of meta_train; select epoch and scale only on remaining 40%; validation and confirmation not loaded",
        "data_sha256": EXPECTED_DATA_SHA256,
        "ensemble_report": str(args.ensemble_report),
        "ensemble_report_sha256": sha256(args.ensemble_report),
        "config": {
            "groups": int(args.groups),
            "batch": int(args.batch),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "learning_rate": float(args.learning_rate),
            "base_logit_scale": float(args.base_logit_scale),
            "seed": int(args.seed),
            "freeze_final_epoch": bool(args.freeze_final_epoch),
        },
        "rows": {
            "total": int(len(values["labels"])),
            "fit_partition": int(fit_mask.sum()),
            "optimized": int(optimize_mask.sum()),
            "internal_audit": int(audit_mask.sum()),
        },
        "feature_names": values["feature_names"],
        "selected": {
            "epoch": int(best["epoch"]),
            "scale": float(best["scale"]),
            "internal_audit": best["metrics"],
        },
        "epoch_contract": (
            "final epoch frozen before training"
            if args.freeze_final_epoch
            else "epoch selected on internal audit"
        ),
        "checks": checks,
        "history": history,
        "model_sha256": sha256(args.model_output),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if decision == "PROMISING" else 3


if __name__ == "__main__":
    raise SystemExit(main())
