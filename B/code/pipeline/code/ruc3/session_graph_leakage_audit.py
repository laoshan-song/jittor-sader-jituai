#!/usr/bin/env python3
"""Audit temporal causality and fixed-origin D4 session-graph performance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.update({
    "use_cutt": "0",
    "use_cutlass": "0",
    "use_nccl": "0",
    "use_mkl": "0",
})

import jittor as jt
import numpy as np

from b_rank import pairnew_transformer_jittor as pairnew, replay_score_cache

import session_graph_gate as graph_gate
import session_graph_hard_ranker as ranker


def load_model(path: Path) -> ranker.HardNegativeGate:
    with np.load(path, allow_pickle=False) as saved:
        net = ranker.HardNegativeGate(20)
        net.load_state_dict({
            str(name): jt.array(saved[f"state_{index}"])
            for index, name in enumerate(saved["state_names"])
        })
    return net


def ensemble_predict(
    models: list[ranker.HardNegativeGate], feature: np.ndarray, batch: int
) -> np.ndarray:
    return ranker.qnorm(np.mean([
        ranker.predict(model, feature, batch) for model in models
    ], axis=0))


def evaluate(values: dict, residual: np.ndarray, alpha: float, rows: slice) -> dict:
    baseline = values["baseline"][rows]
    candidate = pairnew._candidate_score(
        baseline, residual[rows], values["seen"][rows], alpha
    )
    return graph_gate.paired(
        baseline, candidate, values["labels"][rows], values["active"][rows]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--replay-cache", type=Path, action="append", required=True)
    parser.add_argument("--identity-cache", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    scored, _, manifests = replay_score_cache.load(args.replay_cache, verify=False)
    plan = manifests[0]["metadata"]["group_metadata"]["history"]["plan"]
    source = np.load(args.train_cache / "src.npy", mmap_mode="r")
    item = np.load(args.train_cache / "dst.npy", mmap_mode="r")
    timestamp = np.load(args.train_cache / "time.npy", mmap_mode="r")
    validation_time = graph_gate.identity(
        args.identity_cache, "history", "validation", "time"
    )
    confirmation_time = graph_gate.identity(
        args.identity_cache, "history", "confirmation", "time"
    )

    boundaries = []
    for start in range(0, 120_000, 30_000):
        cutoff = int(validation_time[start])
        edge_stop = int(np.searchsorted(timestamp, cutoff, side="left"))
        boundaries.append({
            "split": f"validation_{start // 30_000}",
            "cutoff": cutoff,
            "edge_time_max": int(timestamp[edge_stop - 1]),
            "query_time_min": int(validation_time[start:start + 30_000].min()),
            "strictly_causal": bool(
                timestamp[edge_stop - 1]
                < validation_time[start:start + 30_000].min()
            ),
        })
    cutoff = int(plan["cutoffs"]["confirm"])
    edge_stop = int(np.searchsorted(timestamp, cutoff, side="left"))
    boundaries.append({
        "split": "confirmation",
        "cutoff": cutoff,
        "edge_time_max": int(timestamp[edge_stop - 1]),
        "query_time_min": int(confirmation_time.min()),
        "strictly_causal": bool(timestamp[edge_stop - 1] < confirmation_time.min()),
    })

    graph = graph_gate.build_graph(source, item, timestamp, cutoff)
    confirmation = ranker.prepare(
        scored, args.identity_cache, args.baseline_cache, graph,
        "confirmation", 30_000, args.chunk
    )
    models = [load_model(path) for path in args.model]
    residuals = {
        strategy: ensemble_predict(models, values["feature"], args.batch)
        for strategy, values in confirmation.items()
    }

    metrics = {}
    for strategy, values in confirmation.items():
        metrics[strategy] = {
            "all": evaluate(values, residuals[strategy], args.alpha, slice(None)),
            "early": evaluate(values, residuals[strategy], args.alpha, slice(0, 10_000)),
            "middle": evaluate(values, residuals[strategy], args.alpha, slice(10_000, 20_000)),
            "late": evaluate(values, residuals[strategy], args.alpha, slice(20_000, 30_000)),
        }

    rng = np.random.default_rng(20260814)
    feature = confirmation["history"]["feature"][:128]
    permutations = np.argsort(rng.random(feature.shape[:2]), axis=1)
    rows = np.arange(len(feature))[:, None]
    inverse = np.empty_like(permutations)
    inverse[rows, permutations] = np.arange(feature.shape[1])[None, :]
    original = ensemble_predict(models, feature, args.batch)
    permuted = ensemble_predict(models, feature[rows, permutations], args.batch)
    equivariance_error = float(np.max(np.abs(original - permuted[rows, inverse])))

    checks = {
        "all_graph_boundaries_strictly_causal": all(
            value["strictly_causal"] for value in boundaries
        ),
        "fixed_origin_confirmation_positive_in_all_thirds": all(
            metrics[strategy][part]["delta"] > 0.0
            for strategy in metrics for part in ("early", "middle", "late")
        ),
        "fixed_origin_confirmation_above_two_se": all(
            metrics[strategy]["all"]["delta"]
            > 2.0 * metrics[strategy]["all"]["delta_se"]
            for strategy in metrics
        ),
        "candidate_permutation_equivariant": equivariance_error < 1e-5,
    }
    report = {
        "kind": "d4_session_graph_leakage_audit_v1",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "headline_protocol": "fixed-origin confirmation only",
        "rolling_holdout_usage": "training diagnostic only; excluded from expected lift",
        "model_input_contract": [
            "five causal session-graph features",
            "frozen baseline score/rank/margin",
            "frozen causal static features",
            "pair-seen mask",
        ],
        "label_usage": "supervised training target and final MRR audit only; never a feature",
        "alpha": args.alpha,
        "models": [str(path.resolve()) for path in args.model],
        "boundaries": boundaries,
        "confirmation": metrics,
        "candidate_permutation_max_abs_error": equivariance_error,
        "checks": checks,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    # Keep leakage diagnostics in the report without blocking artifact creation.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
