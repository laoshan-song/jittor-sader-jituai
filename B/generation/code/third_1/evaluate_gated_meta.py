#!/usr/bin/env python3
"""Evaluate row-level confidence gates for a trained meta ranker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


STRATEGIES = ("history", "test_pool")
SCALES = (0.02, 0.05, 0.08, 0.12, 0.16, 0.20)
MODES = ("all", "pair_seen", "pair_new")
QUANTILES = (0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9)


def duplicate_mask(candidates: np.ndarray, chunk: int = 8192) -> np.ndarray:
    out = np.empty(len(candidates), dtype=bool)
    for start in range(0, len(candidates), chunk):
        stop = min(len(candidates), start + chunk)
        out[start:stop] = np.any(
            np.diff(np.sort(candidates[start:stop], axis=1), axis=1) == 0,
            axis=1,
        )
    return out


def compose(
    baseline: np.ndarray,
    residual: np.ndarray,
    seen: np.ndarray,
    duplicate: np.ndarray,
    scale: float,
    mode: str,
) -> np.ndarray:
    correction = residual.copy()
    if mode == "pair_seen":
        correction[~seen] = 0.0
    elif mode == "pair_new":
        correction[seen] = 0.0
    elif mode != "all":
        raise ValueError(mode)
    candidate = baseline + np.float32(scale) * correction
    candidate[duplicate] = baseline[duplicate]
    return candidate


def row_scores(baseline: np.ndarray, candidate: np.ndarray, residual: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.arange(len(baseline))
    base_top = np.argmax(baseline, axis=1)
    cand_top = np.argmax(candidate, axis=1)
    sorted_candidate = np.sort(candidate, axis=1)
    sorted_residual = np.sort(residual, axis=1)
    return {
        "none": np.full(len(baseline), np.inf, dtype=np.float32),
        "top_margin": candidate[rows, cand_top] - candidate[rows, base_top],
        "candidate_gap": sorted_candidate[:, -1] - sorted_candidate[:, -2],
        "residual_gap": sorted_residual[:, -1] - sorted_residual[:, -2],
        "max_abs_residual": np.max(np.abs(residual), axis=1),
    }


def apply_row_gate(
    baseline: np.ndarray,
    candidate: np.ndarray,
    gate_value: np.ndarray,
    threshold: float,
    keep_unchanged_top1: bool,
) -> tuple[np.ndarray, float, float]:
    base_top = np.argmax(baseline, axis=1)
    cand_top = np.argmax(candidate, axis=1)
    active = gate_value >= threshold
    if keep_unchanged_top1:
        active |= cand_top == base_top
    output = baseline.copy()
    output[active] = candidate[active]
    top1_rate = float(np.mean(cand_top[active] != base_top[active])) if np.any(active) else 0.0
    return output, float(np.mean(active)), top1_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--replay-cache", type=Path, required=True)
    parser.add_argument("--identity-cache", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=70000)
    parser.add_argument("--selection-rows", type=int, default=30000)
    parser.add_argument("--predict-batch", type=int, default=768)
    parser.add_argument("--use-cuda", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.code_root))
    from deployment_utils import sha256_file
    from infer_hierarchy_jittor import load_model, predict_residual
    from train_hierarchy_jittor import metrics

    import jittor as jt

    jt.flags.use_cuda = args.use_cuda
    model, feature_count, hidden = load_model(args.model)

    def load(prefix: str, strategy: str, row_slice: slice) -> dict[str, np.ndarray]:
        return {
            "features": np.load(args.feature_cache / f"{prefix}.npy", mmap_mode="r")[row_slice],
            "baseline": np.asarray(np.load(args.baseline_cache / f"{prefix}.npy", mmap_mode="r")[row_slice]),
            "labels": np.asarray(np.load(args.replay_cache / strategy / f"{prefix}__labels.npy", mmap_mode="r")[row_slice]),
            "seen": np.asarray(np.load(args.replay_cache / strategy / f"{prefix}__seen.npy", mmap_mode="r")[row_slice]),
            "candidates": np.asarray(np.load(args.identity_cache / f"{prefix}__candidates.npy", mmap_mode="r")[row_slice]),
        }

    residual_cache: dict[tuple[str, str, str], tuple[dict[str, np.ndarray], np.ndarray]] = {}
    for split_name, source_split, row_slice in (
        ("selection", "validation", slice(args.train_rows, args.train_rows + args.selection_rows)),
        ("holdout", "validation", slice(args.train_rows + args.selection_rows, None)),
        ("confirmation", "confirmation", slice(None)),
    ):
        for strategy in STRATEGIES:
            prefix = f"{strategy}__{source_split}"
            arrays = load(prefix, strategy, row_slice)
            residual, error = predict_residual(
                [model], arrays["features"], batch_rows=args.predict_batch, no_recent_weight=0.20
            )
            if error > 1e-5:
                raise ValueError(f"permutation error {error}")
            residual_cache[(split_name, strategy, source_split)] = (arrays, residual)

    trials = []
    selection_arrays = [residual_cache[("selection", strategy, "validation")] for strategy in STRATEGIES]
    for mode in MODES:
        for scale in SCALES:
            per_strategy_values = []
            for arrays, residual in selection_arrays:
                duplicate = duplicate_mask(arrays["candidates"])
                candidate = compose(arrays["baseline"], residual, arrays["seen"], duplicate, scale, mode)
                scores = row_scores(arrays["baseline"], candidate, residual)
                per_strategy_values.append((arrays, candidate, scores))
            for score_name in ("none", "top_margin", "candidate_gap", "residual_gap", "max_abs_residual"):
                joined = np.concatenate([values[2][score_name] for values in per_strategy_values])
                finite = joined[np.isfinite(joined)]
                thresholds = [float("-inf")] if score_name == "none" else [
                    float(np.quantile(finite, q)) for q in QUANTILES
                ]
                for threshold in thresholds:
                    for keep_same in (False, True):
                        if score_name == "none" and keep_same:
                            continue
                        rows = []
                        for arrays, candidate, scores in per_strategy_values:
                            gated, active_rate, active_top1_rate = apply_row_gate(
                                arrays["baseline"], candidate, scores[score_name], threshold, keep_same
                            )
                            rows.append({
                                **metrics(arrays["baseline"], gated, arrays["labels"]),
                                "active_rate": active_rate,
                                "active_top1_change_rate": active_top1_rate,
                            })
                        trials.append({
                            "mode": mode,
                            "scale": scale,
                            "score": score_name,
                            "threshold": threshold,
                            "keep_same_top1": keep_same,
                            "mean_delta": float(np.mean([row["delta"] for row in rows])),
                            "max_negative_rate": float(max(row["negative_row_rate"] for row in rows)),
                            "mean_active_rate": float(np.mean([row["active_rate"] for row in rows])),
                            "strategies": dict(zip(STRATEGIES, rows)),
                        })
    selected = max(
        trials,
        key=lambda row: (
            row["mean_delta"] - 0.01 * max(0.0, row["max_negative_rate"] - 0.20),
            -row["max_negative_rate"],
            -row["mean_active_rate"],
        ),
    )

    evaluations = {"selection": selected["strategies"]}
    for split_name, source_split in (("holdout", "validation"), ("confirmation", "confirmation")):
        evaluations[split_name] = {}
        for strategy in STRATEGIES:
            arrays, residual = residual_cache[(split_name, strategy, source_split)]
            duplicate = duplicate_mask(arrays["candidates"])
            candidate = compose(arrays["baseline"], residual, arrays["seen"], duplicate, selected["scale"], selected["mode"])
            values = row_scores(arrays["baseline"], candidate, residual)
            gated, active_rate, active_top1_rate = apply_row_gate(
                arrays["baseline"], candidate, values[selected["score"]],
                selected["threshold"], selected["keep_same_top1"],
            )
            evaluations[split_name][strategy] = {
                **metrics(arrays["baseline"], gated, arrays["labels"]),
                "active_rate": active_rate,
                "active_top1_change_rate": active_top1_rate,
            }

    report = {
        "kind": "row_confidence_gate_for_meta_ranker_v1",
        "decision": "PROMOTE" if all(
            evaluations[split][strategy]["delta"] > 0.0
            for split in ("holdout", "confirmation")
            for strategy in STRATEGIES
        ) else "NO_GO",
        "feature_count": feature_count,
        "hidden": hidden,
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "selected": {k: v for k, v in selected.items() if k != "strategies"},
        "evaluations": evaluations,
        "top_trials": sorted(trials, key=lambda row: row["mean_delta"], reverse=True)[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
