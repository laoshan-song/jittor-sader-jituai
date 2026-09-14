#!/usr/bin/env python3
"""Fit a causal candidate-level ensemble of frozen D4 temporal checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from . import data_features, temporal_attention_jittor, temporal_history, temporal_validate
except ImportError:
    import data_features
    import temporal_attention_jittor
    import temporal_history
    import temporal_validate


EXPECTED_DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
SEEDS = (20260810, 20260811, 20260812)
GROUP_SIZES = {"train": 100000, "valid": 30000, "confirm": 30000}
PAIR_SEEN_INDEX = data_features.FEATURE_NAMES.index("pair_seen")
EXPECTED_MEAN_MRR = {
    "validation": 0.304189442420445,
    "confirmation": 0.2956817768672844,
}
REPRODUCTION_ATOL = 5e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score_segment(cache, group, history, models, batch_rows: int):
    store = cache.feature_store(group.cutoff)
    scores = np.empty((len(models), group.rows, group.candidate_count), np.float32)
    labels = np.empty(group.rows, np.int64)
    candidate_seen = np.empty((group.rows, group.candidate_count), bool)
    segment_parts: dict[str, list[np.ndarray]] = {}
    offset = 0
    for batch in group.iter_batches(batch_rows=batch_rows):
        temporal = history.lookup(batch.src, batch.time, batch.candidates)
        features = store.features(batch.src, batch.time, batch.candidates)
        stop = offset + len(batch.src)
        labels[offset:stop] = batch.labels
        candidate_seen[offset:stop] = features[:, :, PAIR_SEEN_INDEX] > 0.0
        for index, model in enumerate(models):
            scores[index, offset:stop] = temporal_attention_jittor.predict_scores(
                model,
                temporal.source_indices,
                temporal.candidate_indices,
                temporal.history_item_indices,
                temporal.log_time_deltas,
                features=features,
                batch_size=batch_rows,
            )
        for name, mask in store.evaluation_segments(
            batch.src, batch.time, batch.candidates, batch.labels
        ).items():
            segment_parts.setdefault(name, []).append(np.asarray(mask, dtype=bool))
        offset = stop
    if offset != group.rows:
        raise ValueError("candidate group row count changed")
    segments = {name: np.concatenate(parts) for name, parts in segment_parts.items()}
    return scores, labels, candidate_seen, segments


def metrics(scores, labels, segments):
    return data_features.ranking_metrics(scores, labels, segments=segments)


def segment_mrr(report: dict, name: str) -> float:
    if name == "overall":
        return float(report["mrr"])
    value = report["segments"][name]["mrr"]
    return float(value) if value is not None else float("nan")


def fit_alphas(best_score, mean_score, candidate_seen, labels):
    """Fit the two candidate gates with a coarse-to-fine validation search."""
    seen_alpha = 0.0
    new_alpha = 0.0
    stages = ((1.0, 0.1), (0.1, 0.02), (0.02, 0.005))
    trace = []
    for radius, step in stages:
        seen_values = np.arange(
            max(0.0, seen_alpha - radius),
            min(1.0, seen_alpha + radius) + step / 2.0,
            step,
        )
        new_values = np.arange(
            max(0.0, new_alpha - radius),
            min(1.0, new_alpha + radius) + step / 2.0,
            step,
        )
        candidates = []
        for seen_value in seen_values:
            for new_value in new_values:
                alpha = np.where(candidate_seen, seen_value, new_value).astype(np.float32)
                mixed = best_score + alpha * (mean_score - best_score)
                value = data_features.ranking_metrics(mixed, labels)["mrr"]
                candidates.append(
                    (
                        float(value),
                        -float(seen_value + new_value),
                        -float(seen_value),
                        float(seen_value),
                        float(new_value),
                    )
                )
        value, _, _, seen_alpha, new_alpha = max(candidates)
        trace.append(
            {
                "radius": radius,
                "step": step,
                "seen_alpha": seen_alpha,
                "new_alpha": new_alpha,
                "validation_mrr": value,
            }
        )
    return seen_alpha, new_alpha, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=512)
    args = parser.parse_args()

    data = args.data.resolve()
    output = args.output.resolve()
    checkpoints = [path.resolve() for path in args.checkpoints]
    if output.exists():
        raise FileExistsError(f"refusing output reuse: {output}")
    if sha256(data) != EXPECTED_DATA_SHA256:
        raise ValueError("official data hash differs")
    if args.batch_rows < 1:
        raise ValueError("batch rows must be positive")

    temporal_attention_jittor.configure_cuda()
    cache = data_features.BDataCache.build_or_open(
        data, "dataset4", args.cache_dir.resolve(), verify_hash=True
    )
    groups = data_features.build_split1_groups(
        cache,
        seed=20260810,
        sizes=GROUP_SIZES,
        batch_rows=4096,
        negative_strategy="history",
    )
    valid_history = temporal_validate._history_index(
        cache,
        history_size=32,
        scene="dataset4",
        cutoff=int(groups.plan.cutoffs["valid"]),
    )
    confirm_history = temporal_validate._history_index(
        cache,
        history_size=32,
        scene="dataset4",
        cutoff=int(groups.plan.cutoffs["confirm"]),
        vocabulary=valid_history.vocabulary,
    )

    models = []
    configs = []
    for seed, checkpoint in zip(SEEDS, checkpoints):
        if checkpoint.name != f"seed{seed}.npz":
            raise ValueError(f"checkpoint order differs at seed {seed}: {checkpoint}")
        model, config = temporal_attention_jittor.load_checkpoint(checkpoint)
        if not config.get("static_context_pair_seen_only"):
            raise ValueError(f"checkpoint is not pair-seen-context gated: {checkpoint}")
        models.append(model)
        configs.append(config)
    if len({json.dumps(config, sort_keys=True) for config in configs}) != 1:
        raise ValueError("checkpoint architectures differ")

    scored = {
        "validation": score_segment(
            cache, groups.valid, valid_history, models, int(args.batch_rows)
        ),
        "confirmation": score_segment(
            cache, groups.confirm, confirm_history, models, int(args.batch_rows)
        ),
    }
    validation_scores, validation_labels, validation_seen, validation_segments = scored[
        "validation"
    ]
    individual_validation = [
        metrics(score, validation_labels, validation_segments) for score in validation_scores
    ]
    best_index = max(range(len(models)), key=lambda index: individual_validation[index]["mrr"])
    best_score = validation_scores[best_index]
    mean_score = validation_scores.mean(axis=0)

    seen_alpha, new_alpha, alpha_search = fit_alphas(
        best_score, mean_score, validation_seen, validation_labels
    )

    reports = {}
    for name, (component_scores, labels, candidate_seen, segments) in scored.items():
        individual = {
            str(seed): metrics(score, labels, segments)
            for seed, score in zip(SEEDS, component_scores)
        }
        mean = metrics(component_scores.mean(axis=0), labels, segments)
        alpha = np.where(candidate_seen, seen_alpha, new_alpha).astype(np.float32)
        gated_score = component_scores[best_index] + alpha * (
            component_scores.mean(axis=0) - component_scores[best_index]
        )
        reports[name] = {
            "individual": individual,
            "mean": mean,
            "gated": metrics(gated_score, labels, segments),
        }

    reproduction = {
        name: {
            "expected_mean_mrr": EXPECTED_MEAN_MRR[name],
            "actual_mean_mrr": float(reports[name]["mean"]["mrr"]),
            "absolute_error": abs(
                float(reports[name]["mean"]["mrr"]) - EXPECTED_MEAN_MRR[name]
            ),
        }
        for name in EXPECTED_MEAN_MRR
    }
    protocol_checks = {
        f"{name}_mean_reproduced": value["absolute_error"] <= REPRODUCTION_ATOL
        for name, value in reproduction.items()
    }
    confirmation = reports["confirmation"]
    checks = {}
    for stratum, tolerance in (
        ("overall", 0.0005),
        ("source_hot", 0.005),
        ("pair_seen", 0.005),
        ("pair_new", 0.005),
    ):
        best = max(segment_mrr(value, stratum) for value in confirmation["individual"].values())
        checks[stratum] = segment_mrr(confirmation["gated"], stratum) >= best - tolerance
    checks["multi_model_active"] = seen_alpha > 0.0 or new_alpha > 0.0
    checks["validation_noninferior"] = reports["validation"]["gated"]["mrr"] >= max(
        value["mrr"] for value in reports["validation"]["individual"].values()
    )

    report = {
        "kind": "b_rank_temporal_ensemble_v1",
        "decision": "PASS" if all(protocol_checks.values()) and all(checks.values()) else "NO_GO",
        "scene": "dataset4",
        "data_sha256": EXPECTED_DATA_SHA256,
        "group_metadata": groups.metadata,
        "best_seed": SEEDS[best_index],
        "seen_alpha": float(seen_alpha),
        "new_alpha": float(new_alpha),
        "candidate_gate": "FeatureStore pair_seen at the segment cutoff",
        "alpha_search": alpha_search,
        "segments": reports,
        "reproduction": reproduction,
        "protocol_checks": protocol_checks,
        "decision_checks": checks,
        "checkpoints": [
            {"seed": seed, "path": str(path), "sha256": sha256(path)}
            for seed, path in zip(SEEDS, checkpoints)
        ],
        "source_hashes": {
            Path(module.__file__).name: sha256(Path(module.__file__).resolve())
            for module in (data_features, temporal_attention_jittor, temporal_history, temporal_validate)
        }
        | {Path(__file__).name: sha256(Path(__file__).resolve())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if report["decision"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
