#!/usr/bin/env python3
"""Fit and audit a D4 ensemble across temporal and full-history MF models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import (
    d4_transition_research,
    data_features,
    implicit_mf_jittor,
    pairnew_transformer_jittor,
    pool_association,
    temporal_attention_jittor,
    temporal_history,
    verify_run,
)


PAIR_SEEN_INDEX = data_features.FEATURE_NAMES.index("pair_seen")
PAIR_LOG_COUNT_INDEX = data_features.FEATURE_NAMES.index("pair_log_count")
PAIR_RECENCY_INDEX = data_features.FEATURE_NAMES.index("pair_recency")
POOL_COMPONENT_NAMES = (
    "pool_source_log_count",
    "pool_source_excess",
    "pool_source_leaveone_lift",
    "pool_source_leaveone_deviance",
    "pool_source_leaveone_deviance_low",
    "pool_source_leaveone_deviance_mid",
    "pool_source_leaveone_deviance_high",
)
CALIBRATION_COMPONENT_NAMES = POOL_COMPONENT_NAMES[-5:]
TRANSITION_COMPONENT_NAMES = ("transition_last_count",)
INNOVATION_COMPONENT_NAMES = (*CALIBRATION_COMPONENT_NAMES, *TRANSITION_COMPONENT_NAMES)


def _is_innovation(name: str) -> bool:
    return name in INNOVATION_COMPONENT_NAMES or name.startswith("transition_mf_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _qnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def _mrr(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[np.arange(len(labels)), labels]
    columns = np.arange(scores.shape[1])[None, :]
    rank = 1 + (scores > positive[:, None]).sum(axis=1)
    rank += ((scores == positive[:, None]) & (columns < labels[:, None])).sum(axis=1)
    return float(np.mean(1.0 / rank))


def _tune_convex(components: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    individual = np.asarray([_mrr(score, labels) for score in components])
    weights = np.zeros(len(components), dtype=np.float64)
    weights[int(individual.argmax())] = 1.0
    score = np.tensordot(weights, components, axes=(0, 0))
    best = _mrr(score, labels)
    for values in (
        np.arange(0.05, 0.51, 0.05),
        np.arange(0.02, 0.21, 0.02),
        np.arange(0.01, 0.11, 0.01),
        np.arange(0.005, 0.051, 0.005),
        np.arange(0.002, 0.021, 0.002),
        np.arange(0.001, 0.011, 0.001),
    ):
        changed = True
        while changed:
            changed = False
            for index in range(len(components)):
                for alpha in values:
                    candidate = (1.0 - alpha) * score + alpha * components[index]
                    value = _mrr(candidate, labels)
                    if value > best + 1e-10:
                        weights *= 1.0 - alpha
                        weights[index] += alpha
                        score = candidate
                        best = value
                        changed = True
    return weights / weights.sum(), best


def _tune_innovations(
    components: np.ndarray,
    labels: np.ndarray,
    candidate_seen: np.ndarray,
    base_index: int,
    initial_weights: np.ndarray,
    innovation_indices: np.ndarray,
    seen_alpha: float,
    new_alpha: float,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    """Add innovations to the fitted control without re-solving its weights."""
    weights = np.asarray(initial_weights, dtype=np.float64).copy()
    mixed = np.tensordot(weights, components, axes=(0, 0))
    base = components[base_index]
    score = _gated_score(base, mixed, candidate_seen, seen_alpha, new_alpha)
    best = _mrr(score, labels)
    trace: list[dict[str, float]] = []
    for values in (
        np.arange(0.05, 0.51, 0.05),
        np.arange(0.02, 0.21, 0.02),
        np.arange(0.01, 0.11, 0.01),
        np.arange(0.005, 0.051, 0.005),
        np.arange(0.002, 0.021, 0.002),
        np.arange(0.001, 0.011, 0.001),
    ):
        changed = True
        while changed:
            changed = False
            for index in innovation_indices:
                for alpha in values:
                    candidate_mixed = (1.0 - alpha) * mixed + alpha * components[index]
                    candidate = _gated_score(
                        base,
                        candidate_mixed,
                        candidate_seen,
                        seen_alpha,
                        new_alpha,
                    )
                    value = _mrr(candidate, labels)
                    if value > best + 1e-10:
                        weights *= 1.0 - alpha
                        weights[index] += alpha
                        mixed = candidate_mixed
                        best = value
                        trace.append(
                            {
                                "component_index": int(index),
                                "alpha": float(alpha),
                                "mrr": float(value),
                            }
                        )
                        changed = True
    return weights / weights.sum(), best, trace


def _fit_gate(
    base: np.ndarray,
    mixed: np.ndarray,
    candidate_seen: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, list[dict[str, float]]]:
    seen_alpha = 0.0
    new_alpha = 0.0
    trace: list[dict[str, float]] = []
    for radius, step in ((1.0, 0.1), (0.1, 0.02), (0.02, 0.005)):
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
        for seen in seen_values:
            for new in new_values:
                alpha = np.where(candidate_seen, seen, new).astype(np.float32)
                value = _mrr(base + alpha * (mixed - base), labels)
                candidates.append((value, -(seen + new), -seen, seen, new))
        value, _, _, seen_alpha, new_alpha = max(candidates)
        trace.append(
            {
                "radius": float(radius),
                "step": float(step),
                "seen_alpha": float(seen_alpha),
                "new_alpha": float(new_alpha),
                "mrr": float(value),
            }
        )
    return float(seen_alpha), float(new_alpha), trace


def _candidate_counts(
    candidates: np.ndarray, ids: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    flat = np.asarray(candidates).reshape(-1)
    positions = np.searchsorted(ids, flat)
    inside = positions < len(ids)
    matched = np.zeros(len(flat), dtype=bool)
    matched[inside] = ids[positions[inside]] == flat[inside]
    output = np.zeros(len(flat), dtype=np.float32)
    output[matched] = np.log1p(np.asarray(counts[positions[matched]], dtype=np.float32))
    return output.reshape(candidates.shape)


def _pool_source_components(
    group: Any, test_ids: np.ndarray, test_counts: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Count candidate recurrence under each source in the unlabeled pool."""
    source = np.asarray(group.src, dtype=np.uint64)
    candidates = np.asarray(group.candidates, dtype=np.uint32)
    keys = pool_association.source_candidate_keys(source, candidates)
    flat = keys.reshape(-1)
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    source_count = counts[inverse].astype(np.float32).reshape(candidates.shape)
    del keys, flat, inverse, counts

    positions = np.searchsorted(test_ids, candidates.reshape(-1))
    inside = positions < len(test_ids)
    matched = np.zeros(len(positions), dtype=bool)
    matched[inside] = test_ids[positions[inside]] == candidates.reshape(-1)[inside]
    probability = np.zeros(len(positions), dtype=np.float32)
    probability[matched] = (
        np.asarray(test_counts[positions[matched]], dtype=np.float32)
        / float(np.asarray(test_counts, dtype=np.float64).sum())
    )
    probability = probability.reshape(candidates.shape)
    source_ids, source_inverse, source_rows = np.unique(
        np.asarray(group.src), return_inverse=True, return_counts=True
    )
    del source_ids
    expected = 99.0 * source_rows[source_inverse, None] * probability
    row_activity = source_rows[source_inverse]
    activity_cuts = np.quantile(row_activity, (0.5, 0.9), method="higher")
    activity_masks = (
        row_activity <= activity_cuts[0],
        (row_activity > activity_cuts[0]) & (row_activity <= activity_cuts[1]),
        row_activity > activity_cuts[1],
    )
    calibrated = pool_association.leave_one_out_calibration(
        source_count,
        row_activity,
        probability,
        tuple(int(value) for value in activity_cuts),
    )
    components = np.concatenate(
        [
            np.stack(
                [
                    _qnorm(np.log1p(source_count)),
                    _qnorm((source_count - expected) / np.sqrt(expected + 1.0)),
                ],
                axis=0,
            ),
            np.stack([_qnorm(component) for component in calibrated], axis=0),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    audit = {
        "quantiles": [0.5, 0.9],
        "cuts": [int(value) for value in activity_cuts],
        "row_band_shares": [float(np.mean(mask)) for mask in activity_masks],
        "source_rows_min": int(source_rows.min()),
        "source_rows_max": int(source_rows.max()),
    }
    return components, audit


def _metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    segments: dict[str, np.ndarray],
) -> dict[str, Any]:
    return data_features.ranking_metrics(scores, labels, segments=segments)


def _segment_mrr(report: dict[str, Any], name: str) -> float:
    value = report["mrr"] if name == "overall" else report["segments"][name]["mrr"]
    return float(value) if value is not None else float("nan")


def _score_group(
    *,
    cache: Any,
    group: Any,
    temporal_models: list[tuple[str, int, Any]],
    mf_models: list[tuple[str, Any, np.ndarray, np.ndarray]],
    transition_mf_models: list[tuple[str, Any, np.ndarray, np.ndarray]],
    histories: dict[int, temporal_history.TemporalHistory],
    test_ids: np.ndarray,
    test_counts: np.ndarray,
    transition_index: d4_transition_research.TransitionIndex,
    batch_rows: int,
    include_static_features: bool = False,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, Any],
    np.ndarray | None,
]:
    store = cache.feature_store(group.cutoff)
    names = [name for name, _, _ in temporal_models]
    names += [name for name, *_ in mf_models]
    names += [name for name, *_ in transition_mf_models]
    names += [
        "test_frequency",
        "pair_seen_recency",
        *TRANSITION_COMPONENT_NAMES,
        *POOL_COMPONENT_NAMES,
    ]
    pool_source, pool_activity = _pool_source_components(group, test_ids, test_counts)
    scores = np.empty((len(names), group.rows, group.candidate_count), dtype=np.float32)
    labels = np.empty(group.rows, dtype=np.int64)
    candidate_seen = np.empty((group.rows, group.candidate_count), dtype=bool)
    static_features = (
        np.empty(
            (group.rows, group.candidate_count, len(data_features.FEATURE_NAMES)),
            dtype=np.float32,
        )
        if include_static_features
        else None
    )
    segment_parts: dict[str, list[np.ndarray]] = {}
    offset = 0
    for batch in group.iter_batches(batch_rows=batch_rows):
        stop = offset + len(batch.src)
        features = store.features(batch.src, batch.time, batch.candidates)
        if static_features is not None:
            static_features[offset:stop] = features
        labels[offset:stop] = batch.labels
        candidate_seen[offset:stop] = features[:, :, PAIR_SEEN_INDEX] > 0.0
        component = 0
        temporal_cache: dict[int, Any] = {}
        for _, history_size, model in temporal_models:
            if history_size not in temporal_cache:
                temporal_cache[history_size] = histories[history_size].lookup(
                    batch.src, batch.time, batch.candidates
                )
            values = temporal_cache[history_size]
            scores[component, offset:stop] = temporal_attention_jittor.predict_scores(
                model,
                values.source_indices,
                values.candidate_indices,
                values.history_item_indices,
                values.log_time_deltas,
                features=features,
                batch_size=batch_rows,
            )
            component += 1
        for _, model, source_ids, item_ids in mf_models:
            scores[component, offset:stop] = implicit_mf_jittor.predict_scores(
                model,
                batch.src,
                batch.candidates,
                source_ids,
                item_ids,
                batch_size=batch_rows,
            )
            component += 1
        if transition_mf_models:
            positions = np.searchsorted(transition_index.source_ids, batch.src)
            inside = positions < len(transition_index.source_ids)
            matched = np.zeros(len(batch.src), dtype=bool)
            matched[inside] = (
                transition_index.source_ids[positions[inside]] == batch.src[inside]
            )
            previous = np.zeros(len(batch.src), dtype=np.uint32)
            previous[matched] = transition_index.last_items[positions[matched]]
            for _, model, source_ids, item_ids in transition_mf_models:
                scores[component, offset:stop] = implicit_mf_jittor.predict_scores(
                    model,
                    previous,
                    batch.candidates,
                    source_ids,
                    item_ids,
                    batch_size=batch_rows,
                )
                component += 1
        scores[component, offset:stop] = _candidate_counts(
            batch.candidates, test_ids, test_counts
        )
        component += 1
        scores[component, offset:stop] = (
            3.0 * features[:, :, PAIR_SEEN_INDEX]
            + features[:, :, PAIR_RECENCY_INDEX]
            + features[:, :, PAIR_LOG_COUNT_INDEX]
        )
        component += 1
        scores[component, offset:stop] = transition_index.score(
            batch.src, batch.candidates
        )
        for name, mask in store.evaluation_segments(
            batch.src, batch.time, batch.candidates, batch.labels
        ).items():
            segment_parts.setdefault(name, []).append(np.asarray(mask, dtype=bool))
        offset = stop
    if offset != group.rows:
        raise ValueError("candidate group row count changed")
    scores[-len(POOL_COMPONENT_NAMES) :] = pool_source
    del pool_source
    scores = np.stack([_qnorm(score) for score in scores], axis=0)
    segments = {name: np.concatenate(parts) for name, parts in segment_parts.items()}
    return names, scores, labels, candidate_seen, segments, pool_activity, static_features


def _gated_score(
    base: np.ndarray,
    mixed: np.ndarray,
    candidate_seen: np.ndarray,
    seen_alpha: float,
    new_alpha: float,
) -> np.ndarray:
    alpha = np.where(candidate_seen, seen_alpha, new_alpha).astype(np.float32)
    return base + alpha * (mixed - base)


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = args.data.resolve()
    cache_dir = args.cache_dir.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing run directory reuse: {run_dir}")
    if _sha256(data) != verify_run.EXPECTED_DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    if (
        args.batch_rows < 1
        or args.valid_groups < 0
        or args.confirm_groups < 0
        or not args.temporal
    ):
        raise ValueError("invalid group/batch size or no temporal model supplied")
    if args.pairnew_weight_report is not None:
        if not args.pairnew_transformer:
            raise ValueError("--pairnew-weight-report requires --pairnew-transformer")
        if args.valid_groups != 30000 or args.confirm_groups != 30000:
            raise ValueError("weighted v13 requires frozen 30000-row replay groups")
        if args.residual_member:
            raise ValueError("weighted v13 members come only from the frozen v12 report")
    run_dir.mkdir(parents=True)
    temporal_attention_jittor.configure_cuda()
    cache = data_features.BDataCache.build_or_open(
        data, "dataset4", cache_dir, verify_hash=True
    )
    test_ids, test_counts = cache.test_candidate_counts()

    temporal_models = []
    temporal_records = []
    for name, history_size_text, checkpoint_text in args.temporal:
        checkpoint = Path(checkpoint_text).resolve()
        history_size = int(history_size_text)
        model, config = temporal_attention_jittor.load_checkpoint(checkpoint)
        if history_size < 1 or int(config["feature_dim"]) != len(data_features.FEATURE_NAMES):
            raise ValueError(f"invalid temporal model: {name}")
        if not config.get("static_context_pair_seen_only"):
            raise ValueError(f"temporal model lacks the D4 pair-seen context gate: {name}")
        temporal_models.append((name, history_size, model))
        temporal_records.append(
            {
                "name": name,
                "history_size": history_size,
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "config": config,
            }
        )
    if len({name for name, *_ in temporal_models}) != len(temporal_models):
        raise ValueError("duplicate temporal model name")

    mf_models = []
    mf_records = []
    for name, checkpoint_text in args.mf:
        checkpoint = Path(checkpoint_text).resolve()
        model, source_ids, item_ids = implicit_mf_jittor.load_checkpoint(checkpoint)
        mf_models.append((name, model, source_ids, item_ids))
        mf_records.append(
            {"name": name, "path": str(checkpoint), "sha256": _sha256(checkpoint)}
        )

    transition_mf_models = []
    transition_mf_records = []
    for name, checkpoint_text in args.transition_mf:
        if not name.startswith("transition_mf_"):
            raise ValueError(f"transition MF name lacks required prefix: {name}")
        checkpoint = Path(checkpoint_text).resolve()
        model, source_ids, item_ids = implicit_mf_jittor.load_checkpoint(checkpoint)
        if not np.array_equal(source_ids, item_ids):
            raise ValueError(f"transition MF item vocabularies differ: {name}")
        transition_mf_models.append((name, model, source_ids, item_ids))
        transition_mf_records.append(
            {
                "name": name,
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "item_ids_sha256": data_features.sha256_array(item_ids),
            }
        )
    all_names = [
        name for name, *_ in temporal_models + mf_models + transition_mf_models
    ]
    if len(set(all_names)) != len(all_names):
        raise ValueError("duplicate component name")

    group_sizes = {
        "train": 1,
        "valid": None if args.valid_groups == 0 else args.valid_groups,
        "confirm": None if args.confirm_groups == 0 else args.confirm_groups,
    }
    strategy_groups = {
        strategy: data_features.build_split1_groups(
            cache,
            seed=int(args.group_seed),
            sizes=group_sizes,
            batch_rows=4096,
            negative_strategy=strategy,
        )
        for strategy in ("history", "test_pool")
    }
    history_groups = strategy_groups["history"]
    if any(
        groups.plan.cutoffs != history_groups.plan.cutoffs
        for groups in strategy_groups.values()
    ):
        raise ValueError("replay strategies produced different time cutoffs")
    vocabulary = temporal_history.TemporalVocabulary.from_training_edges(
        cache.src,
        cache.dst,
        cache.time,
        id_mode="bipartite",
        cutoff=int(history_groups.valid.cutoff),
    )
    for name, _, model in temporal_models:
        if (
            model.source_count != len(vocabulary.source_ids) + 1
            or model.item_count != len(vocabulary.item_ids) + 1
        ):
            raise ValueError(f"temporal checkpoint vocabulary differs: {name}")
    for name, _, source_ids, item_ids in transition_mf_models:
        if not np.array_equal(source_ids, vocabulary.item_ids) or not np.array_equal(
            item_ids, vocabulary.item_ids
        ):
            raise ValueError(f"transition MF checkpoint vocabulary differs: {name}")
    history_sizes = sorted({size for _, size, _ in temporal_models})
    histories = {
        split: {
            size: temporal_history.TemporalHistory.build(
                cache.src,
                cache.dst,
                cache.time,
                history_size=size,
                id_mode="bipartite",
                cutoff=int(group.cutoff),
                vocabulary=vocabulary,
            )
            for size in history_sizes
        }
        for split, group in (
            ("validation", history_groups.valid),
            ("confirmation", history_groups.confirm),
        )
    }
    transition_indexes = {
        split: d4_transition_research.TransitionIndex.build(cache, int(group.cutoff))
        for split, group in (
            ("validation", history_groups.valid),
            ("confirmation", history_groups.confirm),
        )
    }

    scored = {}
    group_metadata = {}
    pool_activity = {}
    for strategy, groups in strategy_groups.items():
        group_metadata[strategy] = groups.metadata
        for split, group in (("validation", groups.valid), ("confirmation", groups.confirm)):
            names, scores, labels, seen, segments, activity, static_features = _score_group(
                cache=cache,
                group=group,
                temporal_models=temporal_models,
                mf_models=mf_models,
                transition_mf_models=transition_mf_models,
                histories=histories[split],
                test_ids=test_ids,
                test_counts=test_counts,
                transition_index=transition_indexes[split],
                batch_rows=int(args.batch_rows),
                include_static_features=bool(args.pairnew_transformer),
            )
            scored[(strategy, split)] = (
                scores,
                labels,
                seen,
                segments,
                static_features,
            )
            pool_activity.setdefault(strategy, {})[split] = activity

    if args.pairnew_transformer:
        if args.control_fit is None:
            raise ValueError("--control-fit is required for pair-new Transformer")
        members = [
            (int(hidden), int(seed)) for hidden, seed in args.residual_member
        ]
        if not members and not args.residual_checkpoint:
            members = list(pairnew_transformer_jittor.DEFAULT_MEMBERS)
        if args.pairnew_weight_report is not None:
            report = pairnew_transformer_jittor.run_weighted_audit(
                scored=scored,
                component_names=names,
                control_fit_path=args.control_fit.resolve(),
                base_report_path=args.pairnew_weight_report.resolve(),
                batch=int(args.residual_batch_rows),
            )
        else:
            report = pairnew_transformer_jittor.run(
                scored=scored,
                component_names=names,
                control_fit_path=args.control_fit.resolve(),
                run_dir=run_dir,
                members=members,
                train_rows=int(args.residual_train_rows),
                epochs=int(args.residual_epochs),
                batch=int(args.residual_batch_rows),
                pretrained_checkpoints=args.residual_checkpoint,
                baseline_report_path=(
                    args.pairnew_baseline_report.resolve()
                    if args.pairnew_baseline_report is not None
                    else None
                ),
            )
        report.update(
            {
                "created_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "data_sha256": verify_run.EXPECTED_DATA_SHA256,
                "component_names": names,
                "group_metadata": group_metadata,
                "pool_activity": pool_activity,
                "temporal_models": temporal_records,
                "mf_models": mf_records,
                "transition_mf_models": transition_mf_records,
                "runtime": {
                    "jittor": str(temporal_attention_jittor.jt.__version__),
                    "has_cuda": bool(temporal_attention_jittor.jt.has_cuda),
                    "use_cuda": bool(temporal_attention_jittor.jt.flags.use_cuda),
                },
                "source_hashes": {
                    Path(__file__).name: _sha256(Path(__file__).resolve()),
                    Path(pairnew_transformer_jittor.__file__).name: _sha256(
                        Path(pairnew_transformer_jittor.__file__).resolve()
                    ),
                },
            }
        )
        _atomic_json(run_dir / "research_report.json", report)
        return report

    validation_scores, validation_labels, validation_seen, _, _ = scored[
        ("history", "validation")
    ]
    control_indices = np.asarray(
        [index for index, name in enumerate(names) if not _is_innovation(name)]
    )
    control_validation = validation_scores[control_indices]
    control_weights, control_convex_mrr = _tune_convex(
        control_validation, validation_labels
    )
    control_individual = np.asarray(
        [_mrr(score, validation_labels) for score in control_validation]
    )
    control_best_local = int(control_individual.argmax())
    control_best_index = int(control_indices[control_best_local])
    control_convex = np.tensordot(
        control_weights, control_validation, axes=(0, 0)
    )
    control_seen_alpha, control_new_alpha, control_gate_trace = _fit_gate(
        validation_scores[control_best_index],
        control_convex,
        validation_seen,
        validation_labels,
    )
    initial_weights = np.zeros(len(names), dtype=np.float64)
    initial_weights[control_indices] = control_weights
    innovation_indices = np.asarray(
        [index for index, name in enumerate(names) if _is_innovation(name)]
    )
    weights, nested_validation_mrr, innovation_trace = _tune_innovations(
        validation_scores,
        validation_labels,
        validation_seen,
        control_best_index,
        initial_weights,
        innovation_indices,
        control_seen_alpha,
        control_new_alpha,
    )
    best_index = control_best_index
    base = validation_scores[best_index]
    convex = np.tensordot(weights, validation_scores, axes=(0, 0))
    convex_mrr = _mrr(convex, validation_labels)
    seen_alpha, new_alpha, gate_trace = _fit_gate(
        base, convex, validation_seen, validation_labels
    )
    refitted_validation_mrr = _mrr(
        _gated_score(base, convex, validation_seen, seen_alpha, new_alpha),
        validation_labels,
    )
    if refitted_validation_mrr + 1e-10 < nested_validation_mrr:
        seen_alpha = control_seen_alpha
        new_alpha = control_new_alpha
        gate_trace.append(
            {
                "fallback_to_control_gate": 1.0,
                "mrr": float(nested_validation_mrr),
            }
        )

    reports = {}
    for (strategy, split), (scores, labels, seen, segments, _) in scored.items():
        convex = np.tensordot(weights, scores, axes=(0, 0))
        gated = _gated_score(
            scores[best_index], convex, seen, seen_alpha, new_alpha
        )
        reports.setdefault(strategy, {})[split] = {
            "individual": {
                name: _metrics(score, labels, segments) for name, score in zip(names, scores)
            },
            "convex": _metrics(convex, labels, segments),
            "gated": _metrics(gated, labels, segments),
        }
        control_mixed = np.tensordot(
            control_weights, scores[control_indices], axes=(0, 0)
        )
        control_gated = _gated_score(
            scores[control_best_index],
            control_mixed,
            seen,
            control_seen_alpha,
            control_new_alpha,
        )
        reports[strategy][split]["control_gated"] = _metrics(
            control_gated, labels, segments
        )

    if args.control_only:
        control_names = [names[index] for index in control_indices]
        control_reports = {}
        control_checks = {
            "multi_model_active": int(np.count_nonzero(control_weights > 1e-12)) > 1
            and (control_seen_alpha > 0.0 or control_new_alpha > 0.0),
        }
        for (strategy, split), (scores, labels, seen, segments, _) in scored.items():
            mixed = np.tensordot(
                control_weights, scores[control_indices], axes=(0, 0)
            )
            gated = _gated_score(
                scores[control_best_index],
                mixed,
                seen,
                control_seen_alpha,
                control_new_alpha,
            )
            individual = {
                name: _metrics(scores[index], labels, segments)
                for name, index in zip(control_names, control_indices)
            }
            section = {
                "individual": individual,
                "convex": _metrics(mixed, labels, segments),
                "gated": _metrics(gated, labels, segments),
            }
            control_reports.setdefault(strategy, {})[split] = section
            base_report = individual[names[control_best_index]]
            tolerance = 0.0005 if strategy == "history" else 0.001
            control_checks[f"{strategy}_{split}_overall"] = (
                _segment_mrr(section["gated"], "overall")
                >= _segment_mrr(base_report, "overall") - tolerance
            )
            for segment in ("pair_new", "pair_seen", "source_hot"):
                control_checks[f"{strategy}_{split}_{segment}"] = (
                    _segment_mrr(section["gated"], segment)
                    >= _segment_mrr(base_report, segment) - 0.003
                )
        validation = control_reports["history"]["validation"]
        control_checks["history_validation_improves"] = (
            validation["gated"]["mrr"]
            > validation["individual"][names[control_best_index]]["mrr"]
        )
        source_files = (
            Path(__file__).resolve(),
            Path(data_features.__file__).resolve(),
            Path(temporal_attention_jittor.__file__).resolve(),
            Path(temporal_history.__file__).resolve(),
            Path(implicit_mf_jittor.__file__).resolve(),
            Path(pool_association.__file__).resolve(),
            Path(d4_transition_research.__file__).resolve(),
        )
        report = {
            "kind": "d4_multimodel_fit_v1",
            "decision": "PASS" if all(control_checks.values()) else "NO_GO",
            "created_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "data_sha256": verify_run.EXPECTED_DATA_SHA256,
            "selection_replay": "history validation only",
            "component_names": control_names,
            "best_component": names[control_best_index],
            "convex_validation_mrr": control_convex_mrr,
            "weights": {
                name: float(weight)
                for name, weight in zip(control_names, control_weights)
            },
            "seen_alpha": control_seen_alpha,
            "new_alpha": control_new_alpha,
            "gate_trace": control_gate_trace,
            "metrics": control_reports,
            "checks": control_checks,
            "group_metadata": group_metadata,
            "pool_activity": pool_activity,
            "temporal_models": temporal_records,
            "mf_models": mf_records,
            "runtime": {
                "jittor": str(temporal_attention_jittor.jt.__version__),
                "has_cuda": bool(temporal_attention_jittor.jt.has_cuda),
                "use_cuda": bool(temporal_attention_jittor.jt.flags.use_cuda),
            },
            "source_hashes": {path.name: _sha256(path) for path in source_files},
            "confirmation_excluded_from_selection": True,
            "test_pool_is_diagnostic_only": True,
        }
        _atomic_json(run_dir / "research_report.json", report)
        return report

    checks = {
        "multi_model_active": int(np.count_nonzero(weights > 1e-12)) > 1
        and (seen_alpha > 0.0 or new_alpha > 0.0),
        "transition_component_active": any(
            weight > 1e-12
            and (name in TRANSITION_COMPONENT_NAMES or name.startswith("transition_mf_"))
            for name, weight in zip(names, weights)
        ),
        "transition_mf_component_active": any(
            weight > 1e-12 and name.startswith("transition_mf_")
            for name, weight in zip(names, weights)
        ),
        "history_validation_improves": reports["history"]["validation"]["gated"]["mrr"]
        > reports["history"]["validation"]["individual"][names[best_index]]["mrr"],
    }
    for strategy in ("history", "test_pool"):
        for split in ("validation", "confirmation"):
            section = reports[strategy][split]
            base_report = section["individual"][names[best_index]]
            tolerance = 0.0005 if strategy == "history" else 0.001
            checks[f"{strategy}_{split}_overall"] = (
                _segment_mrr(section["gated"], "overall")
                >= _segment_mrr(base_report, "overall") - tolerance
            )
            for segment in ("pair_new", "pair_seen", "source_hot"):
                checks[f"{strategy}_{split}_{segment}"] = (
                    _segment_mrr(section["gated"], segment)
                    >= _segment_mrr(base_report, segment) - 0.003
                )
            checks[f"{strategy}_{split}_innovation_ablation"] = (
                _segment_mrr(section["gated"], "overall")
                >= _segment_mrr(section["control_gated"], "overall")
            )

    source_files = (
        Path(__file__).resolve(),
        Path(data_features.__file__).resolve(),
        Path(temporal_attention_jittor.__file__).resolve(),
        Path(temporal_history.__file__).resolve(),
        Path(implicit_mf_jittor.__file__).resolve(),
        Path(pool_association.__file__).resolve(),
        Path(d4_transition_research.__file__).resolve(),
    )
    report = {
        "kind": "d4_transition_mf_nested_multimodel_fit_v6",
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_sha256": verify_run.EXPECTED_DATA_SHA256,
        "selection_replay": "history validation only",
        "component_names": names,
        "best_component": names[best_index],
        "convex_validation_mrr": convex_mrr,
        "weights": {name: float(weight) for name, weight in zip(names, weights)},
        "seen_alpha": seen_alpha,
        "new_alpha": new_alpha,
        "gate_trace": gate_trace,
        "innovation_fit": {
            "fixed_control_seen_alpha": control_seen_alpha,
            "fixed_control_new_alpha": control_new_alpha,
            "nested_validation_mrr": nested_validation_mrr,
            "trace": innovation_trace,
        },
        "innovation_ablation": {
            "component_names": [names[index] for index in control_indices],
            "best_component": names[control_best_index],
            "convex_validation_mrr": control_convex_mrr,
            "weights": {
                names[index]: float(weight)
                for index, weight in zip(control_indices, control_weights)
            },
            "seen_alpha": control_seen_alpha,
            "new_alpha": control_new_alpha,
            "gate_trace": control_gate_trace,
        },
        "metrics": reports,
        "checks": checks,
        "group_metadata": group_metadata,
        "pool_activity": pool_activity,
        "transition_indexes": {
            name: index.metadata() for name, index in transition_indexes.items()
        },
        "temporal_models": temporal_records,
        "mf_models": mf_records,
        "transition_mf_models": transition_mf_records,
        "runtime": {
            "jittor": str(temporal_attention_jittor.jt.__version__),
            "has_cuda": bool(temporal_attention_jittor.jt.has_cuda),
            "use_cuda": bool(temporal_attention_jittor.jt.flags.use_cuda),
        },
        "source_hashes": {path.name: _sha256(path) for path in source_files},
        "confirmation_excluded_from_selection": True,
        "test_pool_is_diagnostic_only": True,
    }
    _atomic_json(run_dir / "research_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--temporal",
        nargs=3,
        action="append",
        metavar=("NAME", "HISTORY_SIZE", "CHECKPOINT"),
        default=[],
    )
    parser.add_argument(
        "--mf", nargs=2, action="append", metavar=("NAME", "CHECKPOINT"), default=[]
    )
    parser.add_argument(
        "--transition-mf",
        nargs=2,
        action="append",
        metavar=("NAME", "CHECKPOINT"),
        default=[],
    )
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument("--group-seed", type=int, default=20260810)
    parser.add_argument(
        "--valid-groups",
        type=int,
        default=30000,
        help="validation rows; 0 uses the complete temporal block",
    )
    parser.add_argument(
        "--confirm-groups",
        type=int,
        default=30000,
        help="confirmation rows; 0 uses the complete temporal block",
    )
    parser.add_argument("--pairnew-transformer", action="store_true")
    parser.add_argument(
        "--control-only",
        action="store_true",
        help="fit and publish only the non-innovation causal control",
    )
    parser.add_argument("--pairnew-baseline-report", type=Path)
    parser.add_argument("--control-fit", type=Path)
    parser.add_argument(
        "--pairnew-weight-report",
        type=Path,
        help="load a frozen v12 report and audit constrained member weights",
    )
    parser.add_argument(
        "--residual-member",
        nargs=2,
        action="append",
        metavar=("HIDDEN", "SEED"),
        default=[],
    )
    parser.add_argument(
        "--residual-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="aggregate an already trained member using the same replay contract",
    )
    parser.add_argument("--residual-train-rows", type=int, default=20000)
    parser.add_argument("--residual-epochs", type=int, default=6)
    parser.add_argument("--residual-batch-rows", type=int, default=128)
    return parser


def main() -> int:
    try:
        print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "kind": "d4_transition_mf_nested_multimodel_fit_v6",
                    "decision": "ERROR",
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
