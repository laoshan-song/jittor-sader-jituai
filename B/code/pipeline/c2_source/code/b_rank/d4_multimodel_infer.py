#!/usr/bin/env python3
"""Stream the audited D3 result and fitted full-history D4 ensemble to ZIP."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import (
    d4_transition_mf_deploy,
    d4_transition_research,
    data_features,
    fullset_ranker_jittor,
    implicit_mf_jittor,
    pairnew_transformer_jittor,
    pool_association,
    replay_score_cache,
    temporal_attention_jittor,
    temporal_history,
    temporal_infer,
    verify_run,
)


PAIR_SEEN_INDEX = data_features.FEATURE_NAMES.index("pair_seen")
PAIR_LOG_COUNT_INDEX = data_features.FEATURE_NAMES.index("pair_log_count")
PAIR_RECENCY_INDEX = data_features.FEATURE_NAMES.index("pair_recency")
MEMBERS = ("dataset3.csv", "dataset4.csv")
ROWS = {"dataset3": 157670, "dataset4": 2322538}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _publish_new(temporary: Path, destination: Path) -> None:
    _require(not destination.exists(), f"refusing output reuse: {destination}")
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _qnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


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


def _duplicate_candidate_rows(candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates)
    _require(candidates.ndim == 2, "candidate matrix must be two-dimensional")
    if candidates.shape[1] < 2:
        return np.zeros(candidates.shape[0], dtype=bool)
    ordered = np.sort(candidates, axis=1)
    return np.any(ordered[:, 1:] == ordered[:, :-1], axis=1)


def _apply_duplicate_candidate_gate(
    candidate: np.ndarray,
    control: np.ndarray,
    candidates: np.ndarray,
    duplicate_rows: np.ndarray,
) -> np.ndarray:
    rows = np.flatnonzero(duplicate_rows)
    if not len(rows):
        return candidate
    ids = np.asarray(candidates)[rows]
    order = np.argsort(ids, axis=1, kind="stable")
    ordered_ids = np.take_along_axis(ids, order, axis=1)
    ordered_control = np.take_along_axis(np.asarray(control)[rows], order, axis=1)
    group_start = np.ones(ordered_ids.shape, dtype=bool)
    group_start[:, 1:] = ordered_ids[:, 1:] != ordered_ids[:, :-1]
    positions = np.arange(ordered_ids.shape[1])[None, :]
    first_positions = np.maximum.accumulate(
        np.where(group_start, positions, 0), axis=1
    )
    canonical = np.take_along_axis(ordered_control, first_positions, axis=1)
    inverse_order = np.argsort(order, axis=1)
    candidate[rows] = np.take_along_axis(canonical, inverse_order, axis=1)
    return candidate


def _build_source_pool_index(
    cache: Any,
    pool_dir: Path,
    *,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a sorted unlabeled (source, candidate) recurrence index.

    The official test pool is streamed into a memmap before one external-ish
    ``np.unique`` pass.  This keeps the 232M-cell D4 matrix out of RAM while
    making exact full-pool counts available to every inference chunk.
    """
    keys_path = pool_dir / "test_source_pair_keys_v1.npy"
    counts_path = pool_dir / "test_source_pair_counts_v1.npy"
    source_ids_path = pool_dir / "test_source_ids_v1.npy"
    source_counts_path = pool_dir / "test_source_counts_v1.npy"
    metadata_path = pool_dir / "test_source_pair_metadata_v1.json"
    if all(
        path.is_file()
        for path in (
            keys_path,
            counts_path,
            source_ids_path,
            source_counts_path,
            metadata_path,
        )
    ):
        metadata = _read_json(metadata_path)
        if metadata.get("archive_hash") != cache.archive_hash:
            raise ValueError("source-pool index archive hash differs")
        return (
            np.load(keys_path, mmap_mode="r", allow_pickle=False),
            np.load(counts_path, mmap_mode="r", allow_pickle=False),
            np.load(source_ids_path, mmap_mode="r", allow_pickle=False),
            np.load(source_counts_path, mmap_mode="r", allow_pickle=False),
        )

    pool_dir.mkdir(parents=True, exist_ok=True)
    stage = pool_dir / f".test_source_pair_pool_v1-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    parts: list[Path] = []
    source_parts: list[np.ndarray] = []
    total_cells = 0
    rows = 0
    width = 0
    try:
        for index, chunk in enumerate(cache.iter_test_chunks(chunk_rows=chunk_rows)):
            width = int(chunk.candidates.shape[1])
            keys = (
                chunk.src.astype(np.uint64, copy=False)[:, None] << np.uint64(32)
            ) | chunk.candidates.astype(np.uint64, copy=False)
            keys = keys.reshape(-1)
            part = stage / f"keys_{index:06d}.npy"
            np.save(part, keys, allow_pickle=False)
            parts.append(part)
            source_parts.append(np.asarray(chunk.src, dtype=np.uint32).copy())
            total_cells += int(keys.size)
            rows += len(chunk.src)

        raw_path = stage / "all_keys.npy"
        raw = np.lib.format.open_memmap(
            raw_path, mode="w+", dtype=np.uint64, shape=(total_cells,)
        )
        offset = 0
        for part in parts:
            values = np.load(part, mmap_mode="r", allow_pickle=False)
            raw[offset : offset + len(values)] = values
            offset += len(values)
        raw.flush()
        del raw
        for part in parts:
            part.unlink(missing_ok=True)

        raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        unique_keys, counts = np.unique(raw, return_counts=True)
        del raw
        raw_path.unlink(missing_ok=True)
        counts = counts.astype(np.uint32, copy=False)
        source_values = np.concatenate(source_parts)
        source_ids, source_counts = np.unique(source_values, return_counts=True)
        source_counts = source_counts.astype(np.uint32, copy=False)
        np.save(stage / "keys.npy", unique_keys, allow_pickle=False)
        np.save(stage / "counts.npy", counts, allow_pickle=False)
        np.save(stage / "source_ids.npy", source_ids.astype(np.uint32), allow_pickle=False)
        np.save(stage / "source_counts.npy", source_counts, allow_pickle=False)
        _atomic_json(
            stage / "metadata.json",
            {
                "kind": "d4_test_source_pair_pool_v1",
                "archive_hash": cache.archive_hash,
                "rows": rows,
                "candidate_width": width,
                "cells": total_cells,
                "unique_source_pair_cells": int(len(unique_keys)),
            },
        )
        for source, destination in (
            (stage / "keys.npy", keys_path),
            (stage / "counts.npy", counts_path),
            (stage / "source_ids.npy", source_ids_path),
            (stage / "source_counts.npy", source_counts_path),
            (stage / "metadata.json", metadata_path),
        ):
            os.replace(source, destination)
        return (
            np.load(keys_path, mmap_mode="r", allow_pickle=False),
            np.load(counts_path, mmap_mode="r", allow_pickle=False),
            np.load(source_ids_path, mmap_mode="r", allow_pickle=False),
            np.load(source_counts_path, mmap_mode="r", allow_pickle=False),
        )
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _build_source_pool_time_index(
    cache: Any,
    pool_dir: Path,
    repeated_pair_ids: np.ndarray,
    repeated_pair_counts: np.ndarray,
    *,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Index exact times only for source-candidate pairs repeated in the pool."""
    expected_repeated = int(np.sum(repeated_pair_counts, dtype=np.int64))
    index_path = pool_dir / "test_source_pair_time_index_v2.npy"
    scores_path = pool_dir / "test_source_pair_time_scores_v2.npy"
    metadata_path = pool_dir / "test_source_pair_time_metadata_v2.json"
    if all(path.is_file() for path in (index_path, scores_path, metadata_path)):
        metadata = _read_json(metadata_path)
        if (
            metadata.get("kind") != "d4_test_source_pair_time_index_v2"
            or metadata.get("archive_hash") != cache.archive_hash
            or int(metadata.get("repeated_occurrence_cells", -1))
            != expected_repeated
        ):
            raise ValueError("source-pool time index metadata differs")
        index = np.load(index_path, mmap_mode="r", allow_pickle=False)
        scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
        if (
            index.dtype != pool_association.PAIR_TIME_DTYPE
            or scores.dtype != np.float32
            or len(index) != len(scores)
            or int(metadata.get("unique_pair_times", -1)) != len(index)
        ):
            raise ValueError("source-pool time index arrays differ")
        return index, scores, metadata

    pool_dir.mkdir(parents=True, exist_ok=True)
    stage = pool_dir / f".test_source_pair_time_v2-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    key_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    total_cells = 0
    try:
        for chunk in cache.iter_test_chunks(chunk_rows=chunk_rows):
            keys = pool_association.source_candidate_keys(
                chunk.src, chunk.candidates
            ).reshape(-1)
            positions = np.searchsorted(repeated_pair_ids, keys)
            inside = positions < len(repeated_pair_ids)
            repeated = np.zeros(len(keys), dtype=bool)
            repeated[inside] = (
                repeated_pair_ids[positions[inside]] == keys[inside]
            )
            key_parts.append(keys[repeated].copy())
            flat_time = np.repeat(
                np.asarray(chunk.time, dtype=np.uint32), chunk.candidates.shape[1]
            )
            time_parts.append(flat_time[repeated].copy())
            total_cells += len(keys)
        repeated_keys = np.concatenate(key_parts)
        repeated_time = np.concatenate(time_parts)
        if len(repeated_keys) != expected_repeated:
            raise ValueError("source-pool repeated-pair occurrence count differs")
        index, scores = pool_association.pair_time_index(
            repeated_keys, repeated_time
        )
        del repeated_keys, repeated_time, key_parts, time_parts
        np.save(stage / "index.npy", index, allow_pickle=False)
        np.save(stage / "scores.npy", scores, allow_pickle=False)
        metadata = {
            "kind": "d4_test_source_pair_time_index_v2",
            "archive_hash": cache.archive_hash,
            "pool_cells": int(total_cells),
            "repeated_occurrence_cells": expected_repeated,
            "unique_pair_times": int(len(index)),
        }
        _atomic_json(stage / "metadata.json", metadata)
        for source, destination in (
            (stage / "index.npy", index_path),
            (stage / "scores.npy", scores_path),
            (stage / "metadata.json", metadata_path),
        ):
            os.replace(source, destination)
        stage.rmdir()
        return (
            np.load(index_path, mmap_mode="r", allow_pickle=False),
            np.load(scores_path, mmap_mode="r", allow_pickle=False),
            metadata,
        )
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _pool_source_components(
    src: np.ndarray,
    query_time: np.ndarray,
    candidates: np.ndarray,
    pair_ids: np.ndarray,
    pair_counts: np.ndarray,
    source_ids: np.ndarray,
    source_counts: np.ndarray,
    test_ids: np.ndarray,
    test_counts: np.ndarray,
    pair_time_index: np.ndarray | None,
    pair_time_scores: np.ndarray | None,
    activity_cuts: tuple[int, int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    keys = (
        np.asarray(src, dtype=np.uint64)[:, None] << np.uint64(32)
    ) | np.asarray(candidates, dtype=np.uint64)
    flat = keys.reshape(-1)
    positions = np.searchsorted(pair_ids, flat)
    inside = positions < len(pair_ids)
    matched = np.zeros(len(flat), dtype=bool)
    matched[inside] = pair_ids[positions[inside]] == flat[inside]
    counts = np.ones(len(flat), dtype=np.float32)
    counts[matched] = pair_counts[positions[matched]]
    source = np.asarray(src, dtype=np.uint32)
    source_positions = np.searchsorted(source_ids, source)
    source_inside = source_positions < len(source_ids)
    source_matched = np.zeros(len(source), dtype=bool)
    source_matched[source_inside] = (
        source_ids[source_positions[source_inside]] == source[source_inside]
    )
    rows = np.zeros(len(source), dtype=np.float32)
    rows[source_matched] = source_counts[source_positions[source_matched]]
    candidate = np.asarray(candidates, dtype=np.uint32)
    candidate_positions = np.searchsorted(test_ids, candidate.reshape(-1))
    candidate_inside = candidate_positions < len(test_ids)
    candidate_matched = np.zeros(len(candidate_positions), dtype=bool)
    candidate_matched[candidate_inside] = (
        test_ids[candidate_positions[candidate_inside]]
        == candidate.reshape(-1)[candidate_inside]
    )
    probability = np.zeros(len(candidate_positions), dtype=np.float32)
    probability[candidate_matched] = (
        test_counts[candidate_positions[candidate_matched]].astype(np.float32)
        / float(np.asarray(test_counts, dtype=np.float64).sum())
    )
    expected = 99.0 * rows[:, None] * probability.reshape(candidate.shape)
    counts = counts.reshape(candidate.shape)
    if pair_time_index is None or pair_time_scores is None:
        nearest = np.zeros(candidate.shape, dtype=np.float32)
    else:
        nearest = _qnorm(
            pool_association.lookup_pair_time_scores(
                src,
                candidates,
                query_time,
                pair_time_index,
                pair_time_scores,
            )
        )
    activity_masks = (
        rows <= activity_cuts[0],
        (rows > activity_cuts[0]) & (rows <= activity_cuts[1]),
        rows > activity_cuts[1],
    )
    calibrated = pool_association.leave_one_out_calibration(
        counts,
        rows,
        probability.reshape(candidate.shape),
        activity_cuts,
    )
    return (
        _qnorm(np.log1p(counts)),
        _qnorm((counts - expected) / np.sqrt(expected + 1.0)),
        nearest,
        *(nearest * mask[:, None] for mask in activity_masks),
        *(_qnorm(component) for component in calibrated),
    )


def _row_weighted_activity_cuts(source_counts: np.ndarray) -> tuple[int, int]:
    counts = np.asarray(source_counts, dtype=np.int64)
    if not len(counts) or np.any(counts < 1):
        raise ValueError("source activity counts are invalid")
    order = np.argsort(counts, kind="stable")
    cumulative = np.cumsum(counts[order], dtype=np.int64)
    row_indices = np.ceil(
        np.asarray((0.5, 0.9)) * (cumulative[-1] - 1)
    ).astype(np.int64)
    positions = np.searchsorted(cumulative, row_indices, side="right")
    return tuple(int(value) for value in counts[order[positions]])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _fit_contract(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    _require(
        report.get("kind")
        in {
            "d4_multimodel_fit_v1",
            "d4_poolset_multimodel_fit_v1",
            "d4_pooltime_multimodel_fit_v2",
            "d4_poolcal_multimodel_fit_v3",
            "d4_transition_multimodel_fit_v4",
            "d4_transition_nested_multimodel_fit_v5",
        },
        "fit report kind differs",
    )
    _require(report.get("decision") == "PASS", "fit report did not pass")
    _require(
        report.get("data_sha256") == verify_run.EXPECTED_DATA_SHA256,
        "fit report data differs",
    )
    _require(
        report.get("selection_replay") == "history validation only",
        "fit report selection replay differs",
    )
    _require(report.get("confirmation_excluded_from_selection") is True, "confirmation leak")
    _require(report.get("test_pool_is_diagnostic_only") is True, "test-pool selection leak")
    checks = report.get("checks")
    _require(isinstance(checks, dict) and checks and all(checks.values()), "fit checks failed")
    names = report.get("component_names")
    weights = report.get("weights")
    _require(isinstance(names, list) and len(names) == len(set(names)), "component names differ")
    _require(isinstance(weights, dict) and set(weights) == set(names), "fit weights differ")
    values = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
    _require(np.isfinite(values).all() and np.all(values >= 0.0), "invalid fit weights")
    _require(abs(float(values.sum()) - 1.0) <= 1e-8, "fit weights do not sum to one")
    _require(report.get("best_component") in names, "fit base component differs")
    if report.get("kind") == "d4_pooltime_multimodel_fit_v2":
        nearest = {
            "pool_source_nearest",
            "pool_source_nearest_low",
            "pool_source_nearest_mid",
            "pool_source_nearest_high",
        }
        _require(nearest <= set(names), "nearest pool components are missing")
        _require(
            any(float(weights[name]) > 1e-12 for name in nearest),
            "nearest pool components are inactive",
        )
        _require(isinstance(report.get("pool_activity"), dict), "pool activity audit is missing")
    if report.get("kind") == "d4_poolcal_multimodel_fit_v3":
        calibrated = {
            "pool_source_leaveone_lift",
            "pool_source_leaveone_deviance",
            "pool_source_leaveone_deviance_low",
            "pool_source_leaveone_deviance_mid",
            "pool_source_leaveone_deviance_high",
        }
        _require(calibrated <= set(names), "calibrated pool components are missing")
        _require(
            any(float(weights[name]) > 1e-12 for name in calibrated),
            "calibrated pool components are inactive",
        )
        _require(isinstance(report.get("pool_activity"), dict), "pool activity audit is missing")
    if report.get("kind") in {
        "d4_transition_multimodel_fit_v4",
        "d4_transition_nested_multimodel_fit_v5",
    }:
        transition = {"transition_last_count"}
        _require(transition <= set(names), "transition components are missing")
        _require(
            any(float(weights[name]) > 1e-12 for name in transition),
            "transition components are inactive",
        )
        _require(
            isinstance(report.get("transition_indexes"), dict),
            "transition index audit is missing",
        )
        _require(
            report.get("source_hashes", {}).get("d4_transition_research.py")
            == _sha256(Path(d4_transition_research.__file__).resolve()),
            "transition source hash differs",
        )
    if report.get("kind") == "d4_transition_nested_multimodel_fit_v5":
        nested = report.get("innovation_fit")
        control = report.get("innovation_ablation")
        _require(isinstance(nested, dict), "nested innovation audit is missing")
        _require(isinstance(control, dict), "nested control audit is missing")
        _require(
            float(nested.get("fixed_control_seen_alpha", -1.0))
            == float(control.get("seen_alpha", -2.0))
            and float(nested.get("fixed_control_new_alpha", -1.0))
            == float(control.get("new_alpha", -2.0)),
            "nested control gate differs",
        )
        validation = report.get("metrics", {}).get("history", {}).get("validation", {})
        _require(
            float(validation.get("gated", {}).get("mrr", -1.0))
            >= float(validation.get("control_gated", {}).get("mrr", 1.0)),
            "nested validation regressed from control",
        )
    for key in ("seen_alpha", "new_alpha"):
        value = float(report[key])
        _require(0.0 <= value <= 1.0, f"{key} is outside [0, 1]")
    return report


def _pairnew_contract(path: Path, control_path: Path) -> dict[str, Any]:
    report = _read_json(path)
    kind = report.get("kind")
    _require(
        kind in {
            "d4_pairnew_rank_slot_candidate_set_transformer_v12",
            "d4_pairnew_rank_slot_scaled_replay_transformer_v21",
            "d4_pairnew_rank_slot_weighted_scaled_v21_c2",
        },
        "pair-new fit report kind differs",
    )
    _require(report.get("decision") == "PASS", "pair-new fit did not pass")
    _require(
        report.get("data_sha256") == verify_run.EXPECTED_DATA_SHA256,
        "pair-new fit data differs",
    )
    _require(report.get("confirmation_excluded_from_selection") is True, "pair-new confirmation leak")
    _require(
        report.get("test_pool_holdout_is_diagnostic_only") is True,
        "pair-new test-pool selection leak",
    )
    checks = report.get("checks")
    _require(isinstance(checks, dict) and checks and all(checks.values()), "pair-new checks failed")
    _require(
        report.get("control_fit", {}).get("sha256") == _sha256(control_path),
        "pair-new frozen control differs",
    )
    _require(
        report.get("source_hashes", {}).get("pairnew_transformer_jittor.py")
        == _sha256(Path(pairnew_transformer_jittor.__file__).resolve()),
        "pair-new source hash differs",
    )
    if kind in {
        "d4_pairnew_rank_slot_scaled_replay_transformer_v21",
        "d4_pairnew_rank_slot_weighted_scaled_v21_c2",
    }:
        _require(
            report.get("source_hashes", {}).get("d4_multimodel_fit.py")
            == _sha256(Path(__file__).with_name("d4_multimodel_fit.py")),
            "scaled v21 fit source hash differs",
        )
    names = report.get("component_names")
    _require(isinstance(names, list) and len(names) == len(set(names)), "pair-new components differ")
    expected_features = [
        *(f"component:{name}" for name in names),
        *(f"causal:{name}" for name in data_features.FEATURE_NAMES),
        "frozen_control",
        "candidate_pair_new",
    ]
    _require(report.get("feature_names") == expected_features, "pair-new feature order differs")
    alpha = float(report.get("residual_alpha", -1.0))
    _require(0.025 <= alpha <= 2.0, "pair-new residual weight is inactive")
    if kind in {
        "d4_pairnew_rank_slot_scaled_replay_transformer_v21",
        "d4_pairnew_rank_slot_weighted_scaled_v21_c2",
    }:
        training = report.get("training", {})
        members = training.get("members")
        requested_rows = int(training.get("requested_rows_per_replay", 0))
        _require(
            requested_rows > 0
            and str(report.get("selection_replay", "")).startswith(
                f"history validation rows {requested_rows}:"
            )
            and str(report.get("selection_replay", "")).endswith(" only"),
            "scaled v21 selection replay differs",
        )
        _require(
            report.get("training_replays")
            == [
                f"history validation rows 0:{requested_rows}",
                f"test_pool validation rows 0:{requested_rows}",
            ],
            "scaled v21 training replay differs",
        )
        _require(
            isinstance(members, list)
            and len(members) == 6
            and len({int(member["seed"]) for member in members}) == 6
            and [int(member["hidden"]) for member in members]
            == [64, 64, 64, 96, 96, 96],
            "scaled v21 training contract differs",
        )
        if kind == "d4_pairnew_rank_slot_weighted_scaled_v21_c2":
            weights = np.asarray(report.get("member_weights"), dtype=np.float64)
            selected = report.get("selected", {})
            _require(
                weights.shape == (6,)
                and np.isfinite(weights).all()
                and np.all(weights >= 0.0)
                and np.isclose(float(weights.sum()), 1.0)
                and selected.get("name") != "equal6"
                and np.allclose(weights, selected.get("weights")),
                "c2 weighted member selection differs",
            )
    return report


def _fullset_contract(
    path: Path, pairnew_path: Path, control_path: Path
) -> dict[str, Any]:
    report = _read_json(path)
    _require(
        report.get("kind") == "d4_fullset_multiscale_lambdamrr_candidate_v15",
        "full-set fit report kind differs",
    )
    _require(report.get("decision") == "PASS", "full-set fit did not pass")
    _require(
        report.get("data_sha256") == verify_run.EXPECTED_DATA_SHA256,
        "full-set fit data differs",
    )
    _require(
        report.get("confirmation_excluded_from_selection") is True,
        "full-set confirmation leak",
    )
    checks = report.get("checks")
    _require(
        isinstance(checks, dict) and checks and all(checks.values()),
        "full-set checks failed",
    )
    _require(
        report.get("control_fit", {}).get("sha256") == _sha256(control_path),
        "full-set frozen control differs",
    )
    _require(
        report.get("comparison_v12", {}).get("sha256")
        == _sha256(pairnew_path),
        "full-set v12 comparison differs",
    )
    _require(
        report.get("source_hashes", {}).get("fullset_ranker_jittor.py")
        == _sha256(Path(fullset_ranker_jittor.__file__).resolve()),
        "full-set source hash differs",
    )
    _require(
        report.get("source_hashes", {}).get("replay_score_cache.py")
        == _sha256(Path(replay_score_cache.__file__).resolve()),
        "full-set cache source hash differs",
    )
    names = report.get("component_names")
    pairnew = _read_json(pairnew_path)
    _require(
        isinstance(names, list)
        and len(names) == len(set(names))
        and names == pairnew.get("component_names"),
        "full-set components differ",
    )
    expected_features = fullset_ranker_jittor.feature_names(names)
    _require(
        report.get("feature_names") == expected_features
        and int(report.get("feature_count", -1)) == len(expected_features),
        "full-set feature order differs",
    )
    policy = report.get("policy")
    _require(isinstance(policy, dict), "full-set policy is missing")
    _require(
        0.0 <= float(policy.get("new_residual_weight", -1.0)) <= 1.0
        and 0.0 <= float(policy.get("slot_alpha", -1.0)) <= 2.0
        and 0.0 <= float(policy.get("escape_alpha", -1.0)) <= 1.5
        and 0.0 <= float(policy.get("escape_margin", -1.0)) <= 4.0,
        "full-set policy is invalid",
    )
    return report


def _copy_d3(
    source_zip: Path,
    source_manifest: Path,
    destination: zipfile.ZipFile,
) -> dict[str, Any]:
    manifest = _read_json(source_manifest)
    source_hash = _sha256(source_zip)
    _require(
        manifest.get("data_sha256") == verify_run.EXPECTED_DATA_SHA256,
        "D3 source manifest data differs",
    )
    _require(manifest.get("submission_sha256") == source_hash, "D3 source ZIP hash differs")
    # The first validated high-score package used ``dataset3``; later package
    # manifests called the same section ``dataset3_ensemble``.
    d3 = manifest.get("dataset3_ensemble") or manifest.get("dataset3")
    _require(isinstance(d3, dict), "D3 ensemble manifest section is missing")
    _require(int(d3.get("active_component_count", 0)) >= 2, "D3 is not multi-model")
    digest = hashlib.sha256()
    with zipfile.ZipFile(source_zip) as source:
        _require(tuple(source.namelist()) == MEMBERS, "D3 source ZIP members differ")
        _require(source.testzip() is None, "D3 source ZIP CRC failed")
        info = source.getinfo("dataset3.csv")
        with source.open(info) as incoming, destination.open(
            "dataset3.csv", "w", force_zip64=True
        ) as outgoing:
            while True:
                block = incoming.read(8 << 20)
                if not block:
                    break
                digest.update(block)
                outgoing.write(block)
    return {
        "source_zip": str(source_zip),
        "source_zip_sha256": source_hash,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "csv_sha256": digest.hexdigest(),
        "active_component_count": int(d3["active_component_count"]),
        "weights": d3["weights"],
    }


class D4Ensemble:
    def __init__(self, args: argparse.Namespace) -> None:
        self.data = args.data.resolve()
        self.fit_path = args.fit_report.resolve()
        self.fit = _fit_contract(self.fit_path)
        self.names = list(self.fit["component_names"])
        self.weights = {name: float(self.fit["weights"][name]) for name in self.names}
        self.base_name = str(self.fit["best_component"])
        self.pairnew_path = (
            args.pairnew_report.resolve() if args.pairnew_report is not None else None
        )
        self.pairnew = (
            _pairnew_contract(self.pairnew_path, self.fit_path)
            if self.pairnew_path is not None
            else None
        )
        self.fullset_path = (
            args.fullset_report.resolve()
            if args.fullset_report is not None
            else None
        )
        _require(
            self.fullset_path is None or self.pairnew_path is not None,
            "full-set inference requires the v12 comparison report",
        )
        self.fullset = (
            _fullset_contract(
                self.fullset_path, self.pairnew_path, self.fit_path
            )
            if self.fullset_path is not None
            else None
        )
        self.residual_component_names = (
            list((self.fullset or self.pairnew)["component_names"])
            if (self.fullset or self.pairnew) is not None
            else []
        )
        self.needed = {
            name for name, weight in self.weights.items() if weight > 1e-12
        } | {self.base_name} | set(self.residual_component_names)
        self.seen_alpha = float(self.fit["seen_alpha"])
        self.new_alpha = float(self.fit["new_alpha"])
        self.scored_rows = 0
        self.duplicate_candidate_rows = 0

        self.cache = data_features.BDataCache.build_or_open(
            self.data,
            "dataset4",
            args.cache_dir.resolve(),
            chunk_rows=args.cache_chunk_rows,
            verify_hash=True,
        )
        self.test_cutoff = temporal_infer._test_history_cutoff(
            data_features,
            self.data,
            self.cache,
            chunk_rows=args.test_chunk_rows,
        )
        self.history_rows = self.cache.history_end(self.test_cutoff)
        _require(self.history_rows == len(self.cache.src), "D4 test history is incomplete")
        self.store = self.cache.feature_store(self.test_cutoff)
        self.test_ids, self.test_counts = self.cache.test_candidate_counts()
        self.source_pair_ids = None
        self.source_pair_counts = None
        self.repeated_source_pair_ids = None
        self.repeated_source_pair_counts = None
        self.test_source_ids = None
        self.test_source_counts = None
        self.source_pair_time_index = None
        self.source_pair_time_scores = None
        self.source_pair_time_metadata = None
        self.activity_cuts = (0, 0)
        self.transition_index = None
        if "transition_last_count" in self.needed or any(
            name.startswith("transition_mf_") for name in self.needed
        ):
            self.transition_index = d4_transition_research.TransitionIndex.build(
                self.cache, self.test_cutoff
            )
        if any(name.startswith("pool_source_") for name in self.needed):
            (
                self.source_pair_ids,
                self.source_pair_counts,
                self.test_source_ids,
                self.test_source_counts,
            ) = _build_source_pool_index(
                self.cache, self.cache.root / "pools", chunk_rows=args.test_chunk_rows
            )
            self.activity_cuts = _row_weighted_activity_cuts(
                self.test_source_counts
            )
            repeated = np.asarray(self.source_pair_counts) > 1
            self.repeated_source_pair_ids = np.asarray(
                self.source_pair_ids[repeated], dtype=np.uint64
            )
            self.repeated_source_pair_counts = np.asarray(
                self.source_pair_counts[repeated], dtype=np.uint32
            )
            del repeated
        if any(name.startswith("pool_source_nearest") for name in self.needed):
            (
                self.source_pair_time_index,
                self.source_pair_time_scores,
                self.source_pair_time_metadata,
            ) = _build_source_pool_time_index(
                self.cache,
                self.cache.root / "pools",
                self.repeated_source_pair_ids,
                self.repeated_source_pair_counts,
                chunk_rows=args.test_chunk_rows,
            )

        allowed_temporal_kinds = {
            "d4_full_split1_temporal_deploy_v1",
            "d4_full_split1_testpool_temporal_deploy_v1",
        }
        temporal_reports = []
        records = {}
        vocabularies = []
        for report_path in (path.resolve() for path in args.temporal_report):
            report = _read_json(report_path)
            _require(
                report.get("kind") in allowed_temporal_kinds,
                "temporal deploy report kind differs",
            )
            _require(
                report.get("decision") == "READY_FOR_CAUSAL_TEST_INFERENCE",
                "temporal deploy report is not ready",
            )
            _require(report.get("uses_test_labels") is False, "temporal test-label leak")
            _require(int(report["test_cutoff"]) == self.test_cutoff, "temporal cutoff differs")
            _require(int(report["history_rows"]) == self.history_rows, "temporal history differs")
            for record in report["checkpoints"]:
                name = record["name"]
                _require(name not in records, f"duplicate temporal checkpoint: {name}")
                records[name] = record
            vocabularies.append(report["vocabulary"])
            temporal_reports.append(
                {"path": str(report_path), "sha256": _sha256(report_path), "kind": report["kind"]}
            )
        temporal_names = {
            name for name in self.needed if name.startswith("temporal_")
        }
        _require(temporal_names <= set(records), "required temporal checkpoint is missing")
        self.temporal_models = {}
        max_history_size = max(
            (int(records[name]["history_size"]) for name in temporal_names), default=1
        )
        self.history = temporal_history.TemporalHistory.build(
            self.cache.src,
            self.cache.dst,
            self.cache.time,
            history_size=max_history_size,
            id_mode="bipartite",
            cutoff=self.test_cutoff,
        )
        _require(self.history.history_rows == self.history_rows, "temporal index rows differ")
        for vocabulary in vocabularies:
            _require(
                data_features.sha256_array(self.history.vocabulary.source_ids)
                == vocabulary["source_ids_sha256"],
                "temporal source vocabulary differs",
            )
            _require(
                data_features.sha256_array(self.history.vocabulary.item_ids)
                == vocabulary["item_ids_sha256"],
                "temporal item vocabulary differs",
            )
        for name in temporal_names:
            record = records[name]
            checkpoint = Path(record["path"])
            _require(_sha256(checkpoint) == record["sha256"], f"temporal hash differs: {name}")
            model, config = temporal_attention_jittor.load_checkpoint(checkpoint)
            _require(config == record["config"], f"temporal config differs: {name}")
            _require(config.get("static_context_pair_seen_only") is True, f"D4 gate differs: {name}")
            _require(
                model.source_count == self.history.source_vocab_size
                and model.item_count == self.history.item_vocab_size,
                f"temporal vocabulary size differs: {name}",
            )
            self.temporal_models[name] = (int(record["history_size"]), model)

        self.mf_models = {}
        self.mf_reports = {}
        supplied_mf = {name: Path(path).resolve() for name, path in args.mf_report}
        mf_names = {name for name in self.needed if name.startswith("fullhistory_mf_")}
        _require(mf_names <= set(supplied_mf), "required MF deploy report is missing")
        for name in mf_names:
            report_path = supplied_mf[name]
            report = _read_json(report_path)
            _require(
                report.get("kind") == "d4_full_history_implicit_mf_deploy_v1",
                f"MF report kind differs: {name}",
            )
            _require(
                report.get("decision") == "READY_FOR_CAUSAL_TEST_INFERENCE",
                f"MF report is not ready: {name}",
            )
            _require(report.get("uses_test_labels") is False, f"MF test-label leak: {name}")
            _require(int(report["test_cutoff"]) == self.test_cutoff, f"MF cutoff differs: {name}")
            _require(int(report["history_rows"]) == self.history_rows, f"MF history differs: {name}")
            checkpoint = Path(report["checkpoint"]["path"])
            _require(_sha256(checkpoint) == report["checkpoint"]["sha256"], f"MF hash differs: {name}")
            model, source_ids, item_ids = implicit_mf_jittor.load_checkpoint(checkpoint)
            _require(
                data_features.sha256_array(source_ids) == report["vocabulary"]["source_ids_sha256"],
                f"MF source vocabulary differs: {name}",
            )
            _require(
                data_features.sha256_array(item_ids) == report["vocabulary"]["item_ids_sha256"],
                f"MF item vocabulary differs: {name}",
            )
            self.mf_models[name] = (model, source_ids, item_ids)
            self.mf_reports[name] = {
                "path": str(report_path),
                "sha256": _sha256(report_path),
            }

        self.transition_mf_models = {}
        self.transition_mf_reports = {}
        supplied_transition_mf = {
            name: Path(path).resolve() for name, path in args.transition_mf_report
        }
        transition_mf_names = {
            name for name in self.needed if name.startswith("transition_mf_")
        }
        _require(
            transition_mf_names <= set(supplied_transition_mf),
            "required transition MF deploy report is missing",
        )
        for name in transition_mf_names:
            report_path = supplied_transition_mf[name]
            report = _read_json(report_path)
            _require(
                report.get("kind") == "d4_full_history_transition_mf_deploy_v1",
                f"transition MF report kind differs: {name}",
            )
            _require(
                report.get("decision") == "READY_FOR_CAUSAL_TEST_INFERENCE",
                f"transition MF report is not ready: {name}",
            )
            _require(report.get("uses_test_labels") is False, f"transition MF test-label leak: {name}")
            _require(int(report["test_cutoff"]) == self.test_cutoff, f"transition MF cutoff differs: {name}")
            _require(int(report["history_rows"]) == self.history_rows, f"transition MF history differs: {name}")
            _require(
                report.get("source_hashes", {}).get("d4_transition_mf_deploy.py")
                == _sha256(Path(d4_transition_mf_deploy.__file__).resolve()),
                f"transition MF deploy source differs: {name}",
            )
            checkpoint = Path(report["checkpoint"]["path"])
            _require(
                _sha256(checkpoint) == report["checkpoint"]["sha256"],
                f"transition MF hash differs: {name}",
            )
            model, source_ids, item_ids = implicit_mf_jittor.load_checkpoint(checkpoint)
            item_hash = data_features.sha256_array(item_ids)
            _require(
                np.array_equal(source_ids, item_ids)
                and item_hash == report["vocabulary"]["item_ids_sha256"]
                and item_hash
                == data_features.sha256_array(self.history.vocabulary.item_ids),
                f"transition MF item vocabulary differs: {name}",
            )
            transition_metadata = self.transition_index.metadata()
            _require(
                report["query_state"]["source_ids_sha256"]
                == transition_metadata["source_ids_sha256"]
                and report["query_state"]["last_items_sha256"]
                == transition_metadata["last_items_sha256"],
                f"transition MF query state differs: {name}",
            )
            self.transition_mf_models[name] = (model, source_ids, item_ids)
            self.transition_mf_reports[name] = {
                "path": str(report_path),
                "sha256": _sha256(report_path),
            }

        self.residual_models = []
        self.residual_weights = np.empty(0, dtype=np.float32)
        self.residual_static_mean = None
        self.residual_static_std = None
        self.residual_alpha = 0.0
        self.pairnew_record = None
        if self.pairnew is not None:
            members = self.pairnew.get("training", {}).get("members")
            _require(isinstance(members, list) and len(members) == 6, "pair-new member count differs")
            _require(
                sorted(int(record["hidden"]) for record in members) == [64, 64, 64, 96, 96, 96],
                "pair-new capacity ensemble differs",
            )
            _require(
                len({int(record["seed"]) for record in members}) == len(members),
                "pair-new seeds are not independent",
            )
            static_mean = np.asarray(self.pairnew["training"]["static_mean"], dtype=np.float32)
            static_std = np.asarray(self.pairnew["training"]["static_std"], dtype=np.float32)
            expected_count = len(self.residual_component_names) + len(data_features.FEATURE_NAMES) + 2
            _require(
                int(self.pairnew["feature_count"]) == expected_count
                and static_mean.shape == static_std.shape == (len(data_features.FEATURE_NAMES),)
                and np.isfinite(static_mean).all()
                and np.isfinite(static_std).all()
                and np.all(static_std > 0.0),
                "pair-new feature normalization differs",
            )
            for record in members:
                checkpoint = Path(record["checkpoint"])
                _require(_sha256(checkpoint) == record["sha256"], "pair-new checkpoint hash differs")
                model, metadata = pairnew_transformer_jittor.load_checkpoint(checkpoint)
                _require(
                    metadata["feature_count"] == expected_count
                    and metadata["hidden"] == int(record["hidden"])
                    and metadata["seed"] == int(record["seed"])
                    and metadata["epoch"] == int(record["best_epoch"])
                    and np.array_equal(metadata["static_mean"], static_mean)
                    and np.array_equal(metadata["static_std"], static_std),
                    "pair-new checkpoint metadata differs",
                )
                self.residual_models.append(model)
            self.residual_static_mean = static_mean
            self.residual_static_std = static_std
            self.residual_alpha = float(self.pairnew["residual_alpha"])
            weights = np.asarray(
                self.pairnew.get("member_weights", np.ones(len(members))),
                dtype=np.float32,
            )
            _require(
                weights.shape == (len(members),)
                and np.isfinite(weights).all()
                and np.all(weights >= 0.0)
                and np.isclose(float(weights.sum()), 1.0),
                "pair-new member weights differ",
            )
            self.residual_weights = weights
            self.pairnew_record = {
                "path": str(self.pairnew_path),
                "sha256": _sha256(self.pairnew_path),
                "kind": self.pairnew["kind"],
                "member_count": len(self.residual_models),
                "alpha": self.residual_alpha,
                "training_rows_per_replay": int(
                    self.pairnew["training"]["requested_rows_per_replay"]
                ),
            }

        self.fullset_models = []
        self.fullset_static_mean = None
        self.fullset_static_std = None
        self.fullset_policy = None
        self.fullset_record = None
        if self.fullset is not None:
            members = self.fullset.get("training", {}).get("members")
            _require(
                isinstance(members, list) and len(members) >= 4,
                "full-set member count differs",
            )
            _require(
                len({int(record["seed"]) for record in members}) == len(members),
                "full-set seeds are not independent",
            )
            static_mean = np.asarray(
                self.fullset["training"]["static_mean"], dtype=np.float32
            )
            static_std = np.asarray(
                self.fullset["training"]["static_std"], dtype=np.float32
            )
            expected_names = fullset_ranker_jittor.feature_names(
                self.residual_component_names
            )
            _require(
                static_mean.shape == static_std.shape
                == (len(data_features.FEATURE_NAMES),)
                and np.isfinite(static_mean).all()
                and np.isfinite(static_std).all()
                and np.all(static_std > 0.0),
                "full-set feature normalization differs",
            )
            for record in members:
                checkpoint = Path(record["checkpoint"])
                _require(
                    _sha256(checkpoint) == record["sha256"],
                    "full-set checkpoint hash differs",
                )
                model, metadata = fullset_ranker_jittor.load_checkpoint(checkpoint)
                _require(
                    metadata["feature_names"] == expected_names
                    and metadata["feature_count"] == len(expected_names)
                    and metadata["hidden"] == int(record["hidden"])
                    and metadata["layers"] == int(record["layers"])
                    and metadata["heads"] == int(record["heads"])
                    and metadata["seed"] == int(record["seed"])
                    and metadata["loss"] == str(record["loss"])
                    and metadata["epoch"] == int(record["best_epoch"])
                    and np.array_equal(metadata["static_mean"], static_mean)
                    and np.array_equal(metadata["static_std"], static_std),
                    "full-set checkpoint metadata differs",
                )
                self.fullset_models.append(model)
            self.fullset_static_mean = static_mean
            self.fullset_static_std = static_std
            self.fullset_policy = dict(self.fullset["policy"])
            self.fullset_record = {
                "path": str(self.fullset_path),
                "sha256": _sha256(self.fullset_path),
                "member_count": len(self.fullset_models),
                "policy": self.fullset_policy,
            }

        available = set(self.temporal_models) | set(self.mf_models) | set(
            self.transition_mf_models
        ) | {
            "test_frequency",
            "pair_seen_recency",
            "pool_source_log_count",
            "pool_source_excess",
            "pool_source_nearest",
            "pool_source_nearest_low",
            "pool_source_nearest_mid",
            "pool_source_nearest_high",
            "pool_source_leaveone_lift",
            "pool_source_leaveone_deviance",
            "pool_source_leaveone_deviance_low",
            "pool_source_leaveone_deviance_mid",
            "pool_source_leaveone_deviance_high",
            "transition_last_count",
        }
        _require(self.needed <= available, f"unsupported fitted components: {sorted(self.needed - available)}")
        self.temporal_reports = temporal_reports

    def score(self, chunk: Any, batch_rows: int) -> np.ndarray:
        duplicate_rows = _duplicate_candidate_rows(chunk.candidates)
        self.scored_rows += len(chunk.src)
        self.duplicate_candidate_rows += int(np.count_nonzero(duplicate_rows))
        features = self.store.features(chunk.src, chunk.time, chunk.candidates)
        candidate_seen = features[:, :, PAIR_SEEN_INDEX] > 0.0
        values: dict[str, np.ndarray] = {}
        if self.temporal_models:
            temporal = self.history.lookup(chunk.src, chunk.time, chunk.candidates)
            for name, (history_size, model) in self.temporal_models.items():
                values[name] = temporal_attention_jittor.predict_scores(
                    model,
                    temporal.source_indices,
                    temporal.candidate_indices,
                    temporal.history_item_indices[:, -history_size:],
                    temporal.log_time_deltas[:, -history_size:],
                    features=features,
                    batch_size=batch_rows,
                )
        for name, (model, source_ids, item_ids) in self.mf_models.items():
            values[name] = implicit_mf_jittor.predict_scores(
                model,
                chunk.src,
                chunk.candidates,
                source_ids,
                item_ids,
                batch_size=batch_rows,
            )
        if self.transition_mf_models:
            positions = np.searchsorted(self.transition_index.source_ids, chunk.src)
            inside = positions < len(self.transition_index.source_ids)
            matched = np.zeros(len(chunk.src), dtype=bool)
            matched[inside] = self.transition_index.source_ids[positions[inside]] == chunk.src[inside]
            previous = np.zeros(len(chunk.src), dtype=np.uint32)
            previous[matched] = self.transition_index.last_items[positions[matched]]
            for name, (model, source_ids, item_ids) in self.transition_mf_models.items():
                values[name] = implicit_mf_jittor.predict_scores(
                    model,
                    previous,
                    chunk.candidates,
                    source_ids,
                    item_ids,
                    batch_size=batch_rows,
                )
        if "test_frequency" in self.needed:
            values["test_frequency"] = _candidate_counts(
                chunk.candidates, self.test_ids, self.test_counts
            )
        if "pair_seen_recency" in self.needed:
            values["pair_seen_recency"] = (
                3.0 * features[:, :, PAIR_SEEN_INDEX]
                + features[:, :, PAIR_RECENCY_INDEX]
                + features[:, :, PAIR_LOG_COUNT_INDEX]
            )
        if "transition_last_count" in self.needed:
            values["transition_last_count"] = self.transition_index.score(
                chunk.src, chunk.candidates
            )
        if any(name.startswith("pool_source_") for name in self.needed):
            (
                pool_log,
                pool_excess,
                nearest_global,
                nearest_low,
                nearest_mid,
                nearest_high,
                leaveone_lift,
                leaveone_deviance,
                leaveone_deviance_low,
                leaveone_deviance_mid,
                leaveone_deviance_high,
            ) = _pool_source_components(
                chunk.src,
                chunk.time,
                chunk.candidates,
                self.repeated_source_pair_ids,
                self.repeated_source_pair_counts,
                self.test_source_ids,
                self.test_source_counts,
                self.test_ids,
                self.test_counts,
                self.source_pair_time_index,
                self.source_pair_time_scores,
                self.activity_cuts,
            )
            if "pool_source_log_count" in self.needed:
                values["pool_source_log_count"] = pool_log
            if "pool_source_excess" in self.needed:
                values["pool_source_excess"] = pool_excess
            for name, score in (
                ("pool_source_nearest", nearest_global),
                ("pool_source_nearest_low", nearest_low),
                ("pool_source_nearest_mid", nearest_mid),
                ("pool_source_nearest_high", nearest_high),
                ("pool_source_leaveone_lift", leaveone_lift),
                ("pool_source_leaveone_deviance", leaveone_deviance),
                ("pool_source_leaveone_deviance_low", leaveone_deviance_low),
                ("pool_source_leaveone_deviance_mid", leaveone_deviance_mid),
                ("pool_source_leaveone_deviance_high", leaveone_deviance_high),
            ):
                if name in self.needed:
                    values[name] = score
        normalized = {name: _qnorm(score) for name, score in values.items()}
        mixed = sum(
            self.weights[name] * normalized[name]
            for name in self.names
            if self.weights[name] > 1e-12
        )
        base = normalized[self.base_name]
        alpha = np.where(candidate_seen, self.seen_alpha, self.new_alpha).astype(np.float32)
        control = base + alpha * (mixed - base)
        if not self.residual_models:
            return _apply_duplicate_candidate_gate(
                control, control, chunk.candidates, duplicate_rows
            )
        component_scores = np.stack(
            [normalized[name] for name in self.residual_component_names], axis=0
        )
        residual_feature = pairnew_transformer_jittor._features(
            component_scores,
            features,
            control,
            candidate_seen,
            self.residual_static_mean,
            self.residual_static_std,
        )
        predictions = [
            pairnew_transformer_jittor._predict(model, residual_feature, batch_rows)
            for model in self.residual_models
        ]
        residual = _qnorm(
            np.sum(
                np.stack(predictions, axis=0)
                * self.residual_weights[:, None, None],
                axis=0,
            )
        )
        if self.fullset_models:
            fullset_features = fullset_ranker_jittor.build_features(
                component_scores,
                features,
                control,
                candidate_seen,
                self.fullset_static_mean,
                self.fullset_static_std,
                self.residual_component_names,
            )
            fullset_residual = _qnorm(
                np.mean(
                    [
                        fullset_ranker_jittor.predict(
                            model, fullset_features, batch_rows
                        )
                        for model in self.fullset_models
                    ],
                    axis=0,
                )
            )
            weight = float(self.fullset_policy["new_residual_weight"])
            residual = _qnorm(
                (1.0 - weight) * residual + weight * fullset_residual
            )
            candidate = fullset_ranker_jittor._candidate_policy(
                control, residual, candidate_seen, self.fullset_policy
            )
        else:
            candidate = pairnew_transformer_jittor._candidate_score(
                control, residual, candidate_seen, self.residual_alpha
            )
        return _apply_duplicate_candidate_gate(
            candidate, control, chunk.candidates, duplicate_rows
        )


def source_hashes() -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[1]
    source_files = (
        Path(__file__).resolve(),
        Path(__file__).with_name("d4_multimodel_shard.py").resolve(),
        Path(__file__).with_name("d4_multimodel_fit.py").resolve(),
        Path(d4_transition_mf_deploy.__file__).resolve(),
        Path(data_features.__file__).resolve(),
        Path(fullset_ranker_jittor.__file__).resolve(),
        Path(implicit_mf_jittor.__file__).resolve(),
        Path(pairnew_transformer_jittor.__file__).resolve(),
        Path(pool_association.__file__).resolve(),
        Path(replay_score_cache.__file__).resolve(),
        Path(d4_transition_research.__file__).resolve(),
        Path(temporal_attention_jittor.__file__).resolve(),
        Path(temporal_history.__file__).resolve(),
        Path(temporal_infer.__file__).resolve(),
        Path(verify_run.__file__).resolve(),
    )
    return {
        path.relative_to(source_root).as_posix(): _sha256(path)
        for path in source_files
    }


def dataset4_manifest(ensemble: D4Ensemble) -> dict[str, Any]:
    return {
        "fit_report": {
            "path": str(ensemble.fit_path),
            "sha256": _sha256(ensemble.fit_path),
        },
        "pairnew_report": ensemble.pairnew_record,
        "fullset_report": ensemble.fullset_record,
        "temporal_reports": ensemble.temporal_reports,
        "mf_reports": ensemble.mf_reports,
        "transition_mf_reports": ensemble.transition_mf_reports,
        "component_names": ensemble.names,
        "residual_component_names": ensemble.residual_component_names,
        "active_components": [
            name for name in ensemble.names if ensemble.weights[name] > 1e-12
        ],
        "weights": ensemble.weights,
        "base_component": ensemble.base_name,
        "seen_alpha": ensemble.seen_alpha,
        "new_alpha": ensemble.new_alpha,
        "residual_alpha": ensemble.residual_alpha,
        "candidate_gate": (
            "v12 rank slots plus confidence-gated pair-new escape"
            if ensemble.fullset_models
            else "pair-new candidates are reordered only within their frozen control rank slots"
            if ensemble.residual_models
            else None
        ),
        "duplicate_candidate_gate": {
            "policy": (
                "rows containing duplicate candidate ids use frozen control scores; "
                "repeated-id columns use the first occurrence score"
            ),
            "gated_rows": int(ensemble.duplicate_candidate_rows),
            "scored_rows": int(ensemble.scored_rows),
            "gated_row_rate": (
                float(ensemble.duplicate_candidate_rows / ensemble.scored_rows)
                if ensemble.scored_rows
                else 0.0
            ),
        },
        "pair_seen_rank_invariant": bool(ensemble.residual_models)
        and not ensemble.fullset_models,
        "pair_seen_rank_invariant_scope": (
            "rows with unique candidate ids; duplicate rows use id-canonicalized control"
            if ensemble.residual_models and not ensemble.fullset_models
            else None
        ),
        "test_cutoff": int(ensemble.test_cutoff),
        "history_rows": int(ensemble.history_rows),
        "pool_activity": {
            "quantiles": [0.5, 0.9],
            "test_cuts": list(ensemble.activity_cuts),
            "pair_time_index": ensemble.source_pair_time_metadata,
        },
        "transition_index": (
            ensemble.transition_index.metadata()
            if ensemble.transition_index is not None
            else None
        ),
        "uses_test_labels": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    manifest_path = args.output.with_suffix(".manifest.json")
    _require(
        data_features.sha256_file(args.data) == verify_run.EXPECTED_DATA_SHA256,
        "official data_B.zip SHA-256 differs",
    )
    _require(args.output.suffix == ".zip", "output must end in .zip")
    _require(not args.output.exists() and not manifest_path.exists(), "output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporal_attention_jittor.configure_cuda()
    ensemble = D4Ensemble(args)

    temporary_output = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    source_root = Path(__file__).resolve().parents[1]
    hashes = source_hashes()
    try:
        with zipfile.ZipFile(
            temporary_output,
            "x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            d3_record = _copy_d3(
                args.dataset3_source.resolve(),
                args.dataset3_manifest.resolve(),
                archive,
            )
            rows = 0
            with archive.open("dataset4.csv", "w", force_zip64=True) as raw:
                with io.TextIOWrapper(raw, encoding="ascii", newline="\n") as text:
                    for chunk in data_features.iter_test_chunks(
                        args.data, "dataset4", chunk_rows=args.test_chunk_rows
                    ):
                        probabilities = temporal_infer._probabilities(
                            ensemble.score(chunk, args.predict_batch_rows)
                        )
                        np.savetxt(
                            text,
                            probabilities,
                            fmt="%.8f",
                            delimiter=",",
                            newline="\n",
                        )
                        rows += len(chunk.src)
                        if args.verbose and rows % 100000 < len(chunk.src):
                            print(f"dataset4 {rows}/{ROWS['dataset4']}", flush=True)
            _require(rows == ROWS["dataset4"], "D4 test row count differs")
            _require(ensemble.scored_rows == rows, "D4 scored row count differs")

        manifest = {
            "kind": verify_run.MULTIMODEL_INFERENCE_MANIFEST_KIND,
            "algorithm_kind": (
                ensemble.fullset["kind"]
                if ensemble.fullset is not None
                else ensemble.pairnew["kind"]
                if ensemble.pairnew is not None
                else ensemble.fit["kind"]
            ),
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "data_sha256": verify_run.EXPECTED_DATA_SHA256,
            "submission_sha256": _sha256(temporary_output),
            "source_hashes": hashes,
            "row_counts": ROWS,
            "format": "exactly dataset3.csv,dataset4.csv; headerless ASCII; %.8f probabilities",
            "dataset3": d3_record,
            "dataset4": dataset4_manifest(ensemble),
            "jittor_runtime": {
                "version": str(temporal_attention_jittor.jt.__version__),
                "has_cuda": bool(temporal_attention_jittor.jt.has_cuda),
                "use_cuda": bool(temporal_attention_jittor.jt.flags.use_cuda),
            },
        }
        _atomic_json(temporary_manifest, manifest)
        verification = verify_run.verify_run(
            args.data,
            temporary_output,
            temporary_manifest,
            source_root=source_root,
        )
        _publish_new(temporary_output, args.output)
        _publish_new(temporary_manifest, manifest_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise
    finally:
        del ensemble
        gc.collect()
    return {
        "kind": (
            "d4_fullset_multiscale_lambdamrr_inference_result_v15"
            if manifest["dataset4"]["fullset_report"] is not None
            else "d4_pairnew_rank_slot_scaled_replay_inference_result_v21"
            if manifest["algorithm_kind"]
            in {
                "d4_pairnew_rank_slot_scaled_replay_transformer_v21",
                "d4_pairnew_rank_slot_weighted_scaled_v21_c2",
            }
            else "d4_pairnew_rank_slot_inference_result_v12"
        ),
        "decision": "PASS",
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "manifest": str(manifest_path),
        "verification": verification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--pairnew-report", type=Path)
    parser.add_argument("--fullset-report", type=Path)
    parser.add_argument(
        "--temporal-report", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--mf-report",
        nargs=2,
        action="append",
        metavar=("NAME", "REPORT"),
        default=[],
    )
    parser.add_argument(
        "--transition-mf-report",
        nargs=2,
        action="append",
        metavar=("NAME", "REPORT"),
        default=[],
    )
    parser.add_argument("--dataset3-source", type=Path, required=True)
    parser.add_argument("--dataset3-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-chunk-rows", type=int, default=250000)
    parser.add_argument("--test-chunk-rows", type=int, default=1024)
    parser.add_argument("--predict-batch-rows", type=int, default=512)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    try:
        print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "kind": "d4_fullset_multiscale_lambdamrr_inference_result_v15",
                    "decision": "ERROR",
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
