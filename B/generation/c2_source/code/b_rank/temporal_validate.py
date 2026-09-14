#!/usr/bin/env python3
"""Research-only temporal-attention validation for Dataset3 and Dataset4.

This entry point intentionally stops after time-separated validation.  It
cannot write a submission ZIP.  A later inference path is allowed only after
the selected protocol has survived this replay and an independent confirmation
block.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from . import data_features, temporal_attention_jittor, temporal_history, verify_run
except ImportError:
    import data_features
    import temporal_attention_jittor
    import temporal_history
    import verify_run


SCENES = ("dataset3", "dataset4")
SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(data_features.__file__).resolve(),
    Path(temporal_attention_jittor.__file__).resolve(),
    Path(temporal_history.__file__).resolve(),
    Path(verify_run.__file__).resolve(),
)


class ProtocolError(RuntimeError):
    """Raised when a temporal validation input would break the audit contract."""


@dataclass(frozen=True)
class TrainingArrays:
    source: np.memmap
    candidates: np.memmap
    history: np.memmap
    history_gap: np.memmap
    features: np.memmap | None
    labels: np.memmap
    sample_weight: np.memmap | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(root).as_posix(): data_features.sha256_file(path)
        for path in SOURCE_FILES
    }


def _new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to reuse run directory: {path}")
    path.mkdir(parents=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mode(scene: str) -> str:
    if scene == "dataset3":
        return "shared"
    if scene == "dataset4":
        return "bipartite"
    raise ProtocolError(f"unknown scene: {scene}")


def _group_sizes(args: argparse.Namespace) -> dict[str, int]:
    values = {
        "train": int(args.train_groups),
        "valid": int(args.valid_groups),
        "confirm": int(args.confirm_groups),
    }
    if any(value < 1 for value in values.values()):
        raise ProtocolError("all replay group sizes must be positive")
    return values


def _write_training_arrays(
    group: Any,
    history: temporal_history.TemporalHistory,
    root: Path,
    *,
    batch_rows: int,
    source_hot_store: Any | None = None,
    source_hot_weight: float = 1.0,
    static_feature_store: Any | None = None,
) -> TrainingArrays:
    if root.exists():
        raise FileExistsError(f"refusing to reuse temporal training arrays: {root}")
    rows = int(group.rows)
    width = int(group.candidate_count)
    _require(width == 100, "official candidate width must be 100")
    try:
        source_hot_weight = float(source_hot_weight)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("source-hot weight must be numeric") from exc
    _require(
        np.isfinite(source_hot_weight) and source_hot_weight >= 1.0,
        "source-hot weight must be finite and at least one",
    )
    weighted = source_hot_weight != 1.0
    static_features = static_feature_store is not None
    if weighted:
        _require(source_hot_store is not None, "hot-source weighting needs a frozen feature store")
        _require(
            int(source_hot_store.cutoff) == int(group.cutoff),
            "hot-source feature-store cutoff must equal the training group cutoff",
        )
    if static_features:
        _require(
            int(static_feature_store.cutoff) == int(group.cutoff),
            "static-feature store cutoff must equal the training group cutoff",
        )
        _require(
            int(static_feature_store.feature_dim) == len(data_features.FEATURE_NAMES),
            "static-feature store must expose the audited feature width",
        )
    root.mkdir(parents=True)
    arrays = {
        "source": np.lib.format.open_memmap(root / "source.npy", mode="w+", dtype=np.int32, shape=(rows,)),
        "candidates": np.lib.format.open_memmap(root / "candidates.npy", mode="w+", dtype=np.int32, shape=(rows, width)),
        "history": np.lib.format.open_memmap(root / "history.npy", mode="w+", dtype=np.int32, shape=(rows, history.history_size)),
        "history_gap": np.lib.format.open_memmap(root / "history_gap.npy", mode="w+", dtype=np.float32, shape=(rows, history.history_size)),
        "labels": np.lib.format.open_memmap(root / "labels.npy", mode="w+", dtype=np.int32, shape=(rows,)),
    }
    if static_features:
        arrays["features"] = np.lib.format.open_memmap(
            root / "features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(rows, width, len(data_features.FEATURE_NAMES)),
        )
    if weighted:
        arrays["sample_weight"] = np.lib.format.open_memmap(
            root / "sample_weight.npy", mode="w+", dtype=np.float32, shape=(rows,)
        )
    try:
        offset = 0
        for batch in group.iter_batches(batch_rows=batch_rows):
            temporal = history.lookup(batch.src, batch.time, batch.candidates)
            stop = offset + len(batch.src)
            arrays["source"][offset:stop] = temporal.source_indices
            arrays["candidates"][offset:stop] = temporal.candidate_indices
            arrays["history"][offset:stop] = temporal.history_item_indices
            arrays["history_gap"][offset:stop] = temporal.log_time_deltas
            arrays["labels"][offset:stop] = batch.labels
            if static_features:
                arrays["features"][offset:stop] = static_feature_store.features(
                    batch.src, batch.time, batch.candidates
                )
            if weighted:
                source_hot = source_hot_store.source_hot_mask(batch.src)
                arrays["sample_weight"][offset:stop] = np.where(
                    source_hot, float(source_hot_weight), 1.0
                )
            offset = stop
        _require(offset == rows, "candidate group row count changed during temporal materialization")
        for value in arrays.values():
            value.flush()
    finally:
        for value in arrays.values():
            value.flush()
    return TrainingArrays(
        source=np.load(root / "source.npy", mmap_mode="r", allow_pickle=False),
        candidates=np.load(root / "candidates.npy", mmap_mode="r", allow_pickle=False),
        history=np.load(root / "history.npy", mmap_mode="r", allow_pickle=False),
        history_gap=np.load(root / "history_gap.npy", mmap_mode="r", allow_pickle=False),
        features=(
            np.load(root / "features.npy", mmap_mode="r", allow_pickle=False)
            if static_features
            else None
        ),
        labels=np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False),
        sample_weight=(
            np.load(root / "sample_weight.npy", mmap_mode="r", allow_pickle=False)
            if weighted
            else None
        ),
    )


def _evaluate(
    cache: Any,
    group: Any,
    history: temporal_history.TemporalHistory,
    model: Any,
    *,
    batch_rows: int,
    static_features: bool = False,
) -> dict[str, Any]:
    """Score a group with temporal inputs; FeatureStore supplies hot segments."""
    store = cache.feature_store(group.cutoff)
    _require(int(store.cutoff) == int(group.cutoff), "evaluation feature-store cutoff mismatch")
    if static_features:
        _require(
            int(store.feature_dim) == len(data_features.FEATURE_NAMES),
            "evaluation static-feature store must expose the audited feature width",
        )
    metrics = data_features.RankingMetrics()
    for batch in group.iter_batches(batch_rows=batch_rows):
        temporal = history.lookup(batch.src, batch.time, batch.candidates)
        features = (
            store.features(batch.src, batch.time, batch.candidates)
            if static_features
            else None
        )
        score = temporal_attention_jittor.predict_scores(
            model,
            temporal.source_indices,
            temporal.candidate_indices,
            temporal.history_item_indices,
            temporal.log_time_deltas,
            features=features,
            batch_size=batch_rows,
        )
        segments = store.evaluation_segments(
            batch.src, batch.time, batch.candidates, batch.labels
        )
        metrics.update(score, batch.labels, segments=segments)
    return metrics.result()


def _history_index(
    cache: Any,
    *,
    history_size: int,
    scene: str,
    cutoff: int,
    vocabulary: temporal_history.TemporalVocabulary | None = None,
) -> temporal_history.TemporalHistory:
    return temporal_history.TemporalHistory.build(
        cache.src,
        cache.dst,
        cache.time,
        history_size=history_size,
        id_mode=_mode(scene),
        cutoff=cutoff,
        vocabulary=vocabulary,
    )


def _scene_report(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    cache = data_features.BDataCache.build_or_open(
        args.data, scene, args.cache_dir, chunk_rows=int(args.cache_chunk_rows), verify_hash=True
    )
    groups = data_features.build_split1_groups(
        cache,
        seed=int(args.group_seed),
        sizes=_group_sizes(args),
        batch_rows=int(args.group_batch_rows),
        negative_strategy="history",
    )
    plan = groups.plan
    # A validation query must not acquire prior labels from its own held-out
    # block.  This mirrors FeatureStore's frozen group cutoff and the official
    # test setting, where prior test destinations are unavailable.
    train_index = _history_index(
        cache,
        history_size=int(args.history_size),
        scene=scene,
        cutoff=int(plan.cutoffs["valid"]),
    )
    vocabulary = train_index.vocabulary
    valid_index = train_index
    confirm_index = _history_index(
        cache,
        history_size=int(args.history_size),
        scene=scene,
        cutoff=int(plan.cutoffs["confirm"]),
        vocabulary=vocabulary,
    )
    _require(
        train_index.history_rows == cache.history_end(int(plan.cutoffs["valid"])),
        "train temporal history does not end at the validation boundary",
    )
    _require(
        valid_index.history_rows == cache.history_end(int(plan.cutoffs["valid"])),
        "validation temporal history includes held-out validation labels",
    )
    _require(
        confirm_index.history_rows == cache.history_end(int(plan.cutoffs["confirm"])),
        "confirmation temporal history includes held-out confirmation labels",
    )
    source_hot_weight = float(args.source_hot_weight)
    static_features = bool(args.static_features)
    static_context = bool(args.static_context)
    static_context_pair_seen_only = bool(args.static_context_pair_seen_only)
    train_store = (
        cache.feature_store(groups.train.cutoff)
        if source_hot_weight != 1.0 or static_features
        else None
    )
    train_hot_store = train_store if source_hot_weight != 1.0 else None
    train_static_store = train_store if static_features else None
    arrays = _write_training_arrays(
        groups.train,
        train_index,
        args.run_dir / "arrays" / scene / "train",
        batch_rows=int(args.group_batch_rows),
        source_hot_store=train_hot_store,
        source_hot_weight=source_hot_weight,
        static_feature_store=train_static_store,
    )
    candidates: dict[str, Any] = {}
    for seed in args.seeds:
        model, losses = temporal_attention_jittor.train_ranker(
            arrays.source,
            arrays.candidates,
            arrays.history,
            arrays.history_gap,
            arrays.labels,
            source_count=train_index.source_vocab_size,
            item_count=train_index.item_vocab_size,
            features=arrays.features,
            sample_weight=arrays.sample_weight,
            embedding_dim=int(args.embedding_dim),
            static_context=static_context,
            static_context_pair_seen_only=static_context_pair_seen_only,
            dropout=float(args.dropout),
            time_scale=float(args.time_scale),
            epochs=int(args.epochs),
            batch_size=int(args.train_batch_rows),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            seed=int(seed),
            verbose=bool(args.verbose),
        )
        checkpoint = temporal_attention_jittor.save_checkpoint(
            args.run_dir / "checkpoints" / scene / f"seed{seed}.npz", model
        )
        valid = _evaluate(
            cache,
            groups.valid,
            valid_index,
            model,
            batch_rows=int(args.eval_batch_rows),
            static_features=static_features,
        )
        confirmation = _evaluate(
            cache,
            groups.confirm,
            confirm_index,
            model,
            batch_rows=int(args.eval_batch_rows),
            static_features=static_features,
        )
        key = str(int(seed))
        candidates[key] = {
            "seed": int(seed),
            "checkpoint": str(checkpoint.relative_to(args.run_dir)),
            "checkpoint_sha256": data_features.sha256_file(checkpoint),
            "train_loss": [float(value) for value in losses],
            "validation": valid,
            "confirmation": confirmation,
        }
    selected_key = max(
        candidates,
        key=lambda key: (
            float(candidates[key]["validation"]["mrr"]),
            float(candidates[key]["validation"]["top1"]),
            -int(candidates[key]["seed"]),
        ),
    )
    selected = dict(candidates[selected_key])
    selected["selection_rule"] = "maximum primary history-replay validation MRR, then Top1; confirmation excluded"
    return {
        "scene": scene,
        "id_mode": _mode(scene),
        "groups": groups.metadata,
        "plan": plan.as_dict(),
        "vocabulary": {
            "source_ids_sha256": data_features.sha256_array(vocabulary.source_ids),
            "item_ids_sha256": data_features.sha256_array(vocabulary.item_ids),
            "source_vocab_size": train_index.source_vocab_size,
            "item_vocab_size": train_index.item_vocab_size,
        },
        "history_rows": {
            "train": train_index.history_rows,
            "valid": valid_index.history_rows,
            "confirm": confirm_index.history_rows,
        },
        "training_weighting": {
            "kind": "source_hot_loss_weight",
            "enabled": source_hot_weight != 1.0,
            "source_hot_weight": source_hot_weight,
            "source_hot_store_cutoff": (
                int(train_hot_store.cutoff) if train_hot_store is not None else None
            ),
            "source_hot_count": (
                int(train_hot_store.metadata["source_hot_count"])
                if train_hot_store is not None
                else None
            ),
            "source": "FeatureStore history strictly before train group cutoff",
        },
        "training_static_features": {
            "enabled": static_features,
            "context_enabled": static_context,
            "context_pair_seen_only": static_context_pair_seen_only,
            "feature_dim": len(data_features.FEATURE_NAMES) if static_features else 0,
            "feature_names": list(data_features.FEATURE_NAMES) if static_features else [],
            "store_cutoff": (
                int(train_static_store.cutoff) if train_static_store is not None else None
            ),
            "source": "FeatureStore history strictly before train group cutoff",
        },
        "candidates": candidates,
        "selected": selected,
        "selection_is_confirmation_blind": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.data = args.data.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    _require(data_features.sha256_file(args.data) == verify_run.EXPECTED_DATA_SHA256, "official data_B.zip SHA-256 differs")
    scenes = tuple(dict.fromkeys(args.scenes))
    _require(scenes and all(scene in SCENES for scene in scenes), "scenes must use dataset3 and/or dataset4")
    _require(int(args.history_size) > 0 and int(args.embedding_dim) > 0, "history-size and embedding-dim must be positive")
    _require(int(args.epochs) > 0 and int(args.train_batch_rows) > 0 and int(args.eval_batch_rows) > 0, "batch sizes and epochs must be positive")
    _require(0.0 <= float(args.dropout) < 1.0 and float(args.time_scale) >= 0.0, "invalid attention regularization")
    _require(float(args.learning_rate) > 0.0 and float(args.weight_decay) >= 0.0, "invalid optimizer hyperparameters")
    _require(
        np.isfinite(float(args.source_hot_weight)) and float(args.source_hot_weight) >= 1.0,
        "source-hot-weight must be finite and at least one",
    )
    _require(
        not bool(args.static_context) or bool(args.static_features),
        "--static-context requires --static-features",
    )
    _require(
        not bool(args.static_context_pair_seen_only) or bool(args.static_context),
        "--static-context-pair-seen-only requires --static-context",
    )
    _new_directory(args.run_dir)
    temporal_attention_jittor.configure_cuda()
    runtime = {
        "jittor": str(temporal_attention_jittor.jt.__version__),
        "has_cuda": bool(temporal_attention_jittor.jt.has_cuda),
        "use_cuda": bool(temporal_attention_jittor.jt.flags.use_cuda),
    }
    reports = {scene: _scene_report(args, scene) for scene in scenes}
    report = {
        "kind": "b_rank_temporal_attention_research_v1",
        "decision": "RESEARCH_ONLY",
        "created_utc": _utc_now(),
        "data": {"path": str(args.data), "sha256": verify_run.EXPECTED_DATA_SHA256},
        "source_hashes": _source_hashes(),
        "runtime": runtime,
        "config": {
            "scenes": list(scenes),
            "seeds": [int(seed) for seed in args.seeds],
            "group_seed": int(args.group_seed),
            "history_size": int(args.history_size),
            "embedding_dim": int(args.embedding_dim),
            "dropout": float(args.dropout),
            "time_scale": float(args.time_scale),
            "epochs": int(args.epochs),
            "train_batch_rows": int(args.train_batch_rows),
            "eval_batch_rows": int(args.eval_batch_rows),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "source_hot_weight": float(args.source_hot_weight),
            "static_features": bool(args.static_features),
            "static_context": bool(args.static_context),
            "static_context_pair_seen_only": bool(args.static_context_pair_seen_only),
            "group_sizes": _group_sizes(args),
            "primary_negative_strategy": "history",
        },
        "datasets": reports,
        "interpretation": "split1 MRR is a causal replay proxy. It is not an online-score claim and cannot itself authorize a submission.",
    }
    _atomic_json(args.run_dir / "research_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--scenes", choices=SCENES, nargs="+", default=list(SCENES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260810, 20260811, 20260812])
    parser.add_argument("--group-seed", type=int, default=20260810)
    parser.add_argument("--train-groups", type=int, default=100000)
    parser.add_argument("--valid-groups", type=int, default=30000)
    parser.add_argument("--confirm-groups", type=int, default=30000)
    parser.add_argument("--cache-chunk-rows", type=int, default=250000)
    parser.add_argument("--group-batch-rows", type=int, default=4096)
    parser.add_argument("--history-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--time-scale", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-rows", type=int, default=256)
    parser.add_argument("--eval-batch-rows", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--source-hot-weight",
        type=float,
        default=1.0,
        help="optional train-loss multiplier for sources hot before the train cutoff",
    )
    parser.add_argument(
        "--static-features",
        action="store_true",
        help="enable frozen causal pair, recency, and popularity features",
    )
    parser.add_argument(
        "--static-context",
        action="store_true",
        help="add a candidate-set context residual to the static-feature head",
    )
    parser.add_argument(
        "--static-context-pair-seen-only",
        action="store_true",
        help="apply the static-context residual only to causally seen source-destination pairs",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"kind": "b_rank_temporal_attention_research_v1", "decision": "ERROR", "error": f"{type(error).__name__}: {error}"}, indent=2, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
