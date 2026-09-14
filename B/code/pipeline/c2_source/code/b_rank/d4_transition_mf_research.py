#!/usr/bin/env python3
"""Research a Jittor low-rank item-to-item transition model for D4."""

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

from . import data_features, implicit_mf_jittor, temporal_history, verify_run


EXACT_TRANSITION_BASELINE = {
    "history": {
        "validation": 0.098091626,
        "confirmation": 0.098521848,
    },
    "test_pool": {
        "validation": 0.094875719,
        "confirmation": 0.096784899,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_transition_data(
    cache: data_features.BDataCache, cutoff: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return strict adjacent item pairs and each source's last item."""
    stop = cache.history_end(cutoff)
    source = np.asarray(cache.src[:stop], dtype=np.uint32)
    item = np.asarray(cache.dst[:stop], dtype=np.uint32)
    time = np.asarray(cache.time[:stop], dtype=np.uint32)
    order = np.argsort(source, kind="stable")
    ordered_source = source[order]
    ordered_item = item[order]
    ordered_time = time[order]
    same_source = ordered_source[1:] == ordered_source[:-1]
    strict_time = ordered_time[1:] > ordered_time[:-1]
    transition = same_source & strict_time
    previous = ordered_item[:-1][transition].copy()
    following = ordered_item[1:][transition].copy()
    ends = np.r_[np.flatnonzero(ordered_source[1:] != ordered_source[:-1]), len(order) - 1]
    source_ids = ordered_source[ends].copy()
    last_items = ordered_item[ends].copy()
    item_ids = np.unique(item).astype(np.uint32, copy=False)
    if not len(previous) or not len(source_ids) or not len(item_ids):
        raise ValueError("strict transition history is empty")
    return previous, following, source_ids, last_items, item_ids


def _last_items_for_sources(
    source: np.ndarray, source_ids: np.ndarray, last_items: np.ndarray
) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint32)
    positions = np.searchsorted(source_ids, source)
    inside = positions < len(source_ids)
    output = np.zeros(source.shape, dtype=np.uint32)
    matched = np.zeros(source.shape, dtype=bool)
    matched[inside] = source_ids[positions[inside]] == source[inside]
    output[matched] = last_items[positions[matched]]
    return output


def _score_group(
    cache: data_features.BDataCache,
    group: Any,
    model: Any,
    item_ids: np.ndarray,
    source_ids: np.ndarray,
    last_items: np.ndarray,
    batch_rows: int,
) -> dict[str, Any]:
    scores = np.empty((group.rows, group.candidate_count), dtype=np.float32)
    labels = np.empty(group.rows, dtype=np.int64)
    segment_parts: dict[str, list[np.ndarray]] = {}
    offset = 0
    store = cache.feature_store(group.cutoff)
    for batch in group.iter_batches(batch_rows=batch_rows):
        stop = offset + len(batch.src)
        previous = _last_items_for_sources(batch.src, source_ids, last_items)
        scores[offset:stop] = implicit_mf_jittor.predict_scores(
            model,
            previous,
            batch.candidates,
            item_ids,
            item_ids,
            batch_size=batch_rows,
        )
        labels[offset:stop] = batch.labels
        for name, mask in store.evaluation_segments(
            batch.src, batch.time, batch.candidates, batch.labels
        ).items():
            segment_parts.setdefault(name, []).append(np.asarray(mask, dtype=bool))
        offset = stop
    if offset != group.rows:
        raise ValueError("candidate group row count changed")
    segments = {name: np.concatenate(parts) for name, parts in segment_parts.items()}
    return data_features.ranking_metrics(scores, labels, segments=segments)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = args.data.resolve()
    cache_dir = args.cache_dir.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing run directory reuse: {run_dir}")
    if data_features.sha256_file(data) != verify_run.EXPECTED_DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    if min(args.embedding_dim, args.negative_count, args.epochs, args.batch_rows) < 1:
        raise ValueError("model dimensions, negatives, epochs, and batch size must be positive")
    run_dir.mkdir(parents=True)
    implicit_mf_jittor.configure_cuda()
    cache = data_features.BDataCache.build_or_open(
        data, "dataset4", cache_dir, chunk_rows=args.cache_chunk_rows, verify_hash=True
    )
    groups = {
        strategy: data_features.build_split1_groups(
            cache,
            seed=20260810,
            sizes={"train": 1, "valid": args.valid_groups, "confirm": args.confirm_groups},
            batch_rows=4096,
            negative_strategy=strategy,
        )
        for strategy in ("history", "test_pool")
    }
    training_cutoff = int(groups["history"].valid.cutoff)
    previous, following, source_ids, last_items, item_ids = _strict_transition_data(
        cache, training_cutoff
    )
    model, losses = implicit_mf_jittor.train_full_history(
        previous,
        following,
        item_ids,
        item_ids,
        embedding_dim=args.embedding_dim,
        negative_count=args.negative_count,
        epochs=args.epochs,
        batch_size=args.batch_rows,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        verbose=args.verbose,
    )
    checkpoint = implicit_mf_jittor.save_checkpoint(
        run_dir / "checkpoints" / f"transition_seed{args.seed}.npz",
        model,
        item_ids,
        item_ids,
    )
    restored, restored_source_ids, restored_item_ids = implicit_mf_jittor.load_checkpoint(
        checkpoint
    )
    metrics = {
        strategy: {
            "validation": _score_group(
                cache, replay.valid, restored, restored_item_ids, source_ids, last_items, args.eval_batch_rows
            ),
            "confirmation": _score_group(
                cache, replay.confirm, restored, restored_item_ids, source_ids, last_items, args.eval_batch_rows
            ),
        }
        for strategy, replay in groups.items()
    }
    deltas = {
        strategy: {
            split: float(metrics[strategy][split]["mrr"] - EXACT_TRANSITION_BASELINE[strategy][split])
            for split in ("validation", "confirmation")
        }
        for strategy in metrics
    }
    source_files = (Path(__file__).resolve(), Path(implicit_mf_jittor.__file__).resolve())
    report = {
        "kind": "d4_transition_item_mf_research_v1",
        "decision": "RESEARCH_ONLY",
        "created_utc": _utc_now(),
        "data_sha256": verify_run.EXPECTED_DATA_SHA256,
        "uses_test_labels": False,
        "selection_rule": "training uses strict transitions before history-validation cutoff; labels are evaluation-only",
        "training": {
            "cutoff": training_cutoff,
            "transition_rows": len(previous),
            "item_vocab_size": len(item_ids) + 1,
            "source_vocab_size": len(item_ids) + 1,
            "query_source_count": len(source_ids),
            "embedding_dim": args.embedding_dim,
            "negative_count": args.negative_count,
            "epochs": args.epochs,
            "batch_rows": args.batch_rows,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "loss": losses,
        },
        "group_metadata": {strategy: replay.metadata for strategy, replay in groups.items()},
        "metrics": metrics,
        "exact_transition_baseline": EXACT_TRANSITION_BASELINE,
        "deltas_vs_exact_transition": deltas,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "runtime": {
            "jittor": str(implicit_mf_jittor.jt.__version__),
            "has_cuda": bool(implicit_mf_jittor.jt.has_cuda),
            "use_cuda": bool(implicit_mf_jittor.jt.flags.use_cuda),
        },
        "source_hashes": {path.name: _sha256(path) for path in source_files},
    }
    _atomic_json(run_dir / "research_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--negative-count", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--eval-batch-rows", type=int, default=2048)
    parser.add_argument("--valid-groups", type=int, default=30000)
    parser.add_argument("--confirm-groups", type=int, default=30000)
    parser.add_argument("--cache-chunk-rows", type=int, default=250000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    try:
        print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"kind": "d4_transition_item_mf_research_v1", "decision": "ERROR", "error": f"{type(error).__name__}: {error}"},
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
