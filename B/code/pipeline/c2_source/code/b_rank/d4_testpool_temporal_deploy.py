#!/usr/bin/env python3
"""Train full-split1 D4 temporal models with unlabeled test-pool negatives."""

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
    data_features,
    temporal_attention_jittor,
    temporal_history,
    temporal_infer,
    temporal_validate,
    verify_run,
)


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


def _full_testpool_group(
    cache: data_features.BDataCache,
    root: Path,
    *,
    seed: int,
    batch_rows: int,
) -> data_features.CandidateGroup:
    lower, upper = cache.split1_bounds()
    cutoff = int(cache.time[lower])
    pool, counts = cache.test_candidate_counts()
    probabilities = np.asarray(counts, dtype=np.float64)
    probabilities /= probabilities.sum()
    pool_metadata = {
        "strategy": "test_pool",
        "source": "unlabeled official test candidate cells only",
        "causal_primary_replay": False,
        "transductive_training_only": True,
        "uses_test_labels": False,
        "ids": int(len(pool)),
        "count_hash": data_features.sha256_array(counts),
    }
    data_features._write_candidate_group(
        destination=root,
        name="train",
        cache=cache,
        lower=lower,
        upper=upper,
        rows=upper - lower,
        cutoff=cutoff,
        pool=pool,
        probabilities=probabilities,
        cold_pool=None,
        cold_fraction=0.0,
        pool_metadata=pool_metadata,
        candidate_count=100,
        rng=np.random.default_rng(np.random.SeedSequence([int(seed), 1])),
        batch_rows=batch_rows,
    )
    group = data_features.CandidateGroup(root)
    group.validate_positive_uniqueness(batch_rows=batch_rows)
    return group


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = args.data.resolve()
    cache_dir = args.cache_dir.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing run directory reuse: {run_dir}")
    if data_features.sha256_file(data) != verify_run.EXPECTED_DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("at least one unique seed is required")
    run_dir.mkdir(parents=True)
    temporal_attention_jittor.configure_cuda()
    cache = data_features.BDataCache.build_or_open(
        data,
        "dataset4",
        cache_dir,
        chunk_rows=args.cache_chunk_rows,
        verify_hash=True,
    )
    test_cutoff = temporal_infer._test_history_cutoff(
        data_features, data, cache, chunk_rows=args.test_chunk_rows
    )
    history_rows = cache.history_end(test_cutoff)
    if history_rows != len(cache.src):
        raise ValueError("test cutoff does not include exactly all official training edges")
    group = _full_testpool_group(
        cache,
        run_dir / "replay" / "train",
        seed=args.group_seed,
        batch_rows=args.group_batch_rows,
    )
    history = temporal_history.TemporalHistory.build(
        cache.src,
        cache.dst,
        cache.time,
        history_size=args.history_size,
        id_mode="bipartite",
        cutoff=test_cutoff,
    )
    if history.history_rows != history_rows:
        raise ValueError("temporal history differs from the allowed training history")
    store = cache.feature_store(group.cutoff)
    arrays = temporal_validate._write_training_arrays(
        group,
        history,
        run_dir / "arrays",
        batch_rows=args.group_batch_rows,
        static_feature_store=store,
    )

    checkpoints = []
    for seed in args.seeds:
        name = f"temporal_testpool_seed{str(seed)[-2:]}"
        model, losses = temporal_attention_jittor.train_ranker(
            arrays.source,
            arrays.candidates,
            arrays.history,
            arrays.history_gap,
            arrays.labels,
            source_count=history.source_vocab_size,
            item_count=history.item_vocab_size,
            features=arrays.features,
            embedding_dim=args.embedding_dim,
            static_context=True,
            static_context_pair_seen_only=True,
            dropout=0.0,
            time_scale=0.25,
            epochs=args.epochs,
            batch_size=args.batch_rows,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=seed,
            verbose=args.verbose,
        )
        checkpoint = temporal_attention_jittor.save_checkpoint(
            run_dir / "checkpoints" / f"{name}.npz", model
        )
        restored, config = temporal_attention_jittor.load_checkpoint(checkpoint)
        if config != temporal_attention_jittor.model_config(model):
            raise ValueError(f"checkpoint save/load config differs: {name}")
        checkpoints.append(
            {
                "name": name,
                "history_size": int(args.history_size),
                "seed": int(seed),
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "config": config,
                "train_loss": [float(value) for value in losses],
            }
        )
        del restored, model
        temporal_attention_jittor.jt.gc()

    source_files = (
        Path(__file__).resolve(),
        Path(data_features.__file__).resolve(),
        Path(temporal_attention_jittor.__file__).resolve(),
        Path(temporal_history.__file__).resolve(),
        Path(temporal_infer.__file__).resolve(),
        Path(temporal_validate.__file__).resolve(),
    )
    report = {
        "kind": "d4_full_split1_testpool_temporal_deploy_v1",
        "decision": "READY_FOR_CAUSAL_TEST_INFERENCE",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_sha256": verify_run.EXPECTED_DATA_SHA256,
        "test_cutoff": int(test_cutoff),
        "history_rows": int(history_rows),
        "uses_test_labels": False,
        "training": {
            "rows": int(group.rows),
            "row_range": group.metadata["train_row_range"],
            "time_range": group.metadata["time_range"],
            "feature_cutoff": int(store.cutoff),
            "negative_pool": group.metadata["negative_pool"],
            "candidate_count": int(group.candidate_count),
            "history_size": int(args.history_size),
            "embedding_dim": int(args.embedding_dim),
            "epochs": int(args.epochs),
            "batch_rows": int(args.batch_rows),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "strict_history_rule": "edge_time < query_time",
        },
        "vocabulary": {
            "source_count": int(history.source_vocab_size),
            "item_count": int(history.item_vocab_size),
            "source_ids_sha256": data_features.sha256_array(history.vocabulary.source_ids),
            "item_ids_sha256": data_features.sha256_array(history.vocabulary.item_ids),
        },
        "checkpoints": checkpoints,
        "runtime": {
            "jittor": str(temporal_attention_jittor.jt.__version__),
            "has_cuda": bool(temporal_attention_jittor.jt.has_cuda),
            "use_cuda": bool(temporal_attention_jittor.jt.flags.use_cuda),
        },
        "source_hashes": {path.name: _sha256(path) for path in source_files},
    }
    _atomic_json(run_dir / "deploy_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260810, 20260811, 20260812])
    parser.add_argument("--group-seed", type=int, default=20260810)
    parser.add_argument("--history-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--group-batch-rows", type=int, default=4096)
    parser.add_argument("--cache-chunk-rows", type=int, default=250000)
    parser.add_argument("--test-chunk-rows", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
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
                    "kind": "d4_full_split1_testpool_temporal_deploy_v1",
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
