#!/usr/bin/env python3
"""Train the validated item-transition MF on all causal D4 history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import (
    d4_transition_mf_research,
    data_features,
    implicit_mf_jittor,
    temporal_infer,
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = args.data.resolve()
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
        data,
        "dataset4",
        args.cache_dir.resolve(),
        chunk_rows=args.cache_chunk_rows,
        verify_hash=True,
    )
    test_cutoff = temporal_infer._test_history_cutoff(
        data_features, data, cache, chunk_rows=args.test_chunk_rows
    )
    history_rows = cache.history_end(test_cutoff)
    if history_rows != len(cache.src):
        raise ValueError("test cutoff does not include all official training edges")
    previous, following, source_ids, last_items, item_ids = (
        d4_transition_mf_research._strict_transition_data(cache, test_cutoff)
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
    restored, restored_source_ids, restored_item_ids = (
        implicit_mf_jittor.load_checkpoint(checkpoint)
    )
    if not (
        data_features.sha256_array(restored_source_ids)
        == data_features.sha256_array(item_ids)
        == data_features.sha256_array(restored_item_ids)
    ):
        raise ValueError("transition MF checkpoint vocabulary differs")
    del restored

    source_files = (
        Path(__file__).resolve(),
        Path(d4_transition_mf_research.__file__).resolve(),
        Path(implicit_mf_jittor.__file__).resolve(),
    )
    report = {
        "kind": "d4_full_history_transition_mf_deploy_v1",
        "decision": "READY_FOR_CAUSAL_TEST_INFERENCE",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_sha256": verify_run.EXPECTED_DATA_SHA256,
        "test_cutoff": int(test_cutoff),
        "history_rows": int(history_rows),
        "training": {
            "transition_rows": int(len(previous)),
            "embedding_dim": int(args.embedding_dim),
            "negative_count": int(args.negative_count),
            "epochs": int(args.epochs),
            "batch_rows": int(args.batch_rows),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "loss": [float(value) for value in losses],
            "strict_time_rule": "previous edge time < next edge time < test cutoff",
        },
        "vocabulary": {
            "item_ids_sha256": data_features.sha256_array(item_ids),
            "item_count": int(len(item_ids) + 1),
        },
        "query_state": {
            "source_ids_sha256": data_features.sha256_array(source_ids),
            "last_items_sha256": data_features.sha256_array(last_items),
            "source_count": int(len(source_ids)),
        },
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "runtime": {
            "jittor": str(implicit_mf_jittor.jt.__version__),
            "has_cuda": bool(implicit_mf_jittor.jt.has_cuda),
            "use_cuda": bool(implicit_mf_jittor.jt.flags.use_cuda),
        },
        "source_hashes": {path.name: _sha256(path) for path in source_files},
        "uses_test_labels": False,
    }
    _atomic_json(run_dir / "deploy_report.json", report)
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
                    "kind": "d4_full_history_transition_mf_deploy_v1",
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
