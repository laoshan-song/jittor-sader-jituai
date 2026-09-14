#!/usr/bin/env python3
"""Train a causal full-history D4 MF checkpoint for official test inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import data_features, implicit_mf_jittor, temporal_history, temporal_infer, verify_run


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict:
    data = args.data.resolve()
    cache_dir = args.cache_dir.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing run directory reuse: {run_dir}")
    if data_features.sha256_file(data) != verify_run.EXPECTED_DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    run_dir.mkdir(parents=True)
    implicit_mf_jittor.configure_cuda()
    cache = data_features.BDataCache.build_or_open(
        data, "dataset4", cache_dir, chunk_rows=args.cache_chunk_rows, verify_hash=True
    )
    test_cutoff = temporal_infer._test_history_cutoff(
        data_features, data, cache, chunk_rows=args.test_chunk_rows
    )
    stop = cache.history_end(test_cutoff)
    vocabulary = temporal_history.TemporalVocabulary.from_training_edges(
        cache.src,
        cache.dst,
        cache.time,
        id_mode="bipartite",
        cutoff=test_cutoff,
    )
    model, losses = implicit_mf_jittor.train_full_history(
        cache.src[:stop],
        cache.dst[:stop],
        vocabulary.source_ids,
        vocabulary.item_ids,
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
        run_dir / "checkpoints" / f"seed{args.seed}.npz",
        model,
        vocabulary.source_ids,
        vocabulary.item_ids,
    )
    report = {
        "kind": "d4_full_history_implicit_mf_deploy_v1",
        "decision": "READY_FOR_CAUSAL_TEST_INFERENCE",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_sha256": verify_run.EXPECTED_DATA_SHA256,
        "test_cutoff": test_cutoff,
        "history_rows": stop,
        "vocabulary": {
            "source_count": len(vocabulary.source_ids) + 1,
            "item_count": len(vocabulary.item_ids) + 1,
            "source_ids_sha256": data_features.sha256_array(vocabulary.source_ids),
            "item_ids_sha256": data_features.sha256_array(vocabulary.item_ids),
        },
        "training": {
            "embedding_dim": args.embedding_dim,
            "negative_count": args.negative_count,
            "epochs": args.epochs,
            "batch_rows": args.batch_rows,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "loss": losses,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "runtime": {
            "jittor": str(implicit_mf_jittor.jt.__version__),
            "has_cuda": bool(implicit_mf_jittor.jt.has_cuda),
            "use_cuda": bool(implicit_mf_jittor.jt.flags.use_cuda),
        },
        "source_hashes": {
            path.name: _sha256(path)
            for path in (Path(__file__).resolve(), Path(implicit_mf_jittor.__file__).resolve())
        },
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
        print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"kind": "d4_full_history_implicit_mf_deploy_v1", "decision": "ERROR", "error": f"{type(error).__name__}: {error}"},
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
