#!/usr/bin/env python3
"""Train the validated temporal rankers and stream an official B-rank ZIP.

The deployment protocol is intentionally fixed to the time-isolated screening
configuration.  Dataset3 uses a static candidate-set residual; Dataset4 uses
the same residual only for candidates whose source-destination pair was seen
before the frozen feature cutoff.  All learned operations are implemented by
Jittor through :mod:`temporal_attention_jittor`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCENES = ("dataset3", "dataset4")
EXPECTED_ARCHIVE_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
GROUP_SEED = 20260810
SELECTED_MODEL_SEED = 20260812
TRAIN_GROUPS = 100000
VALID_GROUPS = 30000
CONFIRM_GROUPS = 30000
HISTORY_SIZE = 32
EMBEDDING_DIM = 32
EPOCHS = 5
TRAIN_BATCH_ROWS = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6

# These records are evidence only.  Selection used validation MRR, never the
# confirmation value, and the same hyperparameters are used below.
SELECTION_EVIDENCE = {
    "dataset3": {
        "seed": SELECTED_MODEL_SEED,
        "validation_mrr": 0.378950040920,
        "confirmation_mrr": 0.338525474566,
        "static_context": True,
        "static_context_pair_seen_only": False,
    },
    "dataset4": {
        "seed": SELECTED_MODEL_SEED,
        "validation_mrr": 0.305231786365,
        "confirmation_mrr": 0.295951869203,
        "static_context": True,
        "static_context_pair_seen_only": True,
    },
}


class InferenceProtocolError(RuntimeError):
    """Raised when a deployment input breaks the audited temporal contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InferenceProtocolError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_new(temporary: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing output reuse: {destination}")
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError(f"refusing output reuse: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _apis() -> tuple[Any, Any, Any, Any, Any]:
    """Import Jittor lazily so ``--help`` has no CUDA side effects."""
    try:
        from . import (
            data_features,
            temporal_attention_jittor,
            temporal_history,
            temporal_validate,
            verify_run,
        )
    except ImportError:
        import data_features
        import temporal_attention_jittor
        import temporal_history
        import temporal_validate
        import verify_run
    return (
        data_features,
        temporal_attention_jittor,
        temporal_history,
        temporal_validate,
        verify_run,
    )


def _source_hashes(data_features: Any, modules: tuple[Any, ...]) -> dict[str, str]:
    code_root = Path(__file__).resolve().parents[1]
    files = (Path(__file__).resolve(),) + tuple(Path(module.__file__).resolve() for module in modules)
    return {
        path.relative_to(code_root).as_posix(): data_features.sha256_file(path)
        for path in files
    }


def _id_mode(scene: str) -> str:
    if scene == "dataset3":
        return "shared"
    if scene == "dataset4":
        return "bipartite"
    raise InferenceProtocolError(f"unknown scene: {scene}")


def _test_history_cutoff(
    data_features: Any,
    data: Path,
    cache: Any,
    *,
    chunk_rows: int,
) -> int:
    """Return the first test timestamp from the explicitly supplied archive.

    ``BDataCache`` intentionally remembers its creation archive for cache
    diagnostics.  Deployment must nevertheless stream test candidates from
    the archive supplied to this invocation, rather than that cached path.
    """
    minimum: int | None = None
    rows = 0
    for chunk in data_features.iter_test_chunks(data, cache.scene, chunk_rows=chunk_rows):
        if len(chunk.time):
            value = int(chunk.time.min())
            minimum = value if minimum is None else min(minimum, value)
            rows += len(chunk.time)
    _require(minimum is not None and rows > 0, f"{cache.scene} official test CSV is empty")
    time_range = cache.metadata.get("time_range")
    _require(isinstance(time_range, list) and len(time_range) == 2, "cache time range is invalid")
    train_maximum = int(time_range[1])
    _require(
        minimum > train_maximum,
        f"{cache.scene} test starts at {minimum}, not strictly after training history {train_maximum}",
    )
    return minimum


def _probabilities(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 100, "ranker scores must be (rows, 100)")
    _require(np.isfinite(values).all(), "ranker produced NaN or infinity")
    values = values - values.max(axis=1, keepdims=True)
    np.exp(values, out=values)
    normalizer = values.sum(axis=1, keepdims=True)
    _require(np.isfinite(normalizer).all() and np.all(normalizer > 0.0), "cannot normalize scores")
    values /= normalizer
    return values


def _deployment_flags(scene: str) -> tuple[bool, bool]:
    evidence = SELECTION_EVIDENCE[scene]
    return bool(evidence["static_context"]), bool(evidence["static_context_pair_seen_only"])


def _train_scene(
    *,
    scene: str,
    args: argparse.Namespace,
    data_features: Any,
    temporal_attention_jittor: Any,
    temporal_history: Any,
    temporal_validate: Any,
) -> dict[str, Any]:
    cache = data_features.BDataCache.build_or_open(
        args.data,
        scene,
        args.cache_dir,
        chunk_rows=int(args.cache_chunk_rows),
        verify_hash=True,
    )
    groups = data_features.build_split1_groups(
        cache,
        seed=GROUP_SEED,
        sizes={"train": TRAIN_GROUPS, "valid": VALID_GROUPS, "confirm": CONFIRM_GROUPS},
        batch_rows=int(args.group_batch_rows),
        negative_strategy="history",
    )
    test_cutoff = _test_history_cutoff(
        data_features,
        args.data,
        cache,
        chunk_rows=int(args.test_chunk_rows),
    )
    # Match the screened training convention: the train vocabulary freezes at
    # the replay validation boundary, while the deployed history can consume
    # every official training edge before the test boundary.
    training_history_cutoff = int(groups.valid.cutoff)
    train_history = temporal_history.TemporalHistory.build(
        cache.src,
        cache.dst,
        cache.time,
        history_size=HISTORY_SIZE,
        id_mode=_id_mode(scene),
        cutoff=training_history_cutoff,
    )
    history = temporal_history.TemporalHistory.build(
        cache.src,
        cache.dst,
        cache.time,
        history_size=HISTORY_SIZE,
        id_mode=_id_mode(scene),
        cutoff=test_cutoff,
        vocabulary=train_history.vocabulary,
    )
    _require(
        history.history_rows == cache.history_end(test_cutoff),
        "deployment history does not contain exactly the allowed training rows",
    )
    _require(
        train_history.history_rows == cache.history_end(training_history_cutoff),
        "training history does not end at the screened validation boundary",
    )
    train_store = cache.feature_store(groups.train.cutoff)
    arrays = temporal_validate._write_training_arrays(
        groups.train,
        history,
        args.run_dir / "arrays" / scene,
        batch_rows=int(args.group_batch_rows),
        static_feature_store=train_store,
    )
    static_context, pair_seen_only = _deployment_flags(scene)
    model, losses = temporal_attention_jittor.train_ranker(
        arrays.source,
        arrays.candidates,
        arrays.history,
        arrays.history_gap,
        arrays.labels,
        source_count=train_history.source_vocab_size,
        item_count=train_history.item_vocab_size,
        features=arrays.features,
        embedding_dim=EMBEDDING_DIM,
        static_context=static_context,
        static_context_pair_seen_only=pair_seen_only,
        dropout=0.0,
        time_scale=0.25,
        epochs=EPOCHS,
        batch_size=TRAIN_BATCH_ROWS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        seed=SELECTED_MODEL_SEED,
        verbose=bool(args.verbose),
    )
    checkpoint = temporal_attention_jittor.save_checkpoint(
        args.run_dir / "checkpoints" / scene / f"seed{SELECTED_MODEL_SEED}.npz", model
    )
    restored, restored_config = temporal_attention_jittor.load_checkpoint(checkpoint)
    _require(
        restored_config == temporal_attention_jittor.model_config(model),
        "checkpoint configuration changed during save/load",
    )
    test_store = cache.feature_store(test_cutoff)
    return {
        "cache": cache,
        "history": history,
        "model": restored,
        "test_store": test_store,
        "checkpoint": checkpoint,
        "losses": losses,
        "groups": groups,
        "test_cutoff": test_cutoff,
        "training_history_cutoff": training_history_cutoff,
        "training_history_rows": train_history.history_rows,
    }


def _write_scene_predictions(
    archive: zipfile.ZipFile,
    *,
    scene: str,
    trained: Mapping[str, Any],
    args: argparse.Namespace,
    data_features: Any,
    temporal_attention_jittor: Any,
) -> int:
    rows = 0
    with archive.open(f"{scene}.csv", "w", force_zip64=True) as raw_member:
        with io.TextIOWrapper(raw_member, encoding="ascii", newline="\n") as member:
            for chunk in data_features.iter_test_chunks(
                args.data, scene, chunk_rows=int(args.test_chunk_rows)
            ):
                temporal = trained["history"].transform_chunk(chunk)
                features = trained["test_store"].features(chunk.src, chunk.time, chunk.candidates)
                scores = temporal_attention_jittor.predict_scores(
                    trained["model"],
                    temporal.source_indices,
                    temporal.candidate_indices,
                    temporal.history_item_indices,
                    temporal.log_time_deltas,
                    features=features,
                    batch_size=int(args.predict_batch_rows),
                )
                np.savetxt(member, _probabilities(scores), fmt="%.8f", delimiter=",", newline="\n")
                rows += len(chunk.src)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.data = args.data.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    args.output = args.output.resolve()
    _require(args.output.suffix == ".zip", "--output must end in .zip")
    _require(args.data.is_file(), f"official archive does not exist: {args.data}")
    _require(_sha256_file(args.data) == EXPECTED_ARCHIVE_SHA256, "official archive hash differs")
    _require(not args.run_dir.exists(), f"refusing to reuse run directory: {args.run_dir}")
    _require(not args.output.exists(), f"refusing output reuse: {args.output}")
    manifest_path = args.output.with_suffix(".manifest.json")
    _require(not manifest_path.exists(), f"refusing manifest reuse: {manifest_path}")
    _require(int(args.cache_chunk_rows) > 0, "cache-chunk-rows must be positive")
    _require(int(args.group_batch_rows) > 0, "group-batch-rows must be positive")
    _require(int(args.test_chunk_rows) > 0, "test-chunk-rows must be positive")
    _require(int(args.predict_batch_rows) > 0, "predict-batch-rows must be positive")

    data_features, temporal_attention_jittor, temporal_history, temporal_validate, verify_run = _apis()
    frozen_source_hashes = _source_hashes(
        data_features,
        (data_features, temporal_attention_jittor, temporal_history, temporal_validate, verify_run),
    )
    temporal_attention_jittor.configure_cuda()
    runtime = {
        "jittor": str(temporal_attention_jittor.jt.__version__),
        "has_cuda": bool(temporal_attention_jittor.jt.has_cuda),
        "use_cuda": bool(temporal_attention_jittor.jt.flags.use_cuda),
    }
    _require(runtime["has_cuda"] and runtime["use_cuda"], "Jittor CUDA is not enabled")
    args.run_dir.mkdir(parents=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    trained: dict[str, dict[str, Any]] = {}
    rows: dict[str, int] = {}
    try:
        with zipfile.ZipFile(
            temporary_output,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for scene in SCENES:
                trained[scene] = _train_scene(
                    scene=scene,
                    args=args,
                    data_features=data_features,
                    temporal_attention_jittor=temporal_attention_jittor,
                    temporal_history=temporal_history,
                    temporal_validate=temporal_validate,
                )
                rows[scene] = _write_scene_predictions(
                    archive,
                    scene=scene,
                    trained=trained[scene],
                    args=args,
                    data_features=data_features,
                    temporal_attention_jittor=temporal_attention_jittor,
                )
        submission_sha256 = _sha256_file(temporary_output)
        final_source_hashes = _source_hashes(
            data_features,
            (data_features, temporal_attention_jittor, temporal_history, temporal_validate, verify_run),
        )
        _require(
            final_source_hashes == frozen_source_hashes,
            "source files changed while inference was running",
        )
        checkpoints: dict[str, Any] = {}
        for scene, value in trained.items():
            checkpoint = value["checkpoint"]
            groups = value["groups"]
            checkpoints[scene] = {
                "path": str(checkpoint.relative_to(args.run_dir)),
                "sha256": data_features.sha256_file(checkpoint),
                "seed": SELECTED_MODEL_SEED,
                "model_config": temporal_attention_jittor.model_config(value["model"]),
                "train_loss": [float(loss) for loss in value["losses"]],
                "history_rows": int(value["history"].history_rows),
                "test_feature_cutoff": int(value["test_cutoff"]),
                "training_feature_cutoff": int(groups.train.cutoff),
                "training_history_cutoff": int(value["training_history_cutoff"]),
                "training_history_rows": int(value["training_history_rows"]),
                "training_group_metadata_sha256": data_features.sha256_file(groups.root / "metadata.json"),
            }
        manifest = {
            "kind": "b_rank_temporal_attention_inference_v1",
            "created_utc": _utc_now(),
            "data_sha256": EXPECTED_ARCHIVE_SHA256,
            "source_hashes": frozen_source_hashes,
            "submission_sha256": submission_sha256,
            "jittor_runtime": runtime,
            "selection": {
                "rule": "maximum validation MRR among the frozen three-seed screens; confirmation excluded",
                "evidence": SELECTION_EVIDENCE,
            },
            "training_protocol": {
                "group_seed": GROUP_SEED,
                "group_sizes": {"train": TRAIN_GROUPS, "valid": VALID_GROUPS, "confirm": CONFIRM_GROUPS},
                "negative_strategy": "history",
                "history_size": HISTORY_SIZE,
                "embedding_dim": EMBEDDING_DIM,
                "epochs": EPOCHS,
                "train_batch_rows": TRAIN_BATCH_ROWS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "static_features": True,
                "training_history_vocabulary": "frozen at the replay validation cutoff",
                "test_history_vocabulary": "reuses the frozen training vocabulary",
            },
            "checkpoints": checkpoints,
            "row_counts": rows,
            "format": "exactly dataset3.csv,dataset4.csv; headerless ASCII; %.8f probabilities",
        }
        _atomic_json(temporary_manifest, manifest)
        verification = verify_run.verify_run(
            args.data,
            temporary_output,
            temporary_manifest,
            source_root=Path(__file__).resolve().parents[1],
        )
        _publish_new(temporary_output, args.output)
        _publish_new(temporary_manifest, manifest_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise
    return {
        "kind": "b_rank_temporal_attention_inference_result_v1",
        "decision": "PASS",
        "output": str(args.output),
        "output_sha256": _sha256_file(args.output),
        "manifest": str(manifest_path),
        "rows": rows,
        "verification": verification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="official data_B.zip")
    parser.add_argument("--cache-dir", required=True, type=Path, help="content-addressed data cache")
    parser.add_argument("--run-dir", required=True, type=Path, help="new checkpoint and audit directory")
    parser.add_argument("--output", required=True, type=Path, help="new submission ZIP path")
    parser.add_argument("--cache-chunk-rows", type=int, default=250000)
    parser.add_argument("--group-batch-rows", type=int, default=4096)
    parser.add_argument("--test-chunk-rows", type=int, default=1024)
    parser.add_argument("--predict-batch-rows", type=int, default=512)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "kind": "b_rank_temporal_attention_inference_v1",
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
