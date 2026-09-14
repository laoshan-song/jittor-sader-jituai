#!/usr/bin/env python3
"""Audit strict-past item-transition counts as a D4 candidate feature."""

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

from . import data_features, verify_run


SIZES = {"train": 1, "valid": 30_000, "confirm": 30_000}


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


class TransitionIndex:
    def __init__(
        self,
        source_ids: np.ndarray,
        last_items: np.ndarray,
        keys: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        self.source_ids = source_ids
        self.last_items = last_items
        self.keys = keys
        self.counts = counts

    @classmethod
    def build(cls, cache: data_features.BDataCache, cutoff: int) -> "TransitionIndex":
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
        transitions = same_source & strict_time
        raw_keys = (
            ordered_item[:-1][transitions].astype(np.uint64) << np.uint64(32)
        ) | ordered_item[1:][transitions].astype(np.uint64)
        keys, counts = np.unique(raw_keys, return_counts=True)
        ends = np.r_[np.flatnonzero(ordered_source[1:] != ordered_source[:-1]), len(order) - 1]
        return cls(
            ordered_source[ends],
            ordered_item[ends],
            keys,
            counts.astype(np.uint32),
        )

    def score(self, source: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        source = np.asarray(source, dtype=np.uint32)
        candidates = np.asarray(candidates, dtype=np.uint32)
        positions = np.searchsorted(self.source_ids, source)
        known = positions < len(self.source_ids)
        matched = np.zeros(len(source), dtype=bool)
        matched[known] = self.source_ids[positions[known]] == source[known]
        last = np.zeros(len(source), dtype=np.uint32)
        last[matched] = self.last_items[positions[matched]]
        query = (last[:, None].astype(np.uint64) << np.uint64(32)) | candidates.astype(
            np.uint64
        )
        flat = query.reshape(-1)
        key_positions = np.searchsorted(self.keys, flat)
        inside = key_positions < len(self.keys)
        found = np.zeros(len(flat), dtype=bool)
        found[inside] = self.keys[key_positions[inside]] == flat[inside]
        score = np.zeros(len(flat), dtype=np.float32)
        score[found] = np.log1p(self.counts[key_positions[found]]).astype(np.float32)
        return score.reshape(candidates.shape)

    def metadata(self) -> dict[str, Any]:
        return {
            "source_count": int(len(self.source_ids)),
            "transition_count": int(len(self.keys)),
            "source_ids_sha256": data_features.sha256_array(self.source_ids),
            "last_items_sha256": data_features.sha256_array(self.last_items),
            "keys_sha256": data_features.sha256_array(self.keys),
            "counts_sha256": data_features.sha256_array(self.counts),
        }


def _evaluate(
    cache: data_features.BDataCache,
    group: data_features.CandidateGroup,
    index: TransitionIndex,
    batch_rows: int,
) -> dict[str, Any]:
    store = cache.feature_store(group.cutoff)
    metrics = data_features.RankingMetrics()
    for batch in group.iter_batches(batch_rows=batch_rows):
        score = index.score(batch.src, batch.candidates)
        segments = store.evaluation_segments(
            batch.src, batch.time, batch.candidates, batch.labels
        )
        metrics.update(score, batch.labels, segments=segments)
    return metrics.result()


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = args.data.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing run directory reuse: {run_dir}")
    if data_features.sha256_file(data) != verify_run.EXPECTED_DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    run_dir.mkdir(parents=True)
    cache = data_features.BDataCache.build_or_open(
        data,
        "dataset4",
        args.cache_dir.resolve(),
        chunk_rows=args.cache_chunk_rows,
        verify_hash=True,
    )
    groups = {
        strategy: data_features.build_split1_groups(
            cache,
            seed=20260810,
            sizes=SIZES,
            batch_rows=4096,
            negative_strategy=strategy,
        )
        for strategy in ("history", "test_pool")
    }
    plan = groups["history"].plan
    indexes = {
        "validation": TransitionIndex.build(cache, int(plan.cutoffs["valid"])),
        "confirmation": TransitionIndex.build(cache, int(plan.cutoffs["confirm"])),
    }
    metrics = {
        strategy: {
            "validation": _evaluate(
                cache, replay.valid, indexes["validation"], args.batch_rows
            ),
            "confirmation": _evaluate(
                cache, replay.confirm, indexes["confirmation"], args.batch_rows
            ),
        }
        for strategy, replay in groups.items()
    }
    report = {
        "kind": "d4_transition_count_research_v1",
        "decision": "RESEARCH_ONLY",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_sha256": verify_run.EXPECTED_DATA_SHA256,
        "uses_test_labels": False,
        "strict_time_rule": "previous edge time < next edge time < query cutoff",
        "selection_rule": "history validation only; confirmation and test-pool excluded",
        "indexes": {name: index.metadata() for name, index in indexes.items()},
        "metrics": metrics,
        "group_metadata": {strategy: replay.metadata for strategy, replay in groups.items()},
        "source_hashes": {
            Path(__file__).name: _sha256(Path(__file__).resolve()),
            Path(data_features.__file__).name: _sha256(Path(data_features.__file__).resolve()),
        },
    }
    _atomic_json(run_dir / "research_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--cache-chunk-rows", type=int, default=250000)
    return parser


def main() -> int:
    try:
        print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "kind": "d4_transition_count_research_v1",
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
