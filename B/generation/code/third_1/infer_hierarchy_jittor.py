#!/usr/bin/env python3
"""Stream Dataset4 inference for the multi-resolution hierarchy ranker."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from itertools import zip_longest
from pathlib import Path

import numpy as np

import data_features
from build_hierarchy_features import (
    FULL_LEVELS,
    RECENT_LEVELS,
    RECENT_WINDOWS,
    build_maps,
    feature_chunk,
)
from deployment_utils import (
    PairSeenIndex,
    duplicate_mask,
    iter_probability_chunks,
    qnorm,
    remap_probability_slots,
    sha256_file,
    strict_rank_scores,
)


EXPECTED_DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
EXPECTED_BASE_SHA256 = "face96780e4cc04d982aa174954aae3479f5d0dd806816390880cb441d56319f"
FEATURE_COUNT = 34


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _map_metadata(
    *, cutoff: int, stop: int, maximum_candidate: int, train_rows: int
) -> dict:
    return {
        "kind": "dataset4_hierarchy_deployment_maps_v1",
        "cutoff": int(cutoff),
        "history_stop": int(stop),
        "train_rows": int(train_rows),
        "maximum_candidate": int(maximum_candidate),
        "full_levels": [list(value) for value in FULL_LEVELS],
        "recent_levels": [list(value) for value in RECENT_LEVELS],
        "recent_windows": list(RECENT_WINDOWS),
        "feature_count": FEATURE_COUNT,
    }


def _save_map(destination: Path, maps: dict[str, np.ndarray]) -> None:
    destination.mkdir()
    for name in ("pair_keys", "pair_counts", "src_keys", "src_counts", "dst_keys", "dst_counts"):
        np.save(destination / f"{name}.npy", maps[name], allow_pickle=False)
    (destination / "rows.txt").write_text(f"{int(maps['rows'])}\n", encoding="ascii")


def _load_map(source: Path) -> dict[str, np.ndarray]:
    result = {
        name: np.load(source / f"{name}.npy", mmap_mode="r")
        for name in ("pair_keys", "pair_counts", "src_keys", "src_counts", "dst_keys", "dst_counts")
    }
    result["rows"] = int((source / "rows.txt").read_text(encoding="ascii"))
    return result


def prepare_map_cache(
    cache: Path,
    source: np.ndarray,
    destination: np.ndarray,
    timestamp: np.ndarray,
    *,
    cutoff: int,
    maximum_candidate: int,
) -> tuple[list[tuple[int, int, dict[str, np.ndarray]]], PairSeenIndex]:
    """Build deployment maps once, then reopen them as bounded memmaps."""
    stop = int(np.searchsorted(timestamp, cutoff, side="left"))
    expected = _map_metadata(
        cutoff=cutoff,
        stop=stop,
        maximum_candidate=maximum_candidate,
        train_rows=len(source),
    )
    metadata_path = cache / "metadata.json"
    if metadata_path.exists():
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"map cache metadata differs: {cache}")
    elif cache.exists():
        raise ValueError(f"incomplete map cache exists: {cache}")
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{cache.name}.", dir=cache.parent))
        try:
            index = 0
            for src_shift, dst_shift in FULL_LEVELS:
                base = (maximum_candidate >> dst_shift) + 1
                maps = build_maps(source, destination, 0, stop, src_shift, dst_shift, base)
                _save_map(temporary / f"feature_{index:02d}", maps)
                print(json.dumps({"map": index, "kind": "full", "src_shift": src_shift,
                                  "dst_shift": dst_shift, "unique_pairs": len(maps["pair_keys"])}), flush=True)
                index += 1
            for window in RECENT_WINDOWS:
                start = int(np.searchsorted(timestamp, cutoff - window, side="left"))
                for src_shift, dst_shift in RECENT_LEVELS:
                    base = (maximum_candidate >> dst_shift) + 1
                    maps = build_maps(source, destination, start, stop, src_shift, dst_shift, base)
                    _save_map(temporary / f"feature_{index:02d}", maps)
                    print(json.dumps({"map": index, "kind": "recent", "window": window,
                                      "src_shift": src_shift, "dst_shift": dst_shift,
                                      "unique_pairs": len(maps["pair_keys"])}), flush=True)
                    index += 1
            seen_base = maximum_candidate + 1
            seen = PairSeenIndex(source[:stop], destination[:stop], seen_base)
            np.save(temporary / "seen_keys.npy", seen.keys, allow_pickle=False)
            _atomic_json(temporary / "metadata.json", expected)
            os.replace(temporary, cache)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    maps = []
    index = 0
    for src_shift, dst_shift in FULL_LEVELS:
        maps.append((src_shift, dst_shift, _load_map(cache / f"feature_{index:02d}")))
        index += 1
    for _window in RECENT_WINDOWS:
        for src_shift, dst_shift in RECENT_LEVELS:
            maps.append((src_shift, dst_shift, _load_map(cache / f"feature_{index:02d}")))
            index += 1
    seen = PairSeenIndex.__new__(PairSeenIndex)
    seen.base = maximum_candidate + 1
    seen.keys = np.load(cache / "seen_keys.npy", mmap_mode="r")
    return maps, seen


def hierarchy_features(
    maps: list[tuple[int, int, dict[str, np.ndarray]]],
    source: np.ndarray,
    candidates: np.ndarray,
    maximum_candidate: int,
) -> np.ndarray:
    output = np.empty((len(source), candidates.shape[1], FEATURE_COUNT), dtype=np.float32)
    channel = 0
    for src_shift, dst_shift, values in maps:
        base = (maximum_candidate >> dst_shift) + 1
        log_pair, pmi = feature_chunk(
            values, source, candidates, src_shift, dst_shift, base
        )
        output[:, :, channel] = log_pair
        output[:, :, channel + 1] = pmi
        channel += 2
    if channel != FEATURE_COUNT or not np.isfinite(output).all():
        raise ValueError("deployment feature tensor is malformed")
    return output


def create_model(feature_count: int, hidden: int):
    """Create the exact topology used by train_hierarchy_jittor.py."""
    import jittor as jt
    from jittor import nn

    class IntensityNet(nn.Module):
        def __init__(self) -> None:
            self.full = nn.Sequential(
                nn.Linear(22, hidden), nn.Relu(), nn.Linear(hidden, hidden), nn.Relu()
            )
            self.recent = nn.Sequential(
                nn.Linear(feature_count - 22, hidden), nn.Relu(),
                nn.Linear(hidden, hidden), nn.Relu(),
            )
            self.local = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Relu())
            self.output = nn.Sequential(
                nn.Linear(3 * hidden, hidden), nn.Relu(), nn.Linear(hidden, 1)
            )

        def execute(self, values):
            full = self.full(values[:, :, :22])
            recent = self.recent(values[:, :, 22:])
            local = self.local(jt.concat((full, recent), dim=2))
            context = local.mean(dim=1, keepdims=True)
            context = context.broadcast((local.shape[0], local.shape[1], local.shape[2]))
            return self.output(jt.concat((full, recent, context), dim=2)).squeeze(-1)

    return IntensityNet()


def load_model(path: Path):
    import jittor as jt

    payload = np.load(path, allow_pickle=False)
    names = [str(value) for value in payload["names"].tolist()]
    state = {name: payload[f"state_{index}"] for index, name in enumerate(names)}
    if "full.0.weight" not in state or "recent.0.weight" not in state:
        raise ValueError(f"checkpoint topology is not recognized: {path}")
    hidden, full_features = state["full.0.weight"].shape
    recent_features = state["recent.0.weight"].shape[1]
    feature_count = int(full_features + recent_features)
    model = create_model(feature_count, int(hidden))
    expected_names = list(model.state_dict())
    if names != expected_names:
        raise ValueError(f"checkpoint parameter inventory differs: {path}")
    model.load_state_dict({name: jt.array(value) for name, value in state.items()})
    model.eval()
    return model, feature_count, int(hidden)


def predict_residual(
    models: list,
    features: np.ndarray,
    *,
    batch_rows: int,
    no_recent_weight: float,
) -> tuple[np.ndarray, float]:
    """Average full, candidate-reversal, and recent-drop TTA views."""
    import jittor as jt

    output = np.empty(features.shape[:2], dtype=np.float32)
    maximum_permutation_error = 0.0
    with jt.no_grad():
        for start in range(0, len(features), batch_rows):
            stop = min(len(features), start + batch_rows)
            block = np.asarray(features[start:stop])
            member_values = []
            for model in models:
                full = qnorm(np.asarray(model(jt.array(block)).data, dtype=np.float32))
                reverse = qnorm(
                    np.asarray(model(jt.array(block[:, ::-1].copy())).data, dtype=np.float32)
                )[:, ::-1]
                maximum_permutation_error = max(
                    maximum_permutation_error, float(np.max(np.abs(full - reverse)))
                )
                no_recent = block.copy()
                no_recent[:, :, 22:] = 0.0
                no_recent = qnorm(
                    np.asarray(model(jt.array(no_recent)).data, dtype=np.float32)
                )
                full_weight = 1.0 - no_recent_weight
                member_values.append(
                    qnorm(0.5 * full_weight * (full + reverse) + no_recent_weight * no_recent)
                )
            output[start:stop] = qnorm(np.mean(member_values, axis=0))
    return output, maximum_permutation_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--model", type=Path, nargs="+", required=True)
    parser.add_argument("--map-cache", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--scene", choices=("dataset4",), default="dataset4")
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--no-recent-weight", type=float, default=0.20)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--predict-batch", type=int, default=512)
    parser.add_argument("--use-cuda", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--skip-input-hash-check", action="store_true")
    args = parser.parse_args()
    if args.output_csv.exists() or args.report_output.exists():
        raise FileExistsError("refusing to overwrite inference output")
    if not 0.0 <= args.no_recent_weight <= 1.0 or args.scale < 0.0:
        raise ValueError("invalid TTA weight or residual scale")
    if not args.skip_input_hash_check:
        if sha256_file(args.data) != EXPECTED_DATA_SHA256:
            raise ValueError("official data SHA256 differs")
        if sha256_file(args.base) != EXPECTED_BASE_SHA256:
            raise ValueError("baseline submission SHA256 differs")

    first_chunk = next(data_features.iter_test_chunks(args.data, args.scene, chunk_rows=1))
    cutoff = int(first_chunk.time[0])
    source = np.load(args.train_cache / "src.npy", mmap_mode="r")
    destination = np.load(args.train_cache / "dst.npy", mmap_mode="r")
    timestamp = np.load(args.train_cache / "time.npy", mmap_mode="r")
    maximum_candidate = int(destination.max())
    if int(timestamp.max()) >= cutoff:
        raise ValueError("training history is not strictly before test queries")
    maps, seen_index = prepare_map_cache(
        args.map_cache,
        source,
        destination,
        timestamp,
        cutoff=cutoff,
        maximum_candidate=maximum_candidate,
    )

    import jittor as jt

    jt.flags.use_cuda = args.use_cuda
    models = []
    model_metadata = []
    for path in args.model:
        model, feature_count, hidden = load_model(path)
        if feature_count != FEATURE_COUNT:
            raise ValueError(f"model expects {feature_count} features, not {FEATURE_COUNT}")
        models.append(model)
        model_metadata.append({"path": str(path), "sha256": sha256_file(path), "hidden": hidden})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output_csv.name}.", suffix=".tmp", dir=args.output_csv.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    rows = 0
    duplicate_rows = 0
    seen_cells = 0
    changed_top1 = 0
    changed_order = 0
    row_sum_max_error = 0.0
    slot_max_error = 0.0
    permutation_error = 0.0
    minimum_probability = 1.0
    maximum_probability = 0.0
    started = time.time()
    test_iterator = data_features.iter_test_chunks(
        args.data, args.scene, chunk_rows=args.chunk_rows
    )
    base_iterator = iter_probability_chunks(
        args.base, f"{args.scene}.csv", chunk_rows=args.chunk_rows
    )
    try:
        with temporary.open("w", encoding="ascii", newline="\n", buffering=8 << 20) as output:
            for test_chunk, base_probability in zip_longest(test_iterator, base_iterator):
                if test_chunk is None or base_probability is None:
                    raise ValueError("test and baseline row counts differ")
                if len(test_chunk.src) != len(base_probability):
                    raise ValueError("test and baseline chunk boundaries differ")
                if args.max_rows is not None:
                    remaining = args.max_rows - rows
                    if remaining <= 0:
                        break
                    if remaining < len(base_probability):
                        base_probability = base_probability[:remaining]
                        source_chunk = test_chunk.src[:remaining]
                        candidate_chunk = test_chunk.candidates[:remaining]
                    else:
                        source_chunk = test_chunk.src
                        candidate_chunk = test_chunk.candidates
                else:
                    source_chunk = test_chunk.src
                    candidate_chunk = test_chunk.candidates
                if int(candidate_chunk.max()) > maximum_candidate:
                    raise ValueError("test candidate ID exceeds deployment key base")
                features = hierarchy_features(
                    maps, source_chunk, candidate_chunk, maximum_candidate
                )
                residual, error = predict_residual(
                    models,
                    features,
                    batch_rows=args.predict_batch,
                    no_recent_weight=args.no_recent_weight,
                )
                permutation_error = max(permutation_error, error)
                seen = seen_index.contains(source_chunk, candidate_chunk)
                duplicate = duplicate_mask(candidate_chunk)
                base_rank = strict_rank_scores(base_probability)
                correction = residual.copy()
                correction[~seen] = 0.0
                candidate_score = base_rank + np.float32(args.scale) * correction
                probability = remap_probability_slots(
                    base_probability, candidate_score, fallback=duplicate
                )
                sorted_base = np.sort(base_probability, axis=1, kind="stable")
                sorted_output = np.sort(probability, axis=1, kind="stable")
                slot_max_error = max(
                    slot_max_error, float(np.max(np.abs(sorted_base - sorted_output)))
                )
                row_sum_max_error = max(
                    row_sum_max_error,
                    float(np.max(np.abs(probability.sum(axis=1) - base_probability.sum(axis=1)))),
                )
                duplicate_rows += int(duplicate.sum())
                seen_cells += int(seen.sum())
                changed_top1 += int(
                    np.sum(np.argmax(base_probability, axis=1) != np.argmax(probability, axis=1))
                )
                changed_order += int(
                    np.sum(np.any(np.argsort(base_probability, axis=1, kind="stable") !=
                                      np.argsort(probability, axis=1, kind="stable"), axis=1))
                )
                minimum_probability = min(minimum_probability, float(probability.min()))
                maximum_probability = max(maximum_probability, float(probability.max()))
                np.savetxt(output, probability, fmt="%.8f", delimiter=",")
                rows += len(probability)
                print(json.dumps({"rows": rows, "elapsed_seconds": time.time() - started,
                                  "top1_changed": changed_top1}), flush=True)
        os.replace(temporary, args.output_csv)
    finally:
        temporary.unlink(missing_ok=True)

    smoke_only = args.max_rows is not None
    checks = {
        "rows_nonzero": rows > 0,
        "full_official_rows": smoke_only or rows == 2_322_538,
        "probability_range": minimum_probability >= 0.0 and maximum_probability <= 1.0,
        "probability_slots_exact": slot_max_error == 0.0,
        "row_sums_preserved": row_sum_max_error <= 1e-12,
        "candidate_permutation_equivariant": permutation_error <= 1e-5,
    }
    report = {
        "kind": "dataset4_hierarchy_residual_stream_inference_v1",
        "decision": "SMOKE_ONLY" if smoke_only else ("PASS" if all(checks.values()) else "NO_GO"),
        "checks": checks,
        "use_cuda": bool(args.use_cuda),
        "rows": rows,
        "scale": args.scale,
        "policy": "pair_seen_only_with_duplicate_row_fallback",
        "tta": {
            "views": ["full", "candidate_reverse", "no_recent"],
            "no_recent_weight": args.no_recent_weight,
            "candidate_permutation_max_error": permutation_error,
        },
        "models": model_metadata,
        "data_sha256": sha256_file(args.data),
        "base_sha256": sha256_file(args.base),
        "output_csv": str(args.output_csv),
        "output_csv_sha256": sha256_file(args.output_csv),
        "duplicate_rows": duplicate_rows,
        "seen_cells": seen_cells,
        "seen_cell_rate": seen_cells / max(1, rows * 100),
        "top1_changed_rows": changed_top1,
        "top1_changed_rate": changed_top1 / max(1, rows),
        "order_changed_rows": changed_order,
        "order_changed_rate": changed_order / max(1, rows),
        "probability_minimum": minimum_probability,
        "probability_maximum": maximum_probability,
        "probability_slot_max_error": slot_max_error,
        "row_sum_preservation_max_error": row_sum_max_error,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(args.report_output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["decision"] in ("PASS", "SMOKE_ONLY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
