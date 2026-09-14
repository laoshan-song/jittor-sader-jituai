#!/usr/bin/env python3
"""Materialize causal multi-resolution ID-bucket intensity features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


SPLITS = ("validation", "confirmation")
STRATEGIES = ("history", "test_pool")
FULL_LEVELS = ((2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (14, 14),
               (4, 6), (6, 4), (6, 8), (8, 6))
RECENT_LEVELS = ((4, 4), (6, 6), (8, 8))
RECENT_WINDOWS = (1800, 21600)


def row_zscore(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    return (centered / np.maximum(scale, 1e-6)).astype(np.float32)


def count_map(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys, counts = np.unique(values, return_counts=True)
    return keys, counts.astype(np.float32)


def lookup(keys: np.ndarray, counts: np.ndarray, query: np.ndarray) -> np.ndarray:
    flat = query.reshape(-1)
    positions = np.searchsorted(keys, flat)
    valid = positions < len(keys)
    result = np.zeros(len(flat), dtype=np.float32)
    selected = np.flatnonzero(valid)
    exact = keys[positions[selected]] == flat[selected]
    selected = selected[exact]
    result[selected] = counts[positions[selected]]
    return result.reshape(query.shape)


def build_maps(
    source: np.ndarray,
    destination: np.ndarray,
    start: int,
    stop: int,
    src_shift: int,
    dst_shift: int,
    base: int,
) -> dict[str, np.ndarray]:
    src_bucket = np.right_shift(source[start:stop], src_shift).astype(np.int64)
    dst_bucket = np.right_shift(destination[start:stop], dst_shift).astype(np.int64)
    pair_keys, pair_counts = count_map(src_bucket * base + dst_bucket)
    src_keys, src_counts = count_map(src_bucket)
    dst_keys, dst_counts = count_map(dst_bucket)
    return {
        "pair_keys": pair_keys,
        "pair_counts": pair_counts,
        "src_keys": src_keys,
        "src_counts": src_counts,
        "dst_keys": dst_keys,
        "dst_counts": dst_counts,
        "rows": stop - start,
    }


def feature_chunk(
    maps: dict[str, np.ndarray],
    query_source: np.ndarray,
    candidates: np.ndarray,
    src_shift: int,
    dst_shift: int,
    base: int,
) -> tuple[np.ndarray, np.ndarray]:
    src_bucket = np.right_shift(query_source, src_shift).astype(np.int64)
    dst_bucket = np.right_shift(candidates, dst_shift).astype(np.int64)
    pair_key = src_bucket[:, None] * base + dst_bucket
    pair = lookup(maps["pair_keys"], maps["pair_counts"], pair_key)
    source = lookup(maps["src_keys"], maps["src_counts"], src_bucket[:, None])
    destination = lookup(maps["dst_keys"], maps["dst_counts"], dst_bucket)
    log_pair = np.log1p(pair)
    pmi = (
        np.log(pair + 0.25)
        + np.log(float(maps["rows"]) + 1.0)
        - np.log(source + 1.0)
        - np.log(destination + 1.0)
    )
    return row_zscore(log_pair), row_zscore(pmi)


def paths(identity: Path, strategy: str, split: str) -> tuple[Path, Path]:
    prefix = f"{strategy}__{split}"
    return identity / f"{prefix}__src.npy", identity / f"{prefix}__candidates.npy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--identity-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source = np.load(args.train_cache / "src.npy", mmap_mode="r")
    destination = np.load(args.train_cache / "dst.npy", mmap_mode="r")
    timestamp = np.load(args.train_cache / "time.npy", mmap_mode="r")
    group_arrays = {}
    maximum_candidate = int(destination.max())
    for strategy in STRATEGIES:
        for split in SPLITS:
            src_path, candidate_path = paths(args.identity_cache, strategy, split)
            query_source = np.load(src_path, mmap_mode="r")
            candidates = np.load(candidate_path, mmap_mode="r")
            time = np.load(args.identity_cache / f"{strategy}__{split}__time.npy", mmap_mode="r")
            maximum_candidate = max(maximum_candidate, int(candidates.max()))
            group_arrays[(strategy, split)] = (query_source, candidates, time)
    feature_names = []
    for src_shift, dst_shift in FULL_LEVELS:
        feature_names.extend((f"full_s{src_shift}_d{dst_shift}_logpair", f"full_s{src_shift}_d{dst_shift}_pmi"))
    for window in RECENT_WINDOWS:
        for src_shift, dst_shift in RECENT_LEVELS:
            feature_names.extend((f"recent{window}_s{src_shift}_d{dst_shift}_logpair", f"recent{window}_s{src_shift}_d{dst_shift}_pmi"))
    outputs = {}
    for key, (query_source, candidates, _) in group_arrays.items():
        strategy, split = key
        outputs[key] = np.lib.format.open_memmap(
            args.output / f"{strategy}__{split}.npy",
            mode="w+", dtype=np.float32,
            shape=(len(query_source), candidates.shape[1], len(feature_names)),
        )
    feature_index = 0
    audit = []
    try:
        for split in SPLITS:
            cutoff = min(int(group_arrays[(strategy, split)][2].min()) for strategy in STRATEGIES)
            stop = int(np.searchsorted(timestamp, cutoff, side="left"))
            for src_shift, dst_shift in FULL_LEVELS:
                base = (maximum_candidate >> dst_shift) + 1
                maps = build_maps(source, destination, 0, stop, src_shift, dst_shift, base)
                for strategy in STRATEGIES:
                    query_source, candidates, _ = group_arrays[(strategy, split)]
                    for begin in range(0, len(query_source), args.chunk_rows):
                        end = min(len(query_source), begin + args.chunk_rows)
                        log_pair, pmi = feature_chunk(
                            maps, query_source[begin:end], candidates[begin:end], src_shift, dst_shift, base
                        )
                        outputs[(strategy, split)][begin:end, :, feature_index] = log_pair
                        outputs[(strategy, split)][begin:end, :, feature_index + 1] = pmi
                audit.append({"split": split, "cutoff": cutoff, "history_rows": stop,
                              "kind": "full", "src_shift": src_shift, "dst_shift": dst_shift,
                              "unique_pairs": len(maps["pair_keys"])})
                print(json.dumps(audit[-1]), flush=True)
                feature_index += 2
            # Full features occupy identical channel indices across splits.
            feature_index -= 2 * len(FULL_LEVELS)
        feature_index = 2 * len(FULL_LEVELS)
        for split in SPLITS:
            cutoff = min(int(group_arrays[(strategy, split)][2].min()) for strategy in STRATEGIES)
            stop = int(np.searchsorted(timestamp, cutoff, side="left"))
            for window in RECENT_WINDOWS:
                start = int(np.searchsorted(timestamp, cutoff - window, side="left"))
                for src_shift, dst_shift in RECENT_LEVELS:
                    base = (maximum_candidate >> dst_shift) + 1
                    maps = build_maps(source, destination, start, stop, src_shift, dst_shift, base)
                    for strategy in STRATEGIES:
                        query_source, candidates, _ = group_arrays[(strategy, split)]
                        for begin in range(0, len(query_source), args.chunk_rows):
                            end = min(len(query_source), begin + args.chunk_rows)
                            log_pair, pmi = feature_chunk(
                                maps, query_source[begin:end], candidates[begin:end], src_shift, dst_shift, base
                            )
                            outputs[(strategy, split)][begin:end, :, feature_index] = log_pair
                            outputs[(strategy, split)][begin:end, :, feature_index + 1] = pmi
                    audit.append({"split": split, "cutoff": cutoff, "history_rows": stop,
                                  "kind": "recent", "window": window, "window_rows": stop - start,
                                  "src_shift": src_shift, "dst_shift": dst_shift,
                                  "unique_pairs": len(maps["pair_keys"])})
                    print(json.dumps(audit[-1]), flush=True)
                    feature_index += 2
            feature_index -= 2 * len(RECENT_LEVELS) * len(RECENT_WINDOWS)
        report = {
            "kind": "causal_multiresolution_id_intensity_features_v1",
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "full_levels": FULL_LEVELS,
            "recent_levels": RECENT_LEVELS,
            "recent_windows": RECENT_WINDOWS,
            "audit": audit,
            "candidate_permutation_equivariant": True,
        }
        temporary = args.output / ".metadata.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output / "metadata.json")
    finally:
        for output in outputs.values():
            output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
