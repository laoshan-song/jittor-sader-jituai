#!/usr/bin/env python3
"""Apply the gated 75-feature Jittor meta ranker on top of ruc4."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("use_cutt", "0")
os.environ.setdefault("use_cutlass", "0")
os.environ.setdefault("use_nccl", "0")
os.environ.setdefault("use_mkl", "0")

import numpy as np


WIDTH = 100
ROWS = {"dataset3.csv": 157_670, "dataset4.csv": 2_322_538}
EXPECTED_DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
EXPECTED_BASE_SHA256 = "face96780e4cc04d982aa174954aae3479f5d0dd806816390880cb441d56319f"
FULL_LEVELS = (
    (2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (14, 14),
    (4, 6), (6, 4), (6, 8), (8, 6),
)
RECENT_LEVELS = ((4, 4), (6, 6), (8, 8))
RECENT_WINDOWS = (1800, 21600)
NEIGHBOR_NAMES = (
    "max_cosine",
    "max_jaccard",
    "top3_cosine_sum",
    "max_rare_overlap",
    "log_neighbors_overlap2",
)
ANALOGY_NAMES = (
    "top3_cosine_sum",
    "top8_cosine_sum",
    "top3_tfidf_cosine_sum",
    "top3_prefix_cosine_sum",
    "top3_recency_1h_sum",
    "top3_recency_6h_sum",
    "top3_recency_24h_sum",
    "max_tfidf_cosine",
    "max_prefix_cosine",
    "log_neighbors_overlap2",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def qnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - values.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    return (centered / np.maximum(scale, np.float32(1e-6))).astype(np.float32)


def normalized_cube(values: np.ndarray) -> np.ndarray:
    output = np.empty(values.shape, dtype=np.float32)
    for index in range(values.shape[2]):
        output[:, :, index] = qnorm(values[:, :, index])
    return output


def strict_rank_scores(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    result = np.empty(values.shape, dtype=np.float32)
    grid = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float32)
    result[np.arange(len(values))[:, None], order] = grid[None, :]
    return result


def duplicate_mask(candidates: np.ndarray) -> np.ndarray:
    return np.any(np.diff(np.sort(candidates, axis=1, kind="stable"), axis=1) == 0, axis=1)


def remap_probability_slots(base_probability: np.ndarray, candidate_score: np.ndarray) -> np.ndarray:
    probability_order = np.argsort(base_probability, axis=1, kind="stable")
    score_order = np.argsort(candidate_score, axis=1, kind="stable")
    sorted_probability = np.take_along_axis(base_probability, probability_order, axis=1)
    output = np.empty_like(base_probability)
    output[np.arange(len(base_probability))[:, None], score_order] = sorted_probability
    return output


def parse_probability_lines(lines: list[str], member: str) -> np.ndarray:
    values = np.fromstring(",".join(line.strip() for line in lines), dtype=np.float64, sep=",")
    require(values.size == len(lines) * WIDTH, f"malformed CSV width in {member}")
    values = values.reshape(len(lines), WIDTH)
    require(np.isfinite(values).all(), f"non-finite probability in {member}")
    require(np.all((0.0 <= values) & (values <= 1.0)), f"probability out of range in {member}")
    return values


def iter_probability_chunks(archive_path: Path, member: str, chunk_rows: int) -> Iterator[np.ndarray]:
    with zipfile.ZipFile(archive_path) as archive:
        require(member in archive.namelist(), f"missing {member}")
        with archive.open(member, "r") as raw:
            with io.TextIOWrapper(raw, encoding="ascii", newline="") as text:
                lines: list[str] = []
                for line in text:
                    if line.strip():
                        lines.append(line)
                    if len(lines) == chunk_rows:
                        yield parse_probability_lines(lines, member)
                        lines.clear()
                if lines:
                    yield parse_probability_lines(lines, member)


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


def build_hierarchy_maps(
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


def hierarchy_feature_chunk(
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
    return qnorm(log_pair), qnorm(pmi)


def save_hierarchy_map(destination: Path, maps: dict[str, np.ndarray]) -> None:
    destination.mkdir()
    for name in ("pair_keys", "pair_counts", "src_keys", "src_counts", "dst_keys", "dst_counts"):
        np.save(destination / f"{name}.npy", maps[name], allow_pickle=False)
    (destination / "rows.txt").write_text(f"{int(maps['rows'])}\n", encoding="ascii")


def load_hierarchy_map(source: Path) -> dict[str, np.ndarray]:
    result = {
        name: np.load(source / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("pair_keys", "pair_counts", "src_keys", "src_counts", "dst_keys", "dst_counts")
    }
    result["rows"] = int((source / "rows.txt").read_text(encoding="ascii"))
    return result


def prepare_hierarchy_map_cache(
    cache: Path,
    source: np.ndarray,
    destination: np.ndarray,
    timestamp: np.ndarray,
    cutoff: int,
    maximum_candidate: int,
) -> list[tuple[int, int, dict[str, np.ndarray]]]:
    stop = int(np.searchsorted(timestamp, cutoff, side="left"))
    expected = {
        "kind": "formal_dataset4_hierarchy_maps_v1",
        "cutoff": int(cutoff),
        "history_rows": int(stop),
        "maximum_candidate": int(maximum_candidate),
        "full_levels": [list(value) for value in FULL_LEVELS],
        "recent_levels": [list(value) for value in RECENT_LEVELS],
        "recent_windows": list(RECENT_WINDOWS),
    }
    metadata = cache / "metadata.json"
    if metadata.exists():
        actual = json.loads(metadata.read_text(encoding="utf-8"))
        require(actual == expected, "hierarchy map cache metadata differs")
    elif cache.exists():
        raise ValueError(f"incomplete hierarchy map cache exists: {cache}")
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{cache.name}.", dir=cache.parent))
        try:
            index = 0
            for src_shift, dst_shift in FULL_LEVELS:
                base = (maximum_candidate >> dst_shift) + 1
                maps = build_hierarchy_maps(source, destination, 0, stop, src_shift, dst_shift, base)
                save_hierarchy_map(stage / f"feature_{index:02d}", maps)
                print(json.dumps({"hierarchy_map": index, "unique_pairs": len(maps["pair_keys"])}), flush=True)
                index += 1
            for window in RECENT_WINDOWS:
                start = int(np.searchsorted(timestamp, cutoff - window, side="left"))
                for src_shift, dst_shift in RECENT_LEVELS:
                    base = (maximum_candidate >> dst_shift) + 1
                    maps = build_hierarchy_maps(source, destination, start, stop, src_shift, dst_shift, base)
                    save_hierarchy_map(stage / f"feature_{index:02d}", maps)
                    print(json.dumps({"hierarchy_map": index, "window": window, "unique_pairs": len(maps["pair_keys"])}), flush=True)
                    index += 1
            atomic_json(stage / "metadata.json", expected)
            os.replace(stage, cache)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    maps = []
    index = 0
    for src_shift, dst_shift in FULL_LEVELS:
        maps.append((src_shift, dst_shift, load_hierarchy_map(cache / f"feature_{index:02d}")))
        index += 1
    for _window in RECENT_WINDOWS:
        for src_shift, dst_shift in RECENT_LEVELS:
            maps.append((src_shift, dst_shift, load_hierarchy_map(cache / f"feature_{index:02d}")))
            index += 1
    return maps


def hierarchy_features(
    maps: list[tuple[int, int, dict[str, np.ndarray]]],
    source: np.ndarray,
    candidates: np.ndarray,
    maximum_candidate: int,
) -> np.ndarray:
    output = np.empty((len(source), candidates.shape[1], 34), dtype=np.float32)
    channel = 0
    for src_shift, dst_shift, values in maps:
        base = (maximum_candidate >> dst_shift) + 1
        log_pair, pmi = hierarchy_feature_chunk(values, source, candidates, src_shift, dst_shift, base)
        output[:, :, channel] = log_pair
        output[:, :, channel + 1] = pmi
        channel += 2
    require(channel == 34 and np.isfinite(output).all(), "hierarchy features malformed")
    return output


def build_neighbor_adjacency(
    source: np.ndarray,
    destination: np.ndarray,
    timestamp: np.ndarray,
    stop: int,
    destination_base: int,
    source_count: int,
    destination_count: int,
    max_destination_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.asarray(source[:stop], dtype=np.int64) * np.int64(destination_base) + np.asarray(destination[:stop], dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_keys) != 0) + 1, len(sorted_keys)]
    last_positions = order[ends - 1]
    unique_keys = sorted_keys[ends - 1]
    unique_source = unique_keys // np.int64(destination_base)
    unique_destination = unique_keys % np.int64(destination_base)
    last_time = np.asarray(timestamp[last_positions], dtype=np.int64)
    source_degree = np.bincount(unique_source, minlength=source_count).astype(np.int64)
    source_offsets = np.empty(source_count + 1, dtype=np.int64)
    source_offsets[0] = 0
    np.cumsum(source_degree, out=source_offsets[1:])
    source_destination = unique_destination.astype(np.int32)
    destination_degree = np.bincount(unique_destination, minlength=destination_count).astype(np.int32)
    destination_order = np.lexsort((-last_time, unique_destination))
    ordered_destination = unique_destination[destination_order]
    group_start = np.r_[0, np.flatnonzero(np.diff(ordered_destination) != 0) + 1]
    group_length = np.diff(np.r_[group_start, len(ordered_destination)])
    rank_in_group = np.arange(len(ordered_destination)) - np.repeat(group_start, group_length)
    selected_order = destination_order[rank_in_group < max_destination_neighbors]
    selected_destination = unique_destination[selected_order]
    selected_source = unique_source[selected_order].astype(np.int32)
    selected_degree = np.bincount(selected_destination, minlength=destination_count).astype(np.int64)
    destination_offsets = np.empty(destination_count + 1, dtype=np.int64)
    destination_offsets[0] = 0
    np.cumsum(selected_degree, out=destination_offsets[1:])
    return source_offsets, source_destination, destination_offsets, selected_source, destination_degree


def prepare_neighbor_cache(
    cache: Path,
    source: np.ndarray,
    destination: np.ndarray,
    timestamp: np.ndarray,
    cutoff: int,
    maximum_source: int,
    maximum_candidate: int,
    max_destination_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stop = int(np.searchsorted(timestamp, cutoff, side="left"))
    expected = {
        "kind": "formal_dataset4_neighbor_adjacency_v1",
        "cutoff": int(cutoff),
        "history_rows": int(stop),
        "maximum_source": int(maximum_source),
        "maximum_candidate": int(maximum_candidate),
        "max_destination_neighbors": int(max_destination_neighbors),
    }
    metadata = cache / "metadata.json"
    names = ("source_offsets", "source_destination", "destination_offsets", "destination_source", "destination_degree")
    if metadata.exists():
        actual = json.loads(metadata.read_text(encoding="utf-8"))
        require(actual == expected, "neighbor cache metadata differs")
    elif cache.exists():
        raise ValueError(f"incomplete neighbor cache exists: {cache}")
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{cache.name}.", dir=cache.parent))
        try:
            arrays = build_neighbor_adjacency(
                source,
                destination,
                timestamp,
                stop,
                maximum_candidate + 1,
                maximum_source + 1,
                maximum_candidate + 1,
                max_destination_neighbors,
            )
            for name, values in zip(names, arrays):
                np.save(stage / f"{name}.npy", values, allow_pickle=False)
            atomic_json(stage / "metadata.json", expected)
            os.replace(stage, cache)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return tuple(np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names)


_NEIGHBOR_SCORER = None


def neighbor_features(adjacency: tuple[np.ndarray, ...], query_source: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    global _NEIGHBOR_SCORER
    if _NEIGHBOR_SCORER is None:
        from numba import njit

        @njit(cache=True)
        def score_queries(
            query_source,
            candidates,
            source_offsets,
            source_destination,
            destination_offsets,
            destination_source,
            destination_degree,
            output,
        ):
            for row in range(len(query_source)):
                query_id = int(query_source[row])
                if query_id + 1 >= len(source_offsets):
                    continue
                query_start = source_offsets[query_id]
                query_end = source_offsets[query_id + 1]
                query_degree = query_end - query_start
                if query_degree == 0:
                    continue
                for column in range(candidates.shape[1]):
                    candidate = int(candidates[row, column])
                    if candidate + 1 >= len(destination_offsets):
                        continue
                    neighbor_start = destination_offsets[candidate]
                    neighbor_end = destination_offsets[candidate + 1]
                    maximum_cosine = 0.0
                    maximum_jaccard = 0.0
                    maximum_rare = 0.0
                    top1 = 0.0
                    top2 = 0.0
                    top3 = 0.0
                    overlap_two = 0
                    for neighbor_index in range(neighbor_start, neighbor_end):
                        neighbor_id = int(destination_source[neighbor_index])
                        if neighbor_id == query_id or neighbor_id + 1 >= len(source_offsets):
                            continue
                        other_start = source_offsets[neighbor_id]
                        other_end = source_offsets[neighbor_id + 1]
                        other_degree = other_end - other_start
                        left = query_start
                        right = other_start
                        overlap = 0
                        rare = 0.0
                        while left < query_end and right < other_end:
                            query_destination = source_destination[left]
                            other_destination = source_destination[right]
                            if query_destination < other_destination:
                                left += 1
                            elif query_destination > other_destination:
                                right += 1
                            else:
                                overlap += 1
                                rare += 1.0 / np.sqrt(float(destination_degree[query_destination]) + 1.0)
                                left += 1
                                right += 1
                        if overlap == 0:
                            continue
                        cosine = overlap / np.sqrt(float(query_degree * other_degree))
                        jaccard = overlap / float(query_degree + other_degree - overlap)
                        if cosine > maximum_cosine:
                            maximum_cosine = cosine
                        if jaccard > maximum_jaccard:
                            maximum_jaccard = jaccard
                        if rare > maximum_rare:
                            maximum_rare = rare
                        if cosine > top1:
                            top3 = top2
                            top2 = top1
                            top1 = cosine
                        elif cosine > top2:
                            top3 = top2
                            top2 = cosine
                        elif cosine > top3:
                            top3 = cosine
                        if overlap >= 2:
                            overlap_two += 1
                    output[row, column, 0] = maximum_cosine
                    output[row, column, 1] = maximum_jaccard
                    output[row, column, 2] = top1 + top2 + top3
                    output[row, column, 3] = maximum_rare
                    output[row, column, 4] = np.log1p(overlap_two)

        _NEIGHBOR_SCORER = score_queries
    output = np.zeros((len(query_source), candidates.shape[1], len(NEIGHBOR_NAMES)), dtype=np.float32)
    _NEIGHBOR_SCORER(np.asarray(query_source), np.asarray(candidates), *adjacency, output)
    return output


def build_temporal_analogy_adjacency(
    source: np.ndarray,
    destination: np.ndarray,
    timestamp: np.ndarray,
    stop: int,
    destination_base: int,
    source_count: int,
    destination_count: int,
    max_destination_neighbors: int,
) -> tuple[np.ndarray, ...]:
    keys = np.asarray(source[:stop], dtype=np.int64) * np.int64(destination_base) + np.asarray(destination[:stop], dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_keys) != 0) + 1, len(sorted_keys)]
    last_positions = order[ends - 1]
    unique_keys = sorted_keys[ends - 1]
    unique_source = unique_keys // np.int64(destination_base)
    unique_destination = unique_keys % np.int64(destination_base)
    unique_last_time = np.asarray(timestamp[last_positions], dtype=np.int64)

    source_degree = np.bincount(unique_source, minlength=source_count).astype(np.int64)
    source_offsets = np.empty(source_count + 1, dtype=np.int64)
    source_offsets[0] = 0
    np.cumsum(source_degree, out=source_offsets[1:])
    source_destination = unique_destination.astype(np.int32)
    source_last_time = unique_last_time.astype(np.int64)

    time_order = np.lexsort((unique_last_time, unique_source))
    source_time = unique_last_time[time_order].astype(np.int64)

    destination_degree = np.bincount(unique_destination, minlength=destination_count).astype(np.int32)
    idf = np.log1p(np.float64(max(1, source_count)) / (destination_degree.astype(np.float64) + 1.0)).astype(np.float32)
    source_idf_norm2 = np.bincount(
        unique_source,
        weights=np.square(idf[unique_destination], dtype=np.float64),
        minlength=source_count,
    ).astype(np.float32)

    destination_order = np.lexsort((-unique_last_time, unique_destination))
    ordered_destination = unique_destination[destination_order]
    group_start = np.r_[0, np.flatnonzero(np.diff(ordered_destination) != 0) + 1]
    group_length = np.diff(np.r_[group_start, len(ordered_destination)])
    rank_in_group = np.arange(len(ordered_destination)) - np.repeat(group_start, group_length)
    selected = destination_order[rank_in_group < max_destination_neighbors]
    selected_destination = unique_destination[selected]
    selected_source = unique_source[selected].astype(np.int32)
    selected_time = unique_last_time[selected].astype(np.int64)
    selected_degree = np.bincount(selected_destination, minlength=destination_count).astype(np.int64)
    destination_offsets = np.empty(destination_count + 1, dtype=np.int64)
    destination_offsets[0] = 0
    np.cumsum(selected_degree, out=destination_offsets[1:])
    return (
        source_offsets,
        source_destination,
        source_last_time,
        source_time,
        source_idf_norm2,
        destination_offsets,
        selected_source,
        selected_time,
        destination_degree,
        idf,
    )


def prepare_temporal_analogy_cache(
    cache: Path,
    source: np.ndarray,
    destination: np.ndarray,
    timestamp: np.ndarray,
    cutoff: int,
    maximum_source: int,
    maximum_candidate: int,
    max_destination_neighbors: int,
) -> tuple[np.ndarray, ...]:
    stop = int(np.searchsorted(timestamp, cutoff, side="left"))
    expected = {
        "kind": "formal_dataset4_temporal_source_analogy_v1",
        "cutoff": int(cutoff),
        "history_rows": int(stop),
        "maximum_source": int(maximum_source),
        "maximum_candidate": int(maximum_candidate),
        "max_destination_neighbors": int(max_destination_neighbors),
        "feature_names": list(ANALOGY_NAMES),
    }
    metadata = cache / "metadata.json"
    names = (
        "source_offsets",
        "source_destination",
        "source_last_time",
        "source_time",
        "source_idf_norm2",
        "destination_offsets",
        "destination_source",
        "destination_source_time",
        "destination_degree",
        "idf",
    )
    if metadata.exists():
        actual = json.loads(metadata.read_text(encoding="utf-8"))
        require(actual == expected, "temporal analogy cache metadata differs")
    elif cache.exists():
        raise ValueError(f"incomplete temporal analogy cache exists: {cache}")
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{cache.name}.", dir=cache.parent))
        try:
            arrays = build_temporal_analogy_adjacency(
                source,
                destination,
                timestamp,
                stop,
                maximum_candidate + 1,
                maximum_source + 1,
                maximum_candidate + 1,
                max_destination_neighbors,
            )
            for name, values in zip(names, arrays):
                np.save(stage / f"{name}.npy", values, allow_pickle=False)
            atomic_json(stage / "metadata.json", expected)
            os.replace(stage, cache)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return tuple(np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names)


_ANALOGY_SCORER = None


def temporal_analogy_features(
    adjacency: tuple[np.ndarray, ...],
    query_source: np.ndarray,
    candidates: np.ndarray,
    cutoff: int,
) -> np.ndarray:
    global _ANALOGY_SCORER
    if _ANALOGY_SCORER is None:
        from numba import njit, prange

        @njit(cache=True, parallel=True)
        def score_queries(
            query_source,
            candidates,
            cutoff,
            source_offsets,
            source_destination,
            source_last_time,
            source_time,
            source_idf_norm2,
            destination_offsets,
            destination_source,
            destination_source_time,
            destination_degree,
            idf,
            output,
        ):
            for row in prange(len(query_source)):
                query_id = int(query_source[row])
                if query_id + 1 >= len(source_offsets):
                    continue
                query_start = source_offsets[query_id]
                query_end = source_offsets[query_id + 1]
                query_degree = query_end - query_start
                if query_degree == 0:
                    continue
                query_norm2 = source_idf_norm2[query_id]
                for column in range(candidates.shape[1]):
                    candidate = int(candidates[row, column])
                    if candidate + 1 >= len(destination_offsets):
                        continue
                    top_cosine = np.zeros(8, dtype=np.float64)
                    top_tfidf = np.zeros(3, dtype=np.float64)
                    top_prefix = np.zeros(3, dtype=np.float64)
                    top_recent_1h = np.zeros(3, dtype=np.float64)
                    top_recent_6h = np.zeros(3, dtype=np.float64)
                    top_recent_24h = np.zeros(3, dtype=np.float64)
                    overlap_two = 0
                    for neighbor_index in range(destination_offsets[candidate], destination_offsets[candidate + 1]):
                        neighbor_id = int(destination_source[neighbor_index])
                        if neighbor_id == query_id or neighbor_id + 1 >= len(source_offsets):
                            continue
                        other_start = source_offsets[neighbor_id]
                        other_end = source_offsets[neighbor_id + 1]
                        other_degree = other_end - other_start
                        if other_degree == 0:
                            continue
                        candidate_time = destination_source_time[neighbor_index]
                        left_bound = other_start
                        right_bound = other_end
                        while left_bound < right_bound:
                            middle = (left_bound + right_bound) // 2
                            if source_time[middle] <= candidate_time:
                                left_bound = middle + 1
                            else:
                                right_bound = middle
                        prefix_degree = left_bound - other_start
                        left = query_start
                        right = other_start
                        overlap = 0
                        prefix_overlap = 0
                        weighted_dot = 0.0
                        while left < query_end and right < other_end:
                            query_destination = source_destination[left]
                            other_destination = source_destination[right]
                            if query_destination < other_destination:
                                left += 1
                            elif query_destination > other_destination:
                                right += 1
                            else:
                                overlap += 1
                                weight = float(idf[query_destination])
                                weighted_dot += weight * weight
                                if source_last_time[right] <= candidate_time:
                                    prefix_overlap += 1
                                left += 1
                                right += 1
                        if overlap == 0:
                            continue
                        cosine = overlap / np.sqrt(float(query_degree * other_degree))
                        tfidf_cosine = weighted_dot / np.sqrt(max(1e-12, float(query_norm2 * source_idf_norm2[neighbor_id])))
                        prefix_cosine = prefix_overlap / np.sqrt(float(max(1, query_degree * prefix_degree)))
                        age = max(0.0, float(cutoff - candidate_time))
                        recent_1h = cosine * np.exp(-age / 3_600.0)
                        recent_6h = cosine * np.exp(-age / 21_600.0)
                        recent_24h = cosine * np.exp(-age / 86_400.0)
                        minimum_index = np.argmin(top_cosine)
                        if cosine > top_cosine[minimum_index]:
                            top_cosine[minimum_index] = cosine
                        minimum_index = np.argmin(top_tfidf)
                        if tfidf_cosine > top_tfidf[minimum_index]:
                            top_tfidf[minimum_index] = tfidf_cosine
                        minimum_index = np.argmin(top_prefix)
                        if prefix_cosine > top_prefix[minimum_index]:
                            top_prefix[minimum_index] = prefix_cosine
                        minimum_index = np.argmin(top_recent_1h)
                        if recent_1h > top_recent_1h[minimum_index]:
                            top_recent_1h[minimum_index] = recent_1h
                        minimum_index = np.argmin(top_recent_6h)
                        if recent_6h > top_recent_6h[minimum_index]:
                            top_recent_6h[minimum_index] = recent_6h
                        minimum_index = np.argmin(top_recent_24h)
                        if recent_24h > top_recent_24h[minimum_index]:
                            top_recent_24h[minimum_index] = recent_24h
                        if overlap >= 2:
                            overlap_two += 1
                    output[row, column, 0] = np.sum(np.sort(top_cosine)[-3:])
                    output[row, column, 1] = np.sum(top_cosine)
                    output[row, column, 2] = np.sum(top_tfidf)
                    output[row, column, 3] = np.sum(top_prefix)
                    output[row, column, 4] = np.sum(top_recent_1h)
                    output[row, column, 5] = np.sum(top_recent_6h)
                    output[row, column, 6] = np.sum(top_recent_24h)
                    output[row, column, 7] = np.max(top_tfidf)
                    output[row, column, 8] = np.max(top_prefix)
                    output[row, column, 9] = np.log1p(overlap_two)

        _ANALOGY_SCORER = score_queries
    output = np.zeros((len(query_source), candidates.shape[1], len(ANALOGY_NAMES)), dtype=np.float32)
    _ANALOGY_SCORER(np.asarray(query_source), np.asarray(candidates), int(cutoff), *adjacency, output)
    return output


def create_model(feature_count: int, hidden: int):
    import jittor as jt
    from jittor import nn

    class IntensityNet(nn.Module):
        def __init__(self) -> None:
            self.full = nn.Sequential(nn.Linear(22, hidden), nn.Relu(), nn.Linear(hidden, hidden), nn.Relu())
            self.recent = nn.Sequential(
                nn.Linear(feature_count - 22, hidden), nn.Relu(), nn.Linear(hidden, hidden), nn.Relu()
            )
            self.local = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Relu())
            self.output = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Relu(), nn.Linear(hidden, 1))

        def execute(self, values):
            full = self.full(values[:, :, :22])
            recent = self.recent(values[:, :, 22:])
            local = self.local(jt.concat((full, recent), dim=2))
            context = local.mean(dim=1, keepdims=True)
            context = context.broadcast((local.shape[0], local.shape[1], local.shape[2]))
            return self.output(jt.concat((full, recent, context), dim=2)).squeeze(-1)

    return IntensityNet()


def create_setmax_model(feature_count: int, hidden: int):
    import jittor as jt
    from jittor import nn

    class SetMaxNet(nn.Module):
        def __init__(self) -> None:
            self.item = nn.Sequential(
                nn.Linear(feature_count, hidden),
                nn.Relu(),
                nn.Linear(hidden, hidden),
                nn.Relu(),
            )
            self.output = nn.Sequential(
                nn.Linear(3 * hidden, hidden),
                nn.Relu(),
                nn.Linear(hidden, 1),
            )

        def execute(self, values):
            local = self.item(values)
            mean = local.mean(dim=1, keepdims=True)
            maximum = local.max(dim=1, keepdims=True)
            mean = mean.broadcast((local.shape[0], local.shape[1], local.shape[2]))
            maximum = maximum.broadcast((local.shape[0], local.shape[1], local.shape[2]))
            return self.output(jt.concat((local, mean, maximum), dim=2)).squeeze(-1)

    return SetMaxNet()


def load_meta_model(path: Path):
    import jittor as jt

    payload = np.load(path, allow_pickle=False)
    names = [str(value) for value in payload["names"].tolist()]
    state = {name: payload[f"state_{index}"] for index, name in enumerate(names)}
    if "item.0.weight" in state:
        hidden, feature_count = state["item.0.weight"].shape
        model = create_setmax_model(int(feature_count), int(hidden))
    else:
        hidden, full_features = state["full.0.weight"].shape
        feature_count = int(full_features + state["recent.0.weight"].shape[1])
        model = create_model(feature_count, int(hidden))
    require(names == list(model.state_dict()), "meta checkpoint parameter order differs")
    model.load_state_dict({name: jt.array(value) for name, value in state.items()})
    model.eval()
    return model, feature_count, int(hidden)


def predict_residual(models: list[Any], features: np.ndarray, batch_rows: int, no_recent_weight: float = 0.20) -> tuple[np.ndarray, float]:
    import jittor as jt

    output = np.empty(features.shape[:2], dtype=np.float32)
    max_error = 0.0
    with jt.no_grad():
        for start in range(0, len(features), batch_rows):
            stop = min(len(features), start + batch_rows)
            block = np.asarray(features[start:stop])
            member_values = []
            for model in models:
                full = qnorm(np.asarray(model(jt.array(block)).data, dtype=np.float32))
                reverse = qnorm(np.asarray(model(jt.array(block[:, ::-1].copy())).data, dtype=np.float32))[:, ::-1]
                max_error = max(max_error, float(np.max(np.abs(full - reverse))))
                no_recent = block.copy()
                no_recent[:, :, 22:] = 0.0
                no_recent = qnorm(np.asarray(model(jt.array(no_recent)).data, dtype=np.float32))
                full_weight = 1.0 - no_recent_weight
                member_values.append(qnorm(0.5 * full_weight * (full + reverse) + no_recent_weight * no_recent))
            output[start:stop] = qnorm(np.mean(member_values, axis=0))
    return output, max_error


def compose_and_gate(
    baseline: np.ndarray,
    residual: np.ndarray,
    duplicate: np.ndarray,
    scale: float,
    threshold: float,
    keep_same_top1: bool,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = baseline + np.float32(scale) * residual
    candidate[duplicate] = baseline[duplicate]
    base_top = np.argmax(baseline, axis=1)
    cand_top = np.argmax(candidate, axis=1)
    rows = np.arange(len(baseline))
    top_margin = candidate[rows, cand_top] - candidate[rows, base_top]
    active = top_margin >= threshold
    if keep_same_top1:
        active |= cand_top == base_top
    gated = baseline.copy()
    gated[active] = candidate[active]
    return gated, active


def stable_rank_of_choice(scores: np.ndarray, choice: np.ndarray) -> np.ndarray:
    rows = np.arange(len(scores))
    value = scores[rows, choice]
    columns = np.arange(scores.shape[1])
    rank = 1 + np.sum(scores > value[:, None], axis=1)
    rank += np.sum((scores == value[:, None]) & (columns[None, :] < choice[:, None]), axis=1)
    return rank.astype(np.float32)


def learned_gate_features(
    base: np.ndarray,
    candidate: np.ndarray,
    residual: np.ndarray,
    seen: np.ndarray,
    features: np.ndarray,
    scale: float,
) -> np.ndarray:
    rows = np.arange(len(base))
    base_top = np.argmax(base, axis=1)
    cand_top = np.argmax(candidate, axis=1)
    sorted_base = np.sort(base, axis=1)
    sorted_candidate = np.sort(candidate, axis=1)
    sorted_residual = np.sort(residual, axis=1)
    scalar = np.column_stack(
        (
            np.full(len(base), scale, dtype=np.float32),
            (cand_top != base_top).astype(np.float32),
            candidate[rows, cand_top] - candidate[rows, base_top],
            sorted_candidate[:, -1] - sorted_candidate[:, -2],
            sorted_base[:, -1] - sorted_base[:, -2],
            sorted_residual[:, -1] - sorted_residual[:, -2],
            np.max(np.abs(residual), axis=1),
            residual[rows, cand_top],
            residual[rows, base_top],
            residual[rows, cand_top] - residual[rows, base_top],
            base[rows, cand_top],
            base[rows, base_top],
            stable_rank_of_choice(base, cand_top) / np.float32(base.shape[1]),
            seen[rows, cand_top].astype(np.float32),
            seen[rows, base_top].astype(np.float32),
            seen.mean(axis=1).astype(np.float32),
        )
    ).astype(np.float32)
    base_feature = features[rows, base_top].astype(np.float32)
    cand_feature = features[rows, cand_top].astype(np.float32)
    return np.concatenate((scalar, cand_feature, base_feature, cand_feature - base_feature), axis=1)


def load_learned_gate(path: Path):
    import jittor as jt
    from jittor import nn

    payload = np.load(path, allow_pickle=False)
    input_dim = int(payload["input_dim"][0])
    hidden = int(payload["hidden"][0])
    threshold = float(payload["threshold"][0])
    scale = float(payload["scale"][0])
    names = [str(value) for value in payload["names"].tolist()]
    state = {name: payload[f"state_{index}"] for index, name in enumerate(names)}

    class GateNet(nn.Module):
        def __init__(self) -> None:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.Relu(),
                nn.Linear(hidden, hidden),
                nn.Relu(),
                nn.Linear(hidden, 1),
            )

        def execute(self, values):
            return self.net(values).squeeze(-1)

    gate = GateNet()
    require(names == list(gate.state_dict()), "learned gate checkpoint parameter order differs")
    gate.load_state_dict({name: jt.array(value) for name, value in state.items()})
    gate.eval()
    return gate, input_dim, hidden, scale, threshold


def predict_learned_gate(gate: Any, values: np.ndarray, batch_rows: int) -> np.ndarray:
    import jittor as jt

    output = np.empty(len(values), dtype=np.float32)
    with jt.no_grad():
        for start in range(0, len(values), batch_rows):
            stop = min(len(values), start + batch_rows)
            logits = gate(jt.array(values[start:stop]))
            output[start:stop] = np.asarray(jt.sigmoid(logits).data, dtype=np.float32)
    return output


def compose_and_learned_gate(
    baseline: np.ndarray,
    residual: np.ndarray,
    seen: np.ndarray,
    features: np.ndarray,
    duplicate: np.ndarray,
    scale: float,
    gate: Any,
    threshold: float,
    batch_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = baseline + np.float32(scale) * residual
    candidate[duplicate] = baseline[duplicate]
    gate_values = learned_gate_features(baseline, candidate, residual, seen, features, scale)
    score = predict_learned_gate(gate, gate_values, batch_rows)
    active = score >= np.float32(threshold)
    gated = baseline.copy()
    gated[active] = candidate[active]
    return gated, active, score


def scan_test(data_features: Any, data: Path, scene: str, chunk_rows: int) -> tuple[int, int, int]:
    cutoff = None
    max_source = 0
    max_candidate = 0
    for chunk in data_features.iter_test_chunks(data, scene, chunk_rows=chunk_rows):
        if cutoff is None:
            cutoff = int(chunk.time[0])
        max_source = max(max_source, int(chunk.src.max(initial=0)))
        max_candidate = max(max_candidate, int(chunk.candidates.max(initial=0)))
    require(cutoff is not None, "empty official test set")
    return int(cutoff), max_source, max_candidate


def extract_components(ensemble: Any, infer: Any, chunk: Any, batch_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = ensemble.store.features(chunk.src, chunk.time, chunk.candidates)
    candidate_seen = features[:, :, infer.PAIR_SEEN_INDEX] > 0.0
    values: dict[str, np.ndarray] = {}
    if ensemble.temporal_models:
        temporal = ensemble.history.lookup(chunk.src, chunk.time, chunk.candidates)
        for name, (history_size, model) in ensemble.temporal_models.items():
            values[name] = infer.temporal_attention_jittor.predict_scores(
                model,
                temporal.source_indices,
                temporal.candidate_indices,
                temporal.history_item_indices[:, -history_size:],
                temporal.log_time_deltas[:, -history_size:],
                features=features,
                batch_size=batch_rows,
            )
    for name, (model, source_ids, item_ids) in ensemble.mf_models.items():
        values[name] = infer.implicit_mf_jittor.predict_scores(
            model, chunk.src, chunk.candidates, source_ids, item_ids, batch_size=batch_rows
        )
    if ensemble.transition_mf_models:
        positions = np.searchsorted(ensemble.transition_index.source_ids, chunk.src)
        inside = positions < len(ensemble.transition_index.source_ids)
        matched = np.zeros(len(chunk.src), dtype=bool)
        matched[inside] = ensemble.transition_index.source_ids[positions[inside]] == chunk.src[inside]
        previous = np.zeros(len(chunk.src), dtype=np.uint32)
        previous[matched] = ensemble.transition_index.last_items[positions[matched]]
        for name, (model, source_ids, item_ids) in ensemble.transition_mf_models.items():
            values[name] = infer.implicit_mf_jittor.predict_scores(
                model, previous, chunk.candidates, source_ids, item_ids, batch_size=batch_rows
            )
    if "test_frequency" in ensemble.needed:
        values["test_frequency"] = infer._candidate_counts(chunk.candidates, ensemble.test_ids, ensemble.test_counts)
    if "pair_seen_recency" in ensemble.needed:
        values["pair_seen_recency"] = (
            3.0 * features[:, :, infer.PAIR_SEEN_INDEX]
            + features[:, :, infer.PAIR_RECENCY_INDEX]
            + features[:, :, infer.PAIR_LOG_COUNT_INDEX]
        )
    if "transition_last_count" in ensemble.needed:
        values["transition_last_count"] = ensemble.transition_index.score(chunk.src, chunk.candidates)
    if any(name.startswith("pool_source_") for name in ensemble.needed):
        pool_values = infer._pool_source_components(
            chunk.src,
            chunk.time,
            chunk.candidates,
            ensemble.repeated_source_pair_ids,
            ensemble.repeated_source_pair_counts,
            ensemble.test_source_ids,
            ensemble.test_source_counts,
            ensemble.test_ids,
            ensemble.test_counts,
            ensemble.source_pair_time_index,
            ensemble.source_pair_time_scores,
            ensemble.activity_cuts,
        )
        for name, score in zip(
            (
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
            ),
            pool_values,
        ):
            if name in ensemble.needed:
                values[name] = score
    normalized = {name: infer._qnorm(score) for name, score in values.items()}
    missing = [name for name in ensemble.residual_component_names if name not in normalized]
    require(not missing, f"missing formal components: {missing}")
    components = np.stack([normalized[name] for name in ensemble.residual_component_names], axis=2)
    return components.astype(np.float32), np.asarray(features, dtype=np.float32), candidate_seen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--map-cache", type=Path, required=True)
    parser.add_argument("--meta-model", type=Path, nargs="+", required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--control-fit", type=Path, required=True)
    parser.add_argument("--pairnew-report", type=Path, required=True)
    parser.add_argument("--temporal-report", type=Path, action="append", required=True)
    parser.add_argument("--mf-report", nargs=2, action="append", metavar=("NAME", "REPORT"), default=[])
    parser.add_argument("--transition-mf-report", nargs=2, action="append", metavar=("NAME", "REPORT"), default=[])
    parser.add_argument("--scale", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.08892796039581305)
    parser.add_argument("--keep-same-top1", action="store_true", default=True)
    parser.add_argument("--learned-gate-model", type=Path)
    parser.add_argument("--learned-gate-threshold", type=float)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--predict-batch", type=int, default=512)
    parser.add_argument("--cache-chunk-rows", type=int, default=250000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--use-cuda", type=int, choices=(0, 1), default=1)
    parser.add_argument("--skip-input-hash-check", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    require(args.output.suffix == ".zip", "output must be a zip")
    require(not args.output.exists() and not args.report_output.exists(), "refusing to overwrite output")
    if not args.skip_input_hash_check:
        require(sha256_file(args.data) == EXPECTED_DATA_SHA256, "official data SHA256 differs")
        require(sha256_file(args.base) == EXPECTED_BASE_SHA256, "ruc4 base SHA256 differs")
    sys.path.insert(0, str(args.code_root))
    from b_rank import d4_multimodel_infer as infer
    from b_rank import data_features, temporal_attention_jittor

    temporal_attention_jittor.configure_cuda()
    import jittor as jt

    jt.flags.use_cuda = args.use_cuda
    models = []
    model_metadata = []
    feature_count = None
    hidden_values = []
    for path in args.meta_model:
        model, current_feature_count, hidden = load_meta_model(path)
        if feature_count is None:
            feature_count = current_feature_count
        else:
            require(current_feature_count == feature_count, f"meta model feature count differs: {path}")
        require(current_feature_count in (75, 85), f"unsupported meta feature count: {current_feature_count}")
        models.append(model)
        hidden_values.append(hidden)
        model_metadata.append({"path": str(path), "sha256": sha256_file(path), "hidden": hidden})
    learned_gate = None
    learned_gate_metadata = None
    if args.learned_gate_model:
        learned_gate, gate_input_dim, gate_hidden, gate_scale, gate_threshold = load_learned_gate(args.learned_gate_model)
        args.scale = gate_scale
        if args.learned_gate_threshold is not None:
            gate_threshold = args.learned_gate_threshold
        learned_gate_metadata = {
            "path": str(args.learned_gate_model),
            "sha256": sha256_file(args.learned_gate_model),
            "input_dim": gate_input_dim,
            "hidden": gate_hidden,
            "scale": float(gate_scale),
            "threshold": float(gate_threshold),
        }

    ensemble_args = argparse.Namespace(
        data=args.data,
        cache_dir=args.cache_dir,
        fit_report=args.control_fit,
        pairnew_report=args.pairnew_report,
        fullset_report=None,
        temporal_report=args.temporal_report,
        mf_report=args.mf_report,
        transition_mf_report=args.transition_mf_report,
        cache_chunk_rows=args.cache_chunk_rows,
        test_chunk_rows=args.chunk_rows,
        predict_batch_rows=args.predict_batch,
    )
    ensemble = infer.D4Ensemble(ensemble_args)
    require(len(ensemble.residual_component_names) == 23, "component count differs from meta training")

    source = np.load(args.train_cache / "src.npy", mmap_mode="r")
    destination = np.load(args.train_cache / "dst.npy", mmap_mode="r")
    timestamp = np.load(args.train_cache / "time.npy", mmap_mode="r")
    cutoff, test_max_source, test_max_candidate = scan_test(data_features, args.data, "dataset4", args.chunk_rows)
    maximum_candidate = max(int(destination.max()), test_max_candidate)
    maximum_source = max(int(source.max()), test_max_source)
    hierarchy_maps = prepare_hierarchy_map_cache(
        args.map_cache / "hierarchy_v1",
        source,
        destination,
        timestamp,
        cutoff,
        maximum_candidate,
    )
    neighbor = prepare_neighbor_cache(
        args.map_cache / "neighbor_v1",
        source,
        destination,
        timestamp,
        cutoff,
        maximum_source,
        maximum_candidate,
        max_destination_neighbors=32,
    )
    analogy = None
    if feature_count == 85:
        analogy = prepare_temporal_analogy_cache(
            args.map_cache / "temporal_source_analogy_v1",
            source,
            destination,
            timestamp,
            cutoff,
            maximum_source,
            maximum_candidate,
            max_destination_neighbors=64,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    rows = 0
    active_rows = 0
    learned_gate_score_sum = 0.0
    duplicate_rows = 0
    top1_changed = 0
    order_changed = 0
    slot_max_error = 0.0
    row_sum_max_error = 0.0
    permutation_error = 0.0
    started = time.time()
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as output_zip:
            if args.max_rows is None:
                with zipfile.ZipFile(args.base) as base_zip:
                    with base_zip.open("dataset3.csv") as incoming, output_zip.open("dataset3.csv", "w", force_zip64=True) as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=8 << 20)
            test_iter = data_features.iter_test_chunks(args.data, "dataset4", chunk_rows=args.chunk_rows)
            base_iter = iter_probability_chunks(args.base, "dataset4.csv", args.chunk_rows)
            with output_zip.open("dataset4.csv", "w", force_zip64=True) as raw:
                with io.TextIOWrapper(raw, encoding="ascii", newline="\n") as text:
                    for test_chunk, base_probability in zip_longest(test_iter, base_iter):
                        require(test_chunk is not None and base_probability is not None, "test/base row count differs")
                        require(len(test_chunk.src) == len(base_probability), "test/base chunk size differs")
                        if args.max_rows is not None:
                            remaining = args.max_rows - rows
                            if remaining <= 0:
                                break
                            if remaining < len(base_probability):
                                base_probability = base_probability[:remaining]
                                test_chunk = data_features.TestChunk(
                                    row_start=test_chunk.row_start,
                                    src=test_chunk.src[:remaining],
                                    time=test_chunk.time[:remaining],
                                    candidates=test_chunk.candidates[:remaining],
                                )
                        components, static, seen = extract_components(ensemble, infer, test_chunk, args.predict_batch)
                        h_features = hierarchy_features(hierarchy_maps, test_chunk.src, test_chunk.candidates, maximum_candidate)
                        n_features = normalized_cube(neighbor_features(neighbor, test_chunk.src, test_chunk.candidates))
                        base_rank = strict_rank_scores(base_probability)
                        features_75 = np.concatenate(
                            (
                                h_features,
                                normalized_cube(components),
                                normalized_cube(static),
                                base_rank[:, :, None].astype(np.float32),
                                seen[:, :, None].astype(np.float32),
                                n_features,
                            ),
                            axis=2,
                        )
                        require(features_75.shape[2] == 75 and np.isfinite(features_75).all(), "formal 75-feature tensor malformed")
                        if feature_count == 85:
                            require(analogy is not None, "missing temporal analogy cache")
                            a_features = normalized_cube(
                                temporal_analogy_features(analogy, test_chunk.src, test_chunk.candidates, cutoff)
                            )
                            features = np.concatenate((features_75, a_features), axis=2)
                        else:
                            features = features_75
                        require(
                            features.shape[2] == feature_count and np.isfinite(features).all(),
                            "formal meta feature tensor malformed",
                        )
                        residual, error = predict_residual(models, features, args.predict_batch)
                        permutation_error = max(permutation_error, error)
                        duplicate = duplicate_mask(test_chunk.candidates)
                        if learned_gate is None:
                            gated_score, active = compose_and_gate(
                                base_rank,
                                residual,
                                duplicate,
                                args.scale,
                                args.threshold,
                                args.keep_same_top1,
                            )
                        else:
                            gated_score, active, gate_score = compose_and_learned_gate(
                                base_rank,
                                residual,
                                seen,
                                features,
                                duplicate,
                                args.scale,
                                learned_gate,
                                float(learned_gate_metadata["threshold"]),
                                args.predict_batch * 4,
                            )
                            learned_gate_score_sum += float(gate_score.sum())
                        probability = remap_probability_slots(base_probability, gated_score)
                        sorted_base = np.sort(base_probability, axis=1, kind="stable")
                        sorted_output = np.sort(probability, axis=1, kind="stable")
                        slot_max_error = max(slot_max_error, float(np.max(np.abs(sorted_base - sorted_output))))
                        row_sum_max_error = max(
                            row_sum_max_error,
                            float(np.max(np.abs(probability.sum(axis=1) - base_probability.sum(axis=1)))),
                        )
                        duplicate_rows += int(duplicate.sum())
                        active_rows += int(active.sum())
                        top1_changed += int(np.sum(np.argmax(base_probability, axis=1) != np.argmax(probability, axis=1)))
                        order_changed += int(
                            np.sum(
                                np.any(
                                    np.argsort(base_probability, axis=1, kind="stable")
                                    != np.argsort(probability, axis=1, kind="stable"),
                                    axis=1,
                                )
                            )
                        )
                        np.savetxt(text, probability, fmt="%.8f", delimiter=",")
                        rows += len(probability)
                        if rows % 50_000 < len(probability):
                            print(json.dumps({"rows": rows, "active": active_rows, "top1_changed": top1_changed, "elapsed": time.time() - started}), flush=True)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    full = args.max_rows is None
    checks = {
        "full_rows": (not full) or rows == ROWS["dataset4.csv"],
        "probability_slots_exact": slot_max_error == 0.0,
        "row_sums_preserved": row_sum_max_error <= 1e-12,
        "permutation_equivariant": permutation_error <= 1e-5,
        "feature_count": feature_count in (75, 85),
        "component_count": len(ensemble.residual_component_names) == 23,
    }
    report = {
        "kind": "ruc4_gated_meta_ranker_formal_inference_v1",
        "decision": "SMOKE_ONLY" if not full else ("PASS" if all(checks.values()) else "NO_GO"),
        "checks": checks,
        "data_sha256": sha256_file(args.data),
        "base_sha256": sha256_file(args.base),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": rows,
        "meta_models": model_metadata,
        "learned_gate": learned_gate_metadata,
        "gate": {
            "mode": "learned" if learned_gate_metadata else "all",
            "scale": float(args.scale),
            "score": "jittor_gate_probability" if learned_gate_metadata else "top_margin",
            "threshold": (
                float(learned_gate_metadata["threshold"])
                if learned_gate_metadata else float(args.threshold)
            ),
            "keep_same_top1": False if learned_gate_metadata else bool(args.keep_same_top1),
        },
        "component_names": list(ensemble.residual_component_names),
        "active_rows": active_rows,
        "active_rate": active_rows / max(1, rows),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": duplicate_rows / max(1, rows),
        "top1_changed_rows": top1_changed,
        "top1_changed_rate": top1_changed / max(1, rows),
        "learned_gate_mean_score": learned_gate_score_sum / max(1, rows) if learned_gate_metadata else None,
        "order_changed_rows": order_changed,
        "order_changed_rate": order_changed / max(1, rows),
        "probability_slot_max_error": slot_max_error,
        "row_sum_max_error": row_sum_max_error,
        "candidate_permutation_max_error": permutation_error,
        "elapsed_seconds": time.time() - started,
        "uses_test_labels": False,
        "external_data_used": False,
        "jittor_runtime": {
            "version": str(jt.__version__),
            "has_cuda": bool(jt.has_cuda),
            "use_cuda": bool(jt.flags.use_cuda),
        },
    }
    atomic_json(args.report_output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["decision"] in {"PASS", "SMOKE_ONLY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
