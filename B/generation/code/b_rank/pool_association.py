#!/usr/bin/env python3
"""Exact source-candidate recurrence features for an unlabeled query pool."""

from __future__ import annotations

import numpy as np


PAIR_TIME_DTYPE = np.dtype([("key", "<u8"), ("time", "<u4")])


def leave_one_out_calibration(
    source_count: np.ndarray,
    source_rows: np.ndarray,
    candidate_probability: np.ndarray,
    activity_cuts: tuple[int, int],
) -> np.ndarray:
    """Calibrate extra pair occurrences against the 99-negative null."""
    count = np.asarray(source_count, dtype=np.float64)
    rows = np.asarray(source_rows, dtype=np.float64)
    probability = np.asarray(candidate_probability, dtype=np.float64)
    if (
        count.ndim != 2
        or probability.shape != count.shape
        or rows.shape != (len(count),)
        or not np.isfinite(count).all()
        or not np.isfinite(rows).all()
        or not np.isfinite(probability).all()
        or np.any(count < 1.0)
        or np.any(rows < 1.0)
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
        or len(activity_cuts) != 2
        or activity_cuts[0] < 1
        or activity_cuts[0] > activity_cuts[1]
    ):
        raise ValueError("invalid source-pool calibration arrays")
    observed = np.maximum(count - 1.0, 0.0)
    expected = 99.0 * np.maximum(rows - 1.0, 0.0)[:, None] * probability
    smoothed_observed = observed + 0.5
    smoothed_expected = expected + 0.5
    log_lift = np.log(smoothed_observed / smoothed_expected)
    deviance = 2.0 * (
        smoothed_observed * log_lift
        - (smoothed_observed - smoothed_expected)
    )
    signed_deviance = np.sign(observed - expected) * np.sqrt(
        np.maximum(deviance, 0.0)
    )
    bands = (
        rows <= activity_cuts[0],
        (rows > activity_cuts[0]) & (rows <= activity_cuts[1]),
        rows > activity_cuts[1],
    )
    return np.stack(
        [
            log_lift,
            signed_deviance,
            *(signed_deviance * mask[:, None] for mask in bands),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def source_candidate_keys(src: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    source = np.asarray(src, dtype=np.uint64)
    candidate = np.asarray(candidates, dtype=np.uint64)
    if candidate.ndim != 2 or len(source) != len(candidate):
        raise ValueError("source/candidate shapes differ")
    return (source[:, None] << np.uint64(32)) | candidate


def nearest_repeat_score(keys: np.ndarray, query_time: np.ndarray) -> np.ndarray:
    """Score each cell by the nearest other occurrence of its exact pair."""
    keys = np.asarray(keys, dtype=np.uint64)
    query_time = np.asarray(query_time, dtype=np.uint32)
    if keys.ndim != 2 or len(keys) != len(query_time):
        raise ValueError("pair-key/time shapes differ")
    flat_keys = keys.reshape(-1)
    flat_time = np.repeat(query_time, keys.shape[1])
    order = np.lexsort((flat_time, flat_keys))
    sorted_keys = flat_keys[order]
    sorted_time = flat_time[order].astype(np.int64, copy=False)
    sentinel = np.iinfo(np.int64).max
    nearest = np.full(len(order), sentinel, dtype=np.int64)
    same = sorted_keys[1:] == sorted_keys[:-1]
    positions = np.flatnonzero(same)
    gaps = np.abs(sorted_time[1:][same] - sorted_time[:-1][same])
    nearest[positions] = np.minimum(nearest[positions], gaps)
    nearest[positions + 1] = np.minimum(nearest[positions + 1], gaps)
    repeated = nearest != sentinel
    sorted_score = np.zeros(len(order), dtype=np.float32)
    sorted_score[repeated] = 1.0 / (
        1.0 + np.log1p(nearest[repeated].astype(np.float64))
    )
    score = np.empty(len(order), dtype=np.float32)
    score[order] = sorted_score
    return score.reshape(keys.shape)


def pair_time_index(
    keys: np.ndarray, query_time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse repeated occurrences into a sorted exact pair-time lookup."""
    keys = np.asarray(keys, dtype=np.uint64).reshape(-1)
    query_time = np.asarray(query_time, dtype=np.uint32).reshape(-1)
    if len(keys) != len(query_time):
        raise ValueError("pair-key/time lengths differ")
    if not len(keys):
        return np.empty(0, dtype=PAIR_TIME_DTYPE), np.empty(0, dtype=np.float32)
    order = np.lexsort((query_time, keys))
    sorted_keys = keys[order]
    sorted_time = query_time[order]
    sentinel = np.iinfo(np.int64).max
    nearest = np.full(len(order), sentinel, dtype=np.int64)
    same_key = sorted_keys[1:] == sorted_keys[:-1]
    positions = np.flatnonzero(same_key)
    gaps = np.abs(
        sorted_time[1:][same_key].astype(np.int64)
        - sorted_time[:-1][same_key].astype(np.int64)
    )
    nearest[positions] = np.minimum(nearest[positions], gaps)
    nearest[positions + 1] = np.minimum(nearest[positions + 1], gaps)
    if np.any(nearest == sentinel):
        raise ValueError("time index contains a non-repeated source-candidate pair")
    scores = 1.0 / (1.0 + np.log1p(nearest.astype(np.float64)))
    first = np.ones(len(order), dtype=bool)
    first[1:] = (sorted_keys[1:] != sorted_keys[:-1]) | (
        sorted_time[1:] != sorted_time[:-1]
    )
    index = np.empty(int(first.sum()), dtype=PAIR_TIME_DTYPE)
    index["key"] = sorted_keys[first]
    index["time"] = sorted_time[first]
    return index, scores[first].astype(np.float32)


def lookup_pair_time_scores(
    src: np.ndarray,
    candidates: np.ndarray,
    query_time: np.ndarray,
    index: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    keys = source_candidate_keys(src, candidates)
    query = np.empty(keys.size, dtype=PAIR_TIME_DTYPE)
    query["key"] = keys.reshape(-1)
    query["time"] = np.repeat(
        np.asarray(query_time, dtype=np.uint32), keys.shape[1]
    )
    positions = np.searchsorted(index, query)
    inside = positions < len(index)
    output = np.zeros(len(query), dtype=np.float32)
    matched = np.zeros(len(query), dtype=bool)
    matched[inside] = index[positions[inside]] == query[inside]
    output[matched] = np.asarray(scores[positions[matched]], dtype=np.float32)
    return output.reshape(keys.shape)


def _self_check() -> None:
    src = np.asarray([1, 1, 1], dtype=np.uint32)
    candidates = np.asarray([[10, 20], [10, 30], [10, 20]], dtype=np.uint32)
    query_time = np.asarray([100, 103, 110], dtype=np.uint32)
    keys = source_candidate_keys(src, candidates)
    direct = nearest_repeat_score(keys, query_time)
    repeated = np.asarray([True, True, True, False, True, True])
    flat_time = np.repeat(query_time, candidates.shape[1])
    index, scores = pair_time_index(keys.reshape(-1)[repeated], flat_time[repeated])
    looked_up = lookup_pair_time_scores(src, candidates, query_time, index, scores)
    np.testing.assert_allclose(looked_up, direct, rtol=0.0, atol=0.0)
    assert direct[1, 1] == 0.0
    assert direct[0, 0] > direct[2, 0] > 0.0

    calibrated = leave_one_out_calibration(
        np.asarray([[1, 2, 3], [1, 1, 1]], dtype=np.float32),
        np.asarray([3, 1], dtype=np.float32),
        np.asarray([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]], dtype=np.float32),
        (2, 5),
    )
    assert calibrated.shape == (5, 2, 3)
    assert np.isfinite(calibrated).all()
    assert calibrated[0, 0, 2] > calibrated[0, 0, 1] > calibrated[0, 0, 0]
    assert not np.any(calibrated[:, 1])
    np.testing.assert_allclose(
        calibrated[1], calibrated[2] + calibrated[3] + calibrated[4]
    )
    permutation = np.asarray([2, 0, 1])
    permuted = leave_one_out_calibration(
        np.asarray([[1, 2, 3], [1, 1, 1]], dtype=np.float32)[:, permutation],
        np.asarray([3, 1], dtype=np.float32),
        np.asarray([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]], dtype=np.float32)[
            :, permutation
        ],
        (2, 5),
    )
    np.testing.assert_allclose(permuted, calibrated[:, :, permutation])


if __name__ == "__main__":
    _self_check()
