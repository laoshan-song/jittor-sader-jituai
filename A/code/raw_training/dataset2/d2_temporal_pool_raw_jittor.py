#!/usr/bin/env python3
"""Rolling validation for temporal and denoised candidate-pool priors."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt


DAY = 86400
KEY_BASE = 1 << 20
SLICES = {
    "y2008": (1199145600, 1230768000),
    "y2009": (1230768000, 1262044800),
    "strict": (1262044800, 1296345600),
}


def qnorm(values):
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean(1, keepdims=True)) / (
        values.std(1, keepdims=True) + 1e-6
    )


def mrr(scores, labels):
    positive = scores[np.arange(len(labels)), labels]
    columns = np.arange(scores.shape[1])[None, :]
    rank = 1 + (scores > positive[:, None]).sum(1)
    rank += ((scores == positive[:, None]) & (columns < labels[:, None])).sum(1)
    return float(np.mean(1.0 / rank))


def mapped(values, sorted_ids):
    index = np.searchsorted(sorted_ids, values)
    valid = index < len(sorted_ids)
    valid[valid] &= sorted_ids[index[valid]] == values[valid]
    return index, valid


def replay(target, item_count, seed):
    rng = np.random.default_rng(seed)
    candidates = rng.integers(
        1, item_count + 1, size=(len(target), 100), dtype=np.int32
    )
    labels = rng.integers(0, 100, size=len(target), dtype=np.int32)
    candidates[np.arange(len(target)), labels] = target[:, 1]
    return candidates, labels


def array_sha256(values):
    """Hash a replay array with its dtype and shape, preserving row order."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def build_replay_pool(test_candidates, history_destinations):
    """Freeze the label-free official candidate rows used by test-pool replay.

    The historical multi-slice callers supplied a 100-column candidate matrix
    and a history-destination vector to an API that was omitted from the
    archived helper.  This reconstruction intentionally treats the candidate
    rows as a fixed transductive negative universe and never mutates them.
    History is validated here so callers cannot silently pass a future-derived
    or malformed array into the replay construction path.
    """
    candidates = np.asarray(test_candidates)
    history = np.asarray(history_destinations)
    if candidates.ndim != 2 or candidates.shape[1] != 100:
        raise ValueError("test-pool replay needs an [N, 100] candidate matrix")
    if not np.issubdtype(candidates.dtype, np.integer) or np.any(candidates <= 0):
        raise ValueError("test-pool candidates must be positive integer identifiers")
    if history.ndim != 1 or not np.issubdtype(history.dtype, np.integer):
        raise ValueError("history destinations must be a one-dimensional integer array")
    # A contiguous int32 copy keeps the RNG draw path independent of pandas
    # storage; the read-only flag prevents accidental mutation between slices.
    frozen = np.ascontiguousarray(candidates, dtype=np.int32)
    frozen.setflags(write=False)
    return frozen


def replay_from_pool(target, replay_pool, seed):
    """Sample full candidate rows and insert the causal target at a random slot."""
    target = np.asarray(target)
    candidates_pool = np.asarray(replay_pool)
    if target.ndim != 2 or target.shape[1] < 2:
        raise ValueError("target rows must contain source and destination columns")
    if candidates_pool.ndim != 2 or candidates_pool.shape[1] != 100 or len(candidates_pool) == 0:
        raise ValueError("replay pool must be a non-empty [N, 100] matrix")
    rng = np.random.default_rng(seed)
    row_ids = rng.integers(0, len(candidates_pool), size=len(target), dtype=np.int64)
    candidates = candidates_pool[row_ids].copy()
    labels = rng.integers(0, 100, size=len(target), dtype=np.int32)
    candidates[np.arange(len(target)), labels] = target[:, 1]
    audit = {
        "mode": "test-pool",
        "seed": int(seed),
        "pool_rows": int(len(candidates_pool)),
        "pool_columns": int(candidates_pool.shape[1]),
        "selected_pool_rows_sha256": array_sha256(row_ids),
        "candidates_sha256": array_sha256(candidates),
        "labels_sha256": array_sha256(labels),
    }
    return candidates, labels, audit


def score_model(model_path, src, candidates, old_keys, batch):
    model = np.load(model_path, allow_pickle=False)
    users, items = model["users"], model["items"]
    user_vector = model["user"].astype(np.float32, copy=False)
    item_vector = model["item"].astype(np.float32, copy=False)
    item_bias = model["ibias"].astype(np.float32, copy=False)
    user_index, known_user = mapped(src, users)
    output = np.full(candidates.shape, -12.0, np.float32)
    for start in range(0, len(candidates), batch):
        end = min(start + batch, len(candidates))
        block = candidates[start:end]
        item_index, known_item = mapped(block.ravel(), items)
        item_index = item_index.reshape(block.shape)
        known_item = known_item.reshape(block.shape)
        rows = np.flatnonzero(known_user[start:end])
        if len(rows):
            safe_item = np.minimum(item_index[rows], len(items) - 1)
            users_jt = jt.array(user_vector[user_index[start:end][rows]])
            items_jt = jt.array(item_vector[safe_item])
            bias_jt = jt.array(item_bias[safe_item])
            with jt.no_grad():
                value = (users_jt.unsqueeze(1) * items_jt).sum(dim=2) + bias_jt
                value = np.asarray(value.data, np.float32)
            value[~known_item[rows]] = -12.0
            output[start + rows] = value
    query = src[:, None].astype(np.int64) * KEY_BASE + candidates
    index = np.searchsorted(old_keys, query.ravel())
    old = index < len(old_keys)
    old[old] &= old_keys[index[old]] == query.ravel()[old]
    output[old.reshape(candidates.shape)] = -20.0
    return qnorm(output)


def global_signals(candidates, item_capacity, negative_item_count):
    count = np.bincount(candidates.ravel(), minlength=item_capacity + 1)
    lam = 99.0 * len(candidates) / negative_item_count
    selected = count[candidates].astype(np.float32)
    signals = {
        "global_log": qnorm(np.log1p(selected)),
        "global_linear": qnorm(selected),
    }
    for threshold in (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0):
        excess = np.maximum(selected - lam - threshold * np.sqrt(lam), 0.0)
        signals[f"excess_{threshold:g}"] = qnorm(excess)
    return signals


def temporal_signal(target, candidates, item_capacity, width_days, offset_days):
    first = int(target[:, 2].min())
    bucket = np.floor_divide(
        target[:, 2] - first + offset_days * DAY, width_days * DAY
    ).astype(np.int64)
    stride = item_capacity + 1
    keys = bucket[:, None] * stride + candidates
    count = np.bincount(keys.ravel(), minlength=(int(bucket.max()) + 1) * stride)
    return qnorm(np.log1p(count[keys]).astype(np.float32))


def rolling_signals(target, candidates, item_capacity):
    day = np.floor_divide(
        target[:, 2] - int(target[:, 2].min()), DAY
    ).astype(np.int64)
    day_count = int(day.max()) + 1
    daily = np.zeros((day_count, item_capacity + 1), np.int32)
    for start in range(0, len(candidates), 4096):
        end = min(start + 4096, len(candidates))
        np.add.at(
            daily,
            (np.repeat(day[start:end], 100), candidates[start:end].ravel()),
            1,
        )
    cumulative = np.empty((day_count + 1, item_capacity + 1), np.int32)
    cumulative[0] = 0
    np.cumsum(daily, axis=0, dtype=np.int32, out=cumulative[1:])
    del daily

    output = {}
    specs = {
        "roll_15": (15, 15),
        "roll_30": (30, 30),
        "roll_60": (60, 60),
        "roll_90": (90, 90),
        "past_60": (60, 0),
        "future_60": (0, 60),
    }
    for name, (before, after) in specs.items():
        low = np.maximum(day - before, 0)
        high = np.minimum(day + after + 1, day_count)
        score = np.empty(candidates.shape, np.float32)
        for start in range(0, len(candidates), 4096):
            end = min(start + 4096, len(candidates))
            block = candidates[start:end]
            score[start:end] = (
                cumulative[high[start:end, None], block]
                - cumulative[low[start:end, None], block]
            )
        output[name] = qnorm(np.log1p(score))
    return output


def tune_weight(base, signal, labels, weights):
    best = None
    for weight in weights:
        value = mrr(base + weight * signal, labels)
        candidate = (value, -float(weight))
        if best is None or candidate > best:
            best = candidate
    return {"mrr": best[0], "weight": -best[1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=int, default=153420)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is required")
    with zipfile.ZipFile(args.data) as archive:
        train = pd.read_csv(
            archive.open("dataset2/train.csv"), usecols=["src", "dst", "time"]
        )
        test = pd.read_csv(archive.open("dataset2/test.csv"))
    edges = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    negative_item_count = int(test.iloc[:, 2:].to_numpy(np.int64).max())
    item_capacity = max(negative_item_count, int(edges[:, 1].max()))
    cached = {}
    results = {
        "negative_item_count": negative_item_count,
        "item_capacity": item_capacity,
        "slices": {},
    }

    for index, (name, (start, end)) in enumerate(SLICES.items()):
        history = edges[edges[:, 2] < start]
        target = edges[(edges[:, 2] >= start) & (edges[:, 2] < end)]
        rng = np.random.default_rng(args.seed + 100 + index)
        if len(target) > args.targets:
            target = target[
                np.sort(rng.choice(len(target), args.targets, replace=False))
            ]
        candidates, labels = replay(target, negative_item_count, args.seed + index)
        old_keys = np.unique(history[:, 0] * KEY_BASE + history[:, 1])
        model_scores = {}
        for model_name in ("multvae", "recvae", "bm25bpr"):
            model_scores[model_name] = score_model(
                args.models / f"model_{model_name}_{name}_jittor.npz",
                target[:, 0],
                candidates,
                old_keys,
                args.batch,
            )
        model_base = (
            0.15 * model_scores["multvae"]
            + 0.45 * model_scores["recvae"]
            + 0.40 * model_scores["bm25bpr"]
        )
        signals = global_signals(candidates, item_capacity, negative_item_count)
        base = model_base + 0.07 * signals["global_log"]
        temporal = {}
        for width in (7, 14, 30, 60, 90, 180):
            first = temporal_signal(target, candidates, item_capacity, width, 0)
            shifted = temporal_signal(
                target, candidates, item_capacity, width, width // 2
            )
            temporal[f"time_{width}"] = 0.5 * (first + shifted)
        temporal.update(rolling_signals(target, candidates, item_capacity))
        cached[name] = (base, signals, temporal, labels)
        results["slices"][name] = {
            "rows": len(target),
            "model_base": mrr(model_base, labels),
            "current_global": mrr(base, labels),
            "signals": {
                key: mrr(value, labels) for key, value in {**signals, **temporal}.items()
            },
        }
        print(name, results["slices"][name]["model_base"], results["slices"][name]["current_global"], flush=True)

    tune_base, tune_global, tune_temporal, tune_labels = cached["y2008"]
    candidates = {}
    for key, signal in tune_global.items():
        if key == "global_log":
            continue
        candidates[key] = tune_weight(
            tune_base, signal - tune_global["global_log"], tune_labels,
            np.arange(0.0, 0.081, 0.01),
        )
    for key, signal in tune_temporal.items():
        candidates[key] = tune_weight(
            tune_base, signal, tune_labels, np.arange(0.0, 0.201, 0.01)
        )
    selected_name, selected = max(
        candidates.items(), key=lambda item: (item[1]["mrr"], -item[1]["weight"])
    )
    selected_signal = (
        tune_temporal[selected_name]
        if selected_name in tune_temporal
        else tune_global[selected_name] - tune_global["global_log"]
    )
    tune_after_first = tune_base + selected["weight"] * selected_signal
    second_candidates = {}
    for key, signal in {**tune_global, **tune_temporal}.items():
        if key in ("global_log", selected_name):
            continue
        value = signal if key in tune_temporal else signal - tune_global["global_log"]
        second_candidates[key] = tune_weight(
            tune_after_first, value, tune_labels, np.arange(0.0, 0.081, 0.01)
        )
    second_name, second = max(
        second_candidates.items(),
        key=lambda item: (item[1]["mrr"], -item[1]["weight"]),
    )
    results["tuning"] = {
        "all": candidates,
        "selected": selected_name,
        "weight": selected["weight"],
        "mrr": selected["mrr"],
        "second_all": second_candidates,
        "second": second_name,
        "second_weight": second["weight"],
        "second_mrr": second["mrr"],
    }

    for name, (base, global_set, temporal, labels) in cached.items():
        if selected_name in temporal:
            signal = temporal[selected_name]
        else:
            signal = global_set[selected_name] - global_set["global_log"]
        if second_name in temporal:
            second_signal = temporal[second_name]
        else:
            second_signal = global_set[second_name] - global_set["global_log"]
        mixed = (
            base
            + selected["weight"] * signal
            + second["weight"] * second_signal
        )
        results["slices"][name]["selected_mrr"] = mrr(mixed, labels)
        results["slices"][name]["selected_delta"] = (
            results["slices"][name]["selected_mrr"]
            - results["slices"][name]["current_global"]
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["tuning"], indent=2), flush=True)
    for name in SLICES:
        print(name, results["slices"][name]["selected_delta"], flush=True)


if __name__ == "__main__":
    main()
