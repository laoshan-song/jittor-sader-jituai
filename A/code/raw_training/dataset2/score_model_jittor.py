#!/usr/bin/env python3
"""Score official dataset2 candidates with a saved Jittor model export."""

import argparse
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt


KEY_BASE = 1 << 20


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mapped(values, sorted_ids):
    values = np.asarray(values)
    index = np.searchsorted(sorted_ids, values)
    valid = index < len(sorted_ids)
    valid[valid] &= sorted_ids[index[valid]] == values[valid]
    return index, valid


def old_mask(src, candidates, keys):
    query = src[:, None].astype(np.int64) * KEY_BASE + candidates
    index = np.searchsorted(keys, query.ravel())
    old = index < len(keys)
    old[old] &= keys[index[old]] == query.ravel()[old]
    return old.reshape(candidates.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1024)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is required")
    with zipfile.ZipFile(args.data) as archive:
        train = pd.read_csv(
            archive.open("dataset2/train.csv"), usecols=["src", "dst", "time"]
        )
        test = pd.read_csv(archive.open("dataset2/test.csv"))
    history = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    src = test.src.to_numpy(np.int64, copy=False)
    candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    model = np.load(args.model, allow_pickle=False)

    if int(model["history_end"]) != int(test.time.min()):
        raise ValueError("model history boundary does not match official test")
    users, items = model["users"], model["items"]
    user_vector = model["user"].astype(np.float32, copy=False)
    item_vector = model["item"].astype(np.float32, copy=False)
    item_bias = model["ibias"].astype(np.float32, copy=False)
    user_index, known_user = mapped(src, users)
    old_keys = np.unique(history[:, 0] * KEY_BASE + history[:, 1])
    output = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float32, shape=candidates.shape
    )

    for start in range(0, len(candidates), args.batch):
        end = min(start + args.batch, len(candidates))
        block = candidates[start:end]
        item_index, known_item = mapped(block.ravel(), items)
        item_index = item_index.reshape(block.shape)
        known_item = known_item.reshape(block.shape)
        score = np.full(block.shape, -12.0, np.float32)
        rows = np.flatnonzero(known_user[start:end])
        if len(rows):
            safe_item = np.minimum(item_index[rows], len(items) - 1)
            users_jt = jt.array(user_vector[user_index[start:end][rows]])
            items_jt = jt.array(item_vector[safe_item])
            bias_jt = jt.array(item_bias[safe_item])
            with jt.no_grad():
                value = (
                    (users_jt.unsqueeze(1) * items_jt).sum(dim=2) + bias_jt
                )
                value_np = np.asarray(value.data, np.float32)
            value_np[~known_item[rows]] = -12.0
            score[rows] = value_np
        score[old_mask(src[start:end], block, old_keys)] = -20.0
        output[start:end] = score
    output.flush()
    print(
        f"wrote {args.output} kind={str(model['kind'])} sha256={sha256(args.output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
