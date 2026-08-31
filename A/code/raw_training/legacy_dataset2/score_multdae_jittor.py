#!/usr/bin/env python3
"""Recreate official dataset2 MultDAE scores from the saved Jittor export."""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

site_paths = [path for path in sys.path if "site-packages" in path]
if os.environ.get("PYTHONPATH"):
    site_paths.append(os.environ["PYTHONPATH"])
os.environ["PYTHONPATH"] = os.pathsep.join(site_paths)
os.environ["HOME"] = os.environ.get("ML_CACHE_ROOT", "/tmp")
if "--cuda" not in sys.argv and os.environ.get("JT_USE_CUDA") != "1":
    os.environ["nvcc_path"] = ""
os.environ.setdefault("log_silent", "1")
os.environ.setdefault("use_mpi", "0")

import jittor as jt


KEY_BASE = 1 << 20
EXPECTED_ROWS = 153_420
EXPECTED_RAW_SHA256 = "2cc482284461724d43eaa85c1fc276ec0d83c9d36615155d88729cd3042a1785"


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


def old_mask(src, cand, keys):
    query = src[:, None].astype(np.int64) * KEY_BASE + cand
    index = np.searchsorted(keys, query.ravel())
    old = index < len(keys)
    old[old] &= keys[index[old]] == query.ravel()[old]
    return old.reshape(cand.shape)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=here.parent / "data_A.zip")
    parser.add_argument("--model", type=Path, default=here / "model_prod_full_jittor.npz")
    parser.add_argument("--output", type=Path, default=here / "official_raw_rebuilt.npy")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    if args.cuda and jt.has_cuda:
        jt.flags.use_cuda = 1

    with zipfile.ZipFile(args.data) as archive:
        train = pd.read_csv(archive.open("dataset2/train.csv"), usecols=["src", "dst", "time"])
        test = pd.read_csv(archive.open("dataset2/test.csv"))
    history = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    src = test.src.to_numpy(np.int64, copy=False)
    cand = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    model = np.load(args.model, allow_pickle=False)

    assert len(history) == 2_261_283
    assert int(history[:, 2].max()) < int(test.time.min())
    assert int(model["history_end"]) == int(test.time.min())
    assert cand.shape == (EXPECTED_ROWS, 100)

    users, items = model["users"], model["items"]
    hidden, weight, bias = model["user"], model["item"], model["ibias"]
    user_index, known_user = mapped(src, users)
    keys = np.unique(history[:, 0] * KEY_BASE + history[:, 1])
    output = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float32, shape=cand.shape)

    for start in range(0, len(cand), args.batch):
        end = min(start + args.batch, len(cand))
        block = cand[start:end]
        item_index, known_item = mapped(block.ravel(), items)
        item_index = item_index.reshape(block.shape)
        known_item = known_item.reshape(block.shape)
        score = np.full(block.shape, -12.0, np.float32)
        rows = np.flatnonzero(known_user[start:end])
        if len(rows):
            safe_index = np.minimum(item_index[rows], len(items) - 1)
            user_vector = jt.array(hidden[user_index[start:end][rows]])
            item_vector = jt.array(weight[safe_index])
            item_bias = jt.array(bias[safe_index])
            with jt.no_grad():
                value = np.asarray(
                    ((user_vector.unsqueeze(1) * item_vector).sum(dim=2) + item_bias).data,
                    np.float32,
                )
            value[~known_item[rows]] = -12.0
            score[rows] = value
        score[old_mask(src[start:end], block, keys)] = -20.0
        output[start:end] = score
    output.flush()

    actual = sha256(args.output)
    print(f"wrote {args.output} sha256={actual}")
    if args.output.name == "official_raw_jittor.npy":
        assert actual == EXPECTED_RAW_SHA256


if __name__ == "__main__":
    main()
