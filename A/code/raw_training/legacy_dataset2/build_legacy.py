#!/usr/bin/env python3
"""Rebuild the frozen legacy dataset2 score checkpoint from its components."""

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DATASET2_SHA256 = "554e6a89ce4a1c8a05693602b14daa222fb1fe4ced9d4a0693b06de5a412e708"


def qnorm(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean(1, keepdims=True)) / (
        values.std(1, keepdims=True) + 1e-6
    )


def softmax(values):
    values = values - values.max(1, keepdims=True)
    output = np.exp(values)
    return output / output.sum(1, keepdims=True)


def load_probability(path):
    with zipfile.ZipFile(path) as archive:
        return np.loadtxt(
            io.BytesIO(archive.read("dataset2.csv")), delimiter=",", dtype=np.float64
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--friend", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--cf", type=Path, required=True)
    parser.add_argument("--multdae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    friend = load_probability(args.friend)
    ours = load_probability(args.ours)
    cf = load_probability(args.cf)
    multdae = np.load(args.multdae, mmap_mode="r")
    expected_shape = (153420, 100)
    if any(x.shape != expected_shape for x in (friend, ours, cf, multdae)):
        raise ValueError("legacy component shape mismatch")

    with zipfile.ZipFile(args.data) as archive:
        train = pd.read_csv(
            archive.open("dataset2/train.csv"), usecols=["dst", "time"]
        )
        test = pd.read_csv(archive.open("dataset2/test.csv"))
    candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    dst = train.dst.to_numpy(np.int64, copy=False)
    tim = train.time.to_numpy(np.float64, copy=False)
    span = max(float(tim.max() - tim.min()), 1.0)
    edge_weight = np.exp(-(tim.max() - tim) / (span / 8.0))
    popularity = np.bincount(
        dst, weights=edge_weight, minlength=int(candidates.max()) + 1
    )

    v18 = (
        0.493 * qnorm(np.log(np.clip(friend, 1e-12, None)))
        + 0.357 * qnorm(np.log(np.clip(ours, 1e-12, None)))
        + 0.150 * qnorm(np.log(np.clip(cf, 1e-12, None)))
    )
    v19 = v18 + 0.04 * qnorm(np.log1p(popularity[candidates]))
    probability = softmax(0.70 * qnorm(v19) + 0.30 * qnorm(multdae))
    text = "".join(
        ",".join(f"{value:.8f}" for value in row) + "\n"
        for row in probability
    ).encode("ascii")
    digest = hashlib.sha256(text).hexdigest()
    if args.require_exact and digest != EXPECTED_DATASET2_SHA256:
        raise ValueError(f"legacy hash mismatch: {digest}")
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dataset2.csv", text)
    print(f"wrote {args.output} dataset2_sha256={digest}")


if __name__ == "__main__":
    main()
