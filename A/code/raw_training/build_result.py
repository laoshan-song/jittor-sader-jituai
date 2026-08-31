#!/usr/bin/env python3
"""Build the candidate-pool-deconvolution submission from model outputs."""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


WEIGHTS = {
    "legacy": 0.40,
    "multvae": 0.12,
    "recvae": 0.21,
    "recvae_capacity": 0.09,
    "bm25bpr": 0.18,
}
POOL_WEIGHT = 0.07
META_WEIGHT = 1.65


def qnorm(values):
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean(1, keepdims=True)) / (
        values.std(1, keepdims=True) + 1e-6
    )


def softmax(values):
    values = values - values.max(1, keepdims=True)
    output = np.exp(values)
    return output / output.sum(1, keepdims=True)


def zip_info(name):
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset1", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--multvae", type=Path, required=True)
    parser.add_argument("--recvae", type=Path, required=True)
    parser.add_argument("--recvae-capacity", type=Path, required=True)
    parser.add_argument("--bm25bpr", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset1 = args.dataset1.read_bytes()
    with zipfile.ZipFile(args.legacy) as archive:
        legacy = np.loadtxt(
            io.BytesIO(archive.read("dataset2.csv")), delimiter=",", dtype=np.float32
        )
    arrays = {
        "multvae": np.load(args.multvae, mmap_mode="r"),
        "recvae": np.load(args.recvae, mmap_mode="r"),
        "recvae_capacity": np.load(args.recvae_capacity, mmap_mode="r"),
        "bm25bpr": np.load(args.bm25bpr, mmap_mode="r"),
    }
    logits = WEIGHTS["legacy"] * qnorm(np.log(np.clip(legacy, 1e-12, None)))
    for name, values in arrays.items():
        if values.shape != (153420, 100) or not np.isfinite(values).all():
            raise ValueError(f"invalid {name}: {values.shape}")
        logits += WEIGHTS[name] * qnorm(values)
    with zipfile.ZipFile(args.data) as archive:
        test = pd.read_csv(archive.open("dataset2/test.csv"))
    candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    if candidates.shape != logits.shape:
        raise ValueError(f"invalid candidates: {candidates.shape}")
    count = np.bincount(candidates.ravel(), minlength=int(candidates.max()) + 1)
    logits += POOL_WEIGHT * qnorm(
        np.log1p(count[candidates]).astype(np.float32)
    )
    meta = np.load(args.meta, mmap_mode="r")
    if meta.shape != logits.shape or not np.isfinite(meta).all():
        raise ValueError(f"invalid meta score: {meta.shape}")
    logits += META_WEIGHT * meta
    probability = softmax(logits.astype(np.float64))

    with zipfile.ZipFile(args.output, "w") as archive:
        archive.writestr(zip_info("dataset1.csv"), dataset1)
        archive.writestr(
            zip_info("dataset2.csv"),
            "".join(
                ",".join(f"{value:.8f}" for value in row) + "\n"
                for row in probability
            ),
        )
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ["dataset1.csv", "dataset2.csv"]
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
