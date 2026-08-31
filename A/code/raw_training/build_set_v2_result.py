#!/usr/bin/env python3
"""Replace a scored Dataset2 meta feature with a later Jittor ranker score."""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np


META_WEIGHT = 1.65


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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--old-meta", type=Path, required=True)
    parser.add_argument("--new-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.baseline) as archive:
        dataset1 = archive.read("dataset1.csv")
        probability = np.loadtxt(
            io.BytesIO(archive.read("dataset2.csv")), delimiter=",",
            dtype=np.float64,
        )
    old_meta = np.load(args.old_meta, mmap_mode="r")
    new_meta = np.load(args.new_meta, mmap_mode="r")
    if probability.shape != old_meta.shape or old_meta.shape != new_meta.shape:
        raise ValueError(
            f"shape mismatch: {probability.shape}, {old_meta.shape}, {new_meta.shape}"
        )
    logits = np.log(np.clip(probability, 1e-12, None))
    logits += META_WEIGHT * (new_meta - old_meta)
    candidate = softmax(logits)
    with zipfile.ZipFile(args.output, "w") as archive:
        archive.writestr(zip_info("dataset1.csv"), dataset1)
        archive.writestr(
            zip_info("dataset2.csv"),
            "".join(
                ",".join(f"{value:.8f}" for value in row) + "\n"
                for row in candidate
            ),
        )
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ["dataset1.csv", "dataset2.csv"]
        assert archive.read("dataset1.csv") == dataset1
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
