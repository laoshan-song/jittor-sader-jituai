#!/usr/bin/env python3
"""Build compact meta features for the existing Jittor hierarchy trainer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


STRATEGIES = ("history", "test_pool")
SPLITS = ("validation", "confirmation")


def qnorm2(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    return centered / np.maximum(scale, 1e-6)


def append_feature(blocks: list[np.ndarray], values: np.ndarray) -> None:
    if values.ndim == 2:
        blocks.append(values[:, :, None].astype(np.float32, copy=False))
    elif values.ndim == 3:
        blocks.append(values.astype(np.float32, copy=False))
    else:
        raise ValueError(values.shape)


def normalized_cube(values: np.ndarray) -> np.ndarray:
    output = np.empty(values.shape, dtype=np.float32)
    for index in range(values.shape[2]):
        output[:, :, index] = qnorm2(np.asarray(values[:, :, index], dtype=np.float32))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-cache", type=Path, required=True)
    parser.add_argument("--neighbor-cache", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    hierarchy_meta = json.loads((args.hierarchy_cache / "metadata.json").read_text())
    neighbor_meta = json.loads((args.neighbor_cache / "manifest.json").read_text())
    feature_names = list(hierarchy_meta["feature_names"])
    feature_names += [f"component_{i:02d}" for i in range(23)]
    feature_names += [f"static_{i:02d}" for i in range(11)]
    feature_names += ["ruc4_rank_slot", "pair_seen"]
    feature_names += [f"neighbor_{name}" for name in neighbor_meta["feature_names"]]

    entries = {}
    for strategy in STRATEGIES:
        root = args.replay_root / strategy
        for split in SPLITS:
            prefix = f"{strategy}__{split}"
            hierarchy = np.load(args.hierarchy_cache / f"{prefix}.npy", mmap_mode="r")
            baseline = np.load(args.baseline_cache / f"{prefix}.npy", mmap_mode="r")
            seen = np.load(root / f"{prefix}__seen.npy", mmap_mode="r")
            scores = np.load(root / f"{prefix}__scores.npy", mmap_mode="r")
            static = np.load(root / f"{prefix}__static.npy", mmap_mode="r")
            neighbor = np.load(args.neighbor_cache / f"{prefix}.npy", mmap_mode="r")

            blocks: list[np.ndarray] = [np.asarray(hierarchy, dtype=np.float32)]
            append_feature(blocks, normalized_cube(np.moveaxis(np.asarray(scores), 0, 2)))
            append_feature(blocks, normalized_cube(np.asarray(static)))
            append_feature(blocks, np.asarray(baseline, dtype=np.float32))
            append_feature(blocks, np.asarray(seen, dtype=np.float32))
            append_feature(blocks, normalized_cube(np.asarray(neighbor)))
            output = np.concatenate(blocks, axis=2).astype(np.float32, copy=False)
            path = args.output / f"{prefix}.npy"
            np.save(path, output, allow_pickle=False)
            entries[prefix] = {"file": path.name, "shape": list(output.shape), "dtype": "float32"}
            print(json.dumps({"built": prefix, "shape": list(output.shape)}), flush=True)
            del hierarchy, baseline, seen, scores, static, neighbor, blocks, output

    metadata = {
        "kind": "ruc4_meta_rank_features_v1",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "entries": entries,
        "external_data_used": False,
    }
    tmp = args.output / "metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output / "metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
