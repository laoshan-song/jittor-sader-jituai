#!/usr/bin/env python3
"""Build hierarchy and neighbor meta features from official training edges."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from infer_meta_ruc4 import (
    NEIGHBOR_NAMES,
    RECENT_LEVELS,
    RECENT_WINDOWS,
    FULL_LEVELS,
    hierarchy_features,
    neighbor_features,
    prepare_hierarchy_map_cache,
    prepare_neighbor_cache,
)


STRATEGIES = ("history", "test_pool")
SPLITS = ("validation", "confirmation")


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def feature_names() -> list[str]:
    names = []
    for src_shift, dst_shift in FULL_LEVELS:
        names += [f"hier_full_log_s{src_shift}_d{dst_shift}", f"hier_full_pmi_s{src_shift}_d{dst_shift}"]
    for window in RECENT_WINDOWS:
        for src_shift, dst_shift in RECENT_LEVELS:
            names += [
                f"hier_recent{window}_log_s{src_shift}_d{dst_shift}",
                f"hier_recent{window}_pmi_s{src_shift}_d{dst_shift}",
            ]
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--identity-cache", type=Path, required=True)
    parser.add_argument("--hierarchy-output", type=Path, required=True)
    parser.add_argument("--neighbor-output", type=Path, required=True)
    parser.add_argument("--map-cache", type=Path, required=True)
    parser.add_argument("--validation-cutoff", type=int, default=1512131937)
    parser.add_argument("--confirmation-cutoff", type=int, default=1512135046)
    args = parser.parse_args()
    if args.hierarchy_output.exists() or args.neighbor_output.exists():
        raise FileExistsError("refusing to overwrite meta side feature cache")

    source = np.load(args.train_cache / "src.npy", mmap_mode="r")
    destination = np.load(args.train_cache / "dst.npy", mmap_mode="r")
    timestamp = np.load(args.train_cache / "time.npy", mmap_mode="r")
    maximum_source = int(source.max())
    maximum_candidate = int(destination.max())

    args.hierarchy_output.mkdir(parents=True)
    args.neighbor_output.mkdir(parents=True)
    entries = {}
    cutoffs = {"validation": args.validation_cutoff, "confirmation": args.confirmation_cutoff}
    for split in SPLITS:
        cutoff = int(cutoffs[split])
        h_maps = prepare_hierarchy_map_cache(
            args.map_cache / f"hierarchy_{split}",
            source,
            destination,
            timestamp,
            cutoff,
            maximum_candidate,
        )
        neighbor = prepare_neighbor_cache(
            args.map_cache / f"neighbor_{split}",
            source,
            destination,
            timestamp,
            cutoff,
            maximum_source,
            maximum_candidate,
            max_destination_neighbors=32,
        )
        for strategy in STRATEGIES:
            prefix = f"{strategy}__{split}"
            src = np.load(args.identity_cache / f"{prefix}__src.npy", mmap_mode="r")
            candidates = np.load(args.identity_cache / f"{prefix}__candidates.npy", mmap_mode="r")
            hierarchy = hierarchy_features(h_maps, src, candidates, maximum_candidate)
            neighbor_values = neighbor_features(neighbor, src, candidates)
            np.save(args.hierarchy_output / f"{prefix}.npy", hierarchy, allow_pickle=False)
            np.save(args.neighbor_output / f"{prefix}.npy", neighbor_values, allow_pickle=False)
            entries[prefix] = {
                "hierarchy_shape": list(hierarchy.shape),
                "neighbor_shape": list(neighbor_values.shape),
                "cutoff": cutoff,
            }
            print(json.dumps({"built": prefix, **entries[prefix]}), flush=True)

    save_json(
        args.hierarchy_output / "metadata.json",
        {
            "kind": "third_1_meta_hierarchy_features_v1",
            "feature_count": len(feature_names()),
            "feature_names": feature_names(),
            "entries": entries,
            "external_data_used": False,
        },
    )
    save_json(
        args.neighbor_output / "manifest.json",
        {
            "kind": "third_1_meta_neighbor_features_v1",
            "feature_count": len(NEIGHBOR_NAMES),
            "feature_names": list(NEIGHBOR_NAMES),
            "entries": entries,
            "external_data_used": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
