#!/usr/bin/env python3
"""Persist replay source/candidate identities aligned to the score-cache rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from b_rank import data_features, replay_score_cache


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--data-cache", type=Path, required=True)
    result.add_argument("--score-cache", type=Path, action="append", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, default=20260810)
    result.add_argument("--valid-groups", type=int, default=120000)
    result.add_argument("--confirm-groups", type=int, default=30000)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing output reuse: {args.output}")
    scored, _, _ = replay_score_cache.load(args.score_cache)
    cache = data_features.BDataCache.build_or_open(
        args.data.resolve(), "dataset4", args.data_cache.resolve(), verify_hash=True
    )
    sizes = {"train": 1, "valid": args.valid_groups, "confirm": args.confirm_groups}
    args.output.mkdir(parents=True)
    records = {}
    for strategy in ("history", "test_pool"):
        groups = data_features.build_split1_groups(
            cache, seed=args.seed, sizes=sizes, batch_rows=4096,
            negative_strategy=strategy
        )
        for split, group in (("validation", groups.valid), ("confirmation", groups.confirm)):
            key = (strategy, split)
            labels = np.asarray(scored[key][1])
            if not np.array_equal(labels, np.asarray(group.labels)):
                raise ValueError(f"identity labels disagree for {key}")
            prefix = f"{strategy}__{split}"
            arrays = {
                "src": np.asarray(group.src, dtype=np.uint64),
                "time": np.asarray(group.time, dtype=np.int64),
                "candidates": np.asarray(group.candidates, dtype=np.uint32),
            }
            records[prefix] = {}
            for name, values in arrays.items():
                path = args.output / f"{prefix}__{name}.npy"
                np.save(path, values, allow_pickle=False)
                records[prefix][name] = {
                    "file": path.name,
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                }
            print(f"saved {prefix} rows={len(labels)}", flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps({"kind": "d4_replay_identity_cache_v33", "entries": records}, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
