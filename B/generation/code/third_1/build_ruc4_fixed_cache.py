#!/usr/bin/env python3
"""Build fixed-cutoff ruc4 replay control from fixed ruc2 cache."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from numba import set_num_threads


STRATEGIES = ("history", "test_pool")
ROWS = {"validation": 120_000, "confirmation": 30_000}


def rr(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positive = scores[np.arange(len(labels)), labels]
    columns = np.arange(scores.shape[1])[None, :]
    rank = 1 + np.sum(scores > positive[:, None], axis=1)
    rank += np.sum((scores == positive[:, None]) & (columns < labels[:, None]), axis=1)
    return 1.0 / rank


def duplicate_mask(candidates: np.ndarray, chunk: int = 8192) -> np.ndarray:
    out = np.empty(len(candidates), dtype=bool)
    for start in range(0, len(candidates), chunk):
        stop = min(len(candidates), start + chunk)
        out[start:stop] = np.any(
            np.diff(np.sort(candidates[start:stop], axis=1), axis=1) == 0,
            axis=1,
        )
    return out


def load_replay(replay_root: Path, strategy: str, split: str, name: str) -> np.ndarray:
    return np.load(replay_root / strategy / f"{strategy}__{split}__{name}.npy", mmap_mode="r")


def load_identity(identity_cache: Path, strategy: str, split: str, name: str) -> np.ndarray:
    return np.load(identity_cache / f"{strategy}__{split}__{name}.npy", mmap_mode="r")


def paired(control: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    before = rr(control, labels)
    after = rr(candidate, labels)
    delta = after - before
    return {
        "control_mrr": float(np.mean(before)),
        "candidate_mrr": float(np.mean(after)),
        "delta": float(delta.mean()),
        "delta_se": float(delta.std(ddof=1) / math.sqrt(len(delta))),
        "positive_row_rate": float(np.mean(delta > 0)),
        "negative_row_rate": float(np.mean(delta < 0)),
        "top1_changed_rate": float(np.mean(np.argmax(control, axis=1) != np.argmax(candidate, axis=1))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rp3-code", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--identity-cache", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--validation-cutoff", type=int, default=1512131937)
    parser.add_argument("--confirmation-cutoff", type=int, default=1512135046)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--kind-index", type=int, default=1)
    parser.add_argument("--max-candidate-degree", type=float, default=4096.0)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    if args.output_cache.exists():
        raise FileExistsError(args.output_cache)
    sys.path.insert(0, str(args.rp3_code))
    import d4_rp3beta_gate as rp3

    set_num_threads(args.threads)
    rp3.set_num_threads(args.threads)
    args.output_cache.mkdir(parents=True)

    raw_source = np.load(args.data_cache / "src.npy", mmap_mode="r")
    raw_item = np.load(args.data_cache / "dst.npy", mmap_mode="r")
    raw_time = np.load(args.data_cache / "time.npy", mmap_mode="r")
    report: dict[str, object] = {
        "kind": "ruc4_fixed_cutoff_replay_control_v1",
        "source_cache": str(args.input_cache),
        "alpha": args.alpha,
        "beta": args.beta,
        "kind_index": args.kind_index,
        "max_candidate_degree": args.max_candidate_degree,
        "external_data_used": False,
        "entries": {},
        "mrr": {},
    }

    for split, cutoff in (
        ("validation", args.validation_cutoff),
        ("confirmation", args.confirmation_cutoff),
    ):
        matrix, reverse, source_base, events = rp3.graph(raw_source, raw_item, raw_time, cutoff)
        print(json.dumps({"split": split, "cutoff": cutoff, "events": events, "unique_edges": len(matrix.indices)}), flush=True)
        for strategy in STRATEGIES:
            src = load_identity(args.identity_cache, strategy, split, "src")
            candidates = load_identity(args.identity_cache, strategy, split, "candidates")
            baseline = np.load(args.input_cache / f"{strategy}__{split}.npy", mmap_mode="r")
            labels = load_replay(args.replay_root, strategy, split, "labels")
            seen = load_replay(args.replay_root, strategy, split, "seen")
            if baseline.shape != (ROWS[split], 100):
                raise ValueError(f"bad baseline shape for {strategy} {split}: {baseline.shape}")
            raw, degree = rp3.score_context(
                matrix, reverse, source_base, src, candidates, args.max_candidate_degree
            )
            residual = rp3.feature(raw, degree, args.kind_index, args.beta)
            duplicate = duplicate_mask(candidates)
            residual[duplicate] = 0.0
            candidate = rp3.candidate_score(np.asarray(baseline), residual, np.asarray(seen), args.alpha)
            path = args.output_cache / f"{strategy}__{split}.npy"
            np.save(path, candidate.astype(np.float32), allow_pickle=False)
            key = f"{strategy}__{split}"
            tail = None
            if split == "validation":
                tail = paired(np.asarray(baseline[-50_000:]), candidate[-50_000:], np.asarray(labels[-50_000:]))
            report["entries"][key] = {
                "file": path.name,
                "shape": list(candidate.shape),
                "dtype": "float32",
                "duplicate_rows": int(duplicate.sum()),
            }
            report["mrr"][key] = {
                "full": paired(np.asarray(baseline), candidate, np.asarray(labels)),
                "tail_50000": tail,
            }
            print(json.dumps({"built": key, **report["mrr"][key]["full"]}, sort_keys=True), flush=True)
            del raw, degree, residual, candidate
            gc.collect()
        del matrix, reverse
        gc.collect()

    report["decision"] = "PASS"
    tmp = args.output_cache / "manifest.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output_cache / "manifest.json")
    print(json.dumps(report["mrr"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
