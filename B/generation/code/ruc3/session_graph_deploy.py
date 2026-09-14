#!/usr/bin/env python3
"""Apply the audited Jittor session-graph ensemble to one frozen D4 shard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.update({
    "use_cutt": "0",
    "use_cutlass": "0",
    "use_nccl": "0",
    "use_mkl": "0",
})

import jittor as jt
import numpy as np

from b_rank import data_features, pairnew_transformer_jittor as pairnew

import session_graph_gate as graph_gate
import session_graph_hard_ranker as ranker


ROWS = 2_322_538


def load_model(path: Path) -> ranker.HardNegativeGate:
    with np.load(path, allow_pickle=False) as saved:
        net = ranker.HardNegativeGate(20)
        net.load_state_dict({
            str(name): jt.array(saved[f"state_{index}"])
            for index, name in enumerate(saved["state_names"])
        })
    return net


def predict(models: list[ranker.HardNegativeGate], feature: np.ndarray, batch: int) -> np.ndarray:
    return ranker.qnorm(np.mean([
        ranker.predict(model, feature, batch) for model in models
    ], axis=0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    start = ROWS * args.shard_index // args.shard_count
    shard_stop = ROWS * (args.shard_index + 1) // args.shard_count
    probabilities = np.loadtxt(args.input, delimiter=",", dtype=np.float32)
    if probabilities.shape != (shard_stop - start, 100):
        raise ValueError(f"baseline shard shape differs: {probabilities.shape}")
    stop = (
        min(shard_stop, start + args.max_rows)
        if args.max_rows is not None else shard_stop
    )

    jt.flags.use_cuda = 1
    models = [load_model(path) for path in args.model]
    store = data_features.FeatureStore(args.feature_store)
    source = np.load(args.train_cache / "src.npy", mmap_mode="r")
    item = np.load(args.train_cache / "dst.npy", mmap_mode="r")
    timestamp = np.load(args.train_cache / "time.npy", mmap_mode="r")
    graph = graph_gate.build_graph(source, item, timestamp, int(store.cutoff))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    top1_changed = 0
    active_rows = 0
    duplicate_rows = 0
    pair_seen_rank_errors = 0
    with args.output.open("x", encoding="ascii", newline="\n") as output:
        for chunk in data_features.iter_test_chunks(
            args.data, "dataset4", chunk_rows=args.chunk
        ):
            if chunk.row_stop <= start:
                continue
            if chunk.row_start >= stop:
                break
            left = max(start, chunk.row_start) - chunk.row_start
            right = min(stop, chunk.row_stop) - chunk.row_start
            src = chunk.src[left:right]
            query_time = chunk.time[left:right]
            candidates = chunk.candidates[left:right]
            count = len(src)
            baseline_probability = probabilities[rows_written:rows_written + count]
            static = store.features(src, query_time, candidates)
            seen = static[:, :, 2] > 0.0
            graph_feature, active = graph_gate.graph_features(
                graph, src, query_time, candidates, args.chunk
            )
            baseline = (
                1.0 - graph_gate.rank_positions(baseline_probability).astype(np.float32) / 99.0
            )
            feature = ranker.feature_plane(graph_feature, baseline, static, seen)
            residual = predict(models, feature, args.batch)
            candidate = pairnew._candidate_score(baseline, residual, seen, args.alpha)
            duplicate = np.any(
                np.diff(np.sort(candidates, axis=1), axis=1) == 0, axis=1
            )
            candidate[duplicate] = baseline[duplicate]

            before_rank = graph_gate.rank_positions(baseline)
            after_rank = graph_gate.rank_positions(candidate)
            pair_seen_rank_errors += int(
                np.count_nonzero(before_rank[seen] != after_rank[seen])
            )
            top1_changed += int(np.count_nonzero(
                np.argmax(baseline, axis=1) != np.argmax(candidate, axis=1)
            ))
            active_rows += int(active.sum())
            duplicate_rows += int(duplicate.sum())

            order = np.argsort(-candidate, axis=1, kind="stable")
            sorted_probability = np.sort(baseline_probability, axis=1)[:, ::-1]
            result = np.empty_like(baseline_probability)
            batch_rows = np.arange(count)[:, None]
            result[batch_rows, order] = sorted_probability
            result[duplicate] = baseline_probability[duplicate]
            np.savetxt(output, result, fmt="%.8f", delimiter=",", newline="\n")
            rows_written += count
            if rows_written % 100_000 < count:
                print(json.dumps({
                    "shard": args.shard_index,
                    "rows": rows_written,
                    "total": stop - start,
                }), flush=True)

    if rows_written != stop - start or pair_seen_rank_errors:
        raise RuntimeError(
            f"deployment invariant failed: rows={rows_written}, seen_errors={pair_seen_rank_errors}"
        )
    report = {
        "kind": "d4_session_graph_ruc2_shard_v1",
        "decision": "PASS",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": start,
        "row_stop": stop,
        "rows": rows_written,
        "alpha": args.alpha,
        "model_count": len(models),
        "active_rows": active_rows,
        "duplicate_rows_preserved": duplicate_rows,
        "top1_changed_rows": top1_changed,
        "top1_changed_rate": top1_changed / rows_written,
        "pair_seen_rank_errors": pair_seen_rank_errors,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
