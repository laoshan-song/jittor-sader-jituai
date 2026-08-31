#!/usr/bin/env python3
"""Score dataset1 with a Jittor high-order collaborative ensemble."""

import argparse
from pathlib import Path

import numpy as np
import jittor as jt

import run


def qnorm(score):
    return (score - score.mean(1, keepdims=True)) / (
        score.std(1, keepdims=True) + 1e-6
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output-raw", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--prop-alpha", type=float, default=0.05)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is required")
    run.DATA = str(args.data)
    train, test = run.read_scene("dataset1")
    max_node = run.node_max(train, test)
    _, freq, src_freq = run.test_pool_freq(test, max_node)
    edges = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    stats = run.Stats(edges, max_node)

    members = []
    for path in args.models:
        net, mu, sd = run.load_model(str(path))
        feature, use_src_freq = run.feat_fn_for_dim(len(mu))
        if feature is not run.raw_features_numba_v3 or not use_src_freq:
            raise ValueError(f"not a dataset1 high-order checkpoint: {path}")
        state = net.state_dict()
        src_embedding = np.asarray(state["src_emb.weight"].data, np.float32)
        dst_embedding = np.asarray(state["dst_emb.weight"].data, np.float32)
        prop_src, prop_dst = run.make_prop_embeddings(
            edges, max_node + 1, src_embedding, dst_embedding
        )
        members.append(
            (net, mu, sd, (src_embedding, dst_embedding, prop_src, prop_dst))
        )

    output = np.empty((len(test), 100), np.float32)
    for start in range(0, len(test), args.batch):
        end = min(start + args.batch, len(test))
        part = test.iloc[start:end]
        src = part.src.to_numpy(np.int64, copy=False)
        tim = part.time.to_numpy(np.int64, copy=False)
        cand = part.iloc[:, 2:].to_numpy(np.int64, copy=False)
        feature = run.raw_features_numba_v3(
            stats, src, tim, cand, freq, src_freq, True
        )
        hist_ids, hist_mask = run.history_ids(stats, src)
        scores = []
        for net, mu, sd, prop in members:
            scaled = run.scale(feature, mu, sd)
            score = run.model_scores_with_prop(
                net,
                scaled,
                src,
                cand,
                hist_ids,
                hist_mask,
                max(128, args.batch // 4),
                prop,
                args.prop_alpha,
            )
            scores.append(qnorm(score))
        output[start:end] = np.mean(scores, axis=0)
        print(f"wrote scores {end}/{len(test)}", flush=True)

    np.save(args.output_raw, output)
    probability = np.exp(output - output.max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)
    with args.output_csv.open("w") as handle:
        for row in probability:
            handle.write(",".join(f"{value:.8f}" for value in row) + "\n")
    print(f"wrote {args.output_raw} and {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
