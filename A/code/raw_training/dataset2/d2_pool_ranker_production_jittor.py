#!/usr/bin/env python3
"""Fit the audited Jittor pool ranker and score official dataset2 candidates."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt
from jittor import nn

import d2_pool_ranker_eval_jittor as rank
import d2_temporal_pool_eval_jittor as pool


SEEDS_EPOCHS = ((20260816, 8), (20260817, 7), (20260818, 8))


def train_full(feature, labels, seed, epochs, batch):
    np.random.seed(seed)
    jt.set_global_seed(seed)
    rng = np.random.default_rng(seed)
    net = rank.Ranker(feature.shape[-1], 32)
    optimizer = jt.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(labels))
        losses = []
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            block = feature[ids]
            score = net(jt.array(block.reshape(-1, block.shape[-1])))
            score = score.reshape((len(ids), 100))
            loss = nn.cross_entropy_loss(score, jt.array(labels[ids]))
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.data).item()))
        print(
            f"seed={seed} epoch={epoch + 1} loss={np.mean(losses):.6f}",
            flush=True,
        )
    return net


def save_checkpoint(path, net, seed, epochs):
    state = {
        name: np.asarray(value.data).copy()
        for name, value in net.state_dict().items()
    }
    names = list(state)
    payload = {f"state_{i}": state[name] for i, name in enumerate(names)}
    payload.update(
        state_names=np.asarray(names),
        seed=np.asarray(seed),
        epochs=np.asarray(epochs),
        hidden=np.asarray(32),
        feature_count=np.asarray(13),
        kind=np.asarray("jittor_pool_meta_ranker_v1"),
    )
    np.savez(path, **payload)


def load_checkpoint(path):
    saved = np.load(path, allow_pickle=False)
    net = rank.Ranker(int(saved["feature_count"]), int(saved["hidden"]))
    state = {
        str(name): jt.array(saved[f"state_{i}"])
        for i, name in enumerate(saved["state_names"])
    }
    net.load_state_dict(state)
    return net


def build_test(edges, test, models, batch):
    history = edges[edges[:, 2] < int(test.time.min())]
    src = test.src.to_numpy(np.int64, copy=False)
    candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    old_keys = np.unique(history[:, 0] * pool.KEY_BASE + history[:, 1])
    scores = {
        name: pool.score_model(
            models / f"model_{name}_prod_jittor.npz",
            src,
            candidates,
            old_keys,
            batch,
        )
        for name in ("multvae", "recvae", "bm25bpr")
    }
    capacity = max(int(edges[:, 1].max()), int(candidates.max()))
    global_set = pool.global_signals(candidates, capacity, int(candidates.max()))
    target = np.zeros((len(test), 3), np.int64)
    target[:, 0] = src
    target[:, 2] = test.time.to_numpy(np.int64, copy=False)
    time60 = 0.5 * (
        pool.temporal_signal(target, candidates, capacity, 60, 0)
        + pool.temporal_signal(target, candidates, capacity, 60, 30)
    )
    current = (
        0.15 * scores["multvae"]
        + 0.45 * scores["recvae"]
        + 0.40 * scores["bm25bpr"]
        + 0.07 * global_set["global_log"]
    )
    feature = np.stack(
        [
            scores["multvae"],
            scores["recvae"],
            scores["bm25bpr"],
            pool.qnorm(current),
            global_set["global_log"],
            global_set["global_linear"],
            time60,
            rank.source_count_signal(src, candidates),
            *rank.item_signals(history, candidates, int(test.time.min())),
        ],
        axis=2,
    ).astype(np.float32)
    return feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=3)
    parser.add_argument("--targets", type=int, default=153420)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--candidate-seed", type=int, default=20260717)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.data) as archive:
        train = pd.read_csv(
            archive.open("dataset2/train.csv"), usecols=["src", "dst", "time"]
        )
        test = pd.read_csv(archive.open("dataset2/test.csv"))
    edges = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    nets = []
    checkpoints = []
    if args.checkpoints:
        for path in args.checkpoints:
            nets.append(load_checkpoint(path))
            checkpoints.append(str(path))
    else:
        strict = rank.build_slice(
            edges,
            test,
            args.models,
            "strict",
            pool.SLICES["strict"],
            args.targets,
            args.batch,
            args.candidate_seed,
        )
        for seed, epochs in SEEDS_EPOCHS:
            net = train_full(
                strict["feature"], strict["labels"], seed, epochs, args.batch
            )
            path = args.output_dir / f"pool_ranker_seed{seed}_jittor.npz"
            save_checkpoint(path, net, seed, epochs)
            reloaded = load_checkpoint(path)
            before = rank.predict(net, strict["feature"][:1024], args.batch)
            after = rank.predict(reloaded, strict["feature"][:1024], args.batch)
            if not np.allclose(before, after, rtol=0, atol=1e-6):
                raise RuntimeError(f"checkpoint reload mismatch: {path}")
            nets.append(reloaded)
            checkpoints.append(str(path))
            print(f"verified {path}", flush=True)
        del strict

    feature = build_test(edges, test, args.models, args.batch)
    predictions = [rank.predict(net, feature, args.batch) for net in nets]
    # Quantization removes sub-1e-6 CUDA reduction jitter so checkpoint-only
    # inference reproduces byte-identical submissions across fresh processes.
    score = np.round(pool.qnorm(np.mean(predictions, axis=0)), 2).astype(np.float32)
    output = args.output_dir / "official_pool_ranker.npy"
    np.save(output, score)
    top1 = [np.argmax(value, axis=1) for value in predictions]
    diagnostics = {
        "shape": list(score.shape),
        "finite": bool(np.isfinite(score).all()),
        "mean": float(score.mean()),
        "std": float(score.std()),
        "seed_top1_agreement": [
            float(np.mean(top1[0] == top1[i])) for i in range(1, len(top1))
        ],
        "checkpoints": checkpoints,
    }
    (args.output_dir / "production_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
