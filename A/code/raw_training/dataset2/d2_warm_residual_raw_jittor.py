#!/usr/bin/env python3
"""Train the audited warm-target residual Transformer checkpoints."""

import argparse
import json
import zipfile
from pathlib import Path

import jittor as jt
import numpy as np
import pandas as pd
from jittor import nn

import d2_multislice_transformer_raw_jittor as transformer
import d2_pool_ranker_raw_jittor as rank
import d2_temporal_pool_raw_jittor as pool
import d2_v2_set_incremental_audit_jittor as set_v2


MEMBERS = (
    (64, 20265701, 7), (64, 20265702, 7), (64, 20265703, 7),
    (96, 20265801, 8), (96, 20265802, 7), (96, 20265803, 7),
)
META_WEIGHT = 1.65


def normalize(score):
    centered = score - score.mean(dim=1, keepdims=True)
    return centered / jt.sqrt(
        (centered * centered).mean(dim=1, keepdims=True) + 1e-6
    )


def train(feature, current, labels, warm, hidden, epochs, batch, seed):
    np.random.seed(seed)
    jt.set_global_seed(seed)
    rng = np.random.default_rng(seed)
    net = transformer.TransformerSetRanker(28, hidden, transformer.LAYERS)
    optimizer = jt.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(labels))
        losses = []
        for start in range(0, len(order), batch):
            ids = order[start:start + batch]
            score = jt.array(current[ids]) + META_WEIGHT * normalize(
                net(jt.array(feature[ids]))
            )
            row_loss = nn.cross_entropy_loss(
                score, jt.array(labels[ids]), reduction="none"
            )
            mask = jt.array(warm[ids].astype(np.float32, copy=False))
            loss = (row_loss * mask).sum() / (mask.sum() + 1e-6)
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.data).item()))
        print(
            f"warm_residual set{hidden} seed={seed} epoch={epoch + 1} "
            f"loss={np.mean(losses):.6f}", flush=True,
        )
    return net


def load_training_data(args, edges, test, candidates):
    if args.cache_dir:
        feature, current, labels, warm = [], [], [], []
        for name in ("y2009", "strict"):
            value_labels = np.load(
                args.cache_dir / f"{name}_labels.npy", mmap_mode="r"
            )
            cold = np.load(
                args.cache_dir / f"{name}_cold_mask.npy", mmap_mode="r"
            )
            feature.append(np.load(
                args.cache_dir / f"{name}_feature_v2.npy", mmap_mode="r"
            ))
            current.append(np.load(
                args.cache_dir / f"{name}_current.npy", mmap_mode="r"
            ))
            labels.append(value_labels)
            warm.append(~cold[np.arange(len(value_labels)), value_labels])
        return tuple(map(np.concatenate, (feature, current, labels, warm))), {}

    replay_pool = pool.build_replay_pool(candidates, edges[:, 1])
    feature, current, labels, warm, audits = [], [], [], [], {}
    for offset, name in enumerate(("y2009", "strict")):
        bounds = pool.SLICES[name]
        values = rank.build_slice(
            edges, test, args.slice_models, name, bounds, 153420,
            args.batch, args.candidate_seed + offset, replay_pool, "test-pool",
        )
        set_v2.extend(values, edges, int(candidates.max()))
        known = np.unique(edges[edges[:, 2] < bounds[0], 1])
        feature.append(values["feature_v2"])
        current.append(values["current"])
        labels.append(values["labels"])
        warm.append(np.isin(values["target"][:, 1], known))
        audits[name] = values["candidate_audit"]
    return tuple(map(np.concatenate, (feature, current, labels, warm))), audits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--slice-models", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--candidate-seed", type=int, default=20260781)
    args = parser.parse_args()
    if args.cache_dir is None and args.slice_models is None:
        parser.error("--cache-dir or --slice-models is required")

    jt.flags.use_cuda = 1
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.data) as archive:
        train_frame = pd.read_csv(
            archive.open("dataset2/train.csv"), usecols=["src", "dst", "time"]
        )
        test = pd.read_csv(archive.open("dataset2/test.csv"))
    edges = train_frame[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    (feature, current, labels, warm), audits = load_training_data(
        args, edges, test, candidates
    )
    paths = []
    for hidden, seed, epochs in MEMBERS:
        net = train(feature, current, labels, warm, hidden, epochs, args.batch, seed)
        path = args.output_dir / (
            f"warm_residual_set{hidden}_seed{seed}_jittor.npz"
        )
        transformer.save_checkpoint(path, net, hidden, seed, epochs)
        reloaded = transformer.load_checkpoint(path)
        before = set_v2.predict_set(net, feature[:1024], args.batch)
        after = set_v2.predict_set(reloaded, feature[:1024], args.batch)
        if not np.allclose(before, after, rtol=0, atol=1e-6):
            raise RuntimeError(f"checkpoint reload mismatch: {path}")
        paths.append(str(path))
        print(f"verified {path}", flush=True)
    diagnostics = {
        "kind": "jittor_warm_residual_transformer2",
        "training_rows": int(len(labels)),
        "warm_rows": int(warm.sum()),
        "candidate_audit": audits,
        "checkpoints": paths,
    }
    (args.output_dir / "warm_residual_training.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
