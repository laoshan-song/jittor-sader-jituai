#!/usr/bin/env python3
"""Train the audited multi-slice Jittor Transformer and score test."""

import argparse
import json
import zipfile
from pathlib import Path

import jittor as jt
import numpy as np
import pandas as pd
from jittor import nn

import d2_pool_ranker_eval_jittor as rank
import d2_temporal_pool_eval_jittor as pool
import d2_v2_set_incremental_audit_jittor as set_v2
import d2_v2_set_production_jittor as production


MEMBERS = (
    (64, 20265701, 8), (64, 20265702, 6), (64, 20265703, 7),
    (96, 20265801, 8), (96, 20265802, 8), (96, 20265803, 7),
)
LAYERS = 2
GROUP64_WEIGHT = 0.275
WARM_RESIDUAL_WEIGHT = 0.525


class TransformerBlock(nn.Module):
    def __init__(self, hidden):
        self.attention = jt.attention.MultiheadAttention(
            hidden, 4, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.Relu(),
            nn.Linear(2 * hidden, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def execute(self, values):
        context, _ = self.attention(
            values, values, values, need_weights=False
        )
        values = self.norm1(values + context)
        return self.norm2(values + self.feedforward(values))


class TransformerSetRanker(nn.Module):
    def __init__(self, dim, hidden, layers):
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.Relu(),
            nn.Linear(hidden, hidden), nn.Relu(),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden) for _ in range(layers)
        ])
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Relu(), nn.Linear(hidden, 1)
        )

    def execute(self, values):
        values = self.encoder(values)
        for block in self.blocks:
            values = block(values)
        return self.output(values).squeeze(-1)


def train_full(feature, labels, hidden, epochs, batch, seed):
    np.random.seed(seed)
    jt.set_global_seed(seed)
    rng = np.random.default_rng(seed)
    net = TransformerSetRanker(feature.shape[-1], hidden, LAYERS)
    optimizer = jt.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(labels))
        losses = []
        for start in range(0, len(order), batch):
            ids = order[start:start + batch]
            score = net(jt.array(feature[ids]))
            loss = nn.cross_entropy_loss(score, jt.array(labels[ids]))
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.data).item()))
        print(
            f"transformer{LAYERS} set{hidden} seed={seed} epoch={epoch + 1} "
            f"loss={np.mean(losses):.6f}", flush=True,
        )
    return net


def save_checkpoint(path, net, hidden, seed, epochs):
    state = {
        name: np.asarray(value.data).copy()
        for name, value in net.state_dict().items()
    }
    names = list(state)
    payload = {f"state_{index}": state[name] for index, name in enumerate(names)}
    payload.update(
        state_names=np.asarray(names), hidden=np.asarray(hidden),
        feature_count=np.asarray(28), layers=np.asarray(LAYERS),
        seed=np.asarray(seed), epochs=np.asarray(epochs),
        kind=np.asarray("jittor_transformer2_pool_meta_multislice"),
    )
    np.savez(path, **payload)


def load_checkpoint(path):
    saved = np.load(path, allow_pickle=False)
    if int(saved["feature_count"]) != 28 or int(saved["layers"]) != LAYERS:
        raise ValueError(f"architecture mismatch in {path}")
    net = TransformerSetRanker(
        int(saved["feature_count"]), int(saved["hidden"]), int(saved["layers"])
    )
    net.load_state_dict({
        str(name): jt.array(saved[f"state_{index}"])
        for index, name in enumerate(saved["state_names"])
    })
    return net


def load_training_data(args, edges, test, candidates):
    if args.cache_dir:
        features = [
            np.load(args.cache_dir / f"{name}_feature_v2.npy", mmap_mode="r")
            for name in ("y2009", "strict")
        ]
        labels = [
            np.load(args.cache_dir / f"{name}_labels.npy", mmap_mode="r")
            for name in ("y2009", "strict")
        ]
        return np.concatenate(features), np.concatenate(labels), {}

    replay_pool = pool.build_replay_pool(candidates, edges[:, 1])
    features, labels, audits = [], [], {}
    for offset, name in enumerate(("y2009", "strict")):
        values = rank.build_slice(
            edges, test, args.slice_models, name, pool.SLICES[name], 153420,
            args.batch, args.candidate_seed + offset, replay_pool, "test-pool",
        )
        set_v2.extend(values, edges, int(candidates.max()))
        features.append(values["feature_v2"])
        labels.append(values["labels"])
        audits[name] = values["candidate_audit"]
    return np.concatenate(features), np.concatenate(labels), audits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--slice-models", type=Path)
    parser.add_argument("--prod-models", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=6)
    parser.add_argument("--warm-checkpoints", type=Path, nargs=6)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--candidate-seed", type=int, default=20260781)
    parser.add_argument("--cpu-inference", action="store_true")
    args = parser.parse_args()
    if not args.checkpoints and args.cache_dir is None and args.slice_models is None:
        parser.error("--cache-dir or --slice-models is required when training")

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
    candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    audits = {}
    if args.checkpoints:
        checkpoint_paths = list(args.checkpoints)
    else:
        feature, labels, audits = load_training_data(
            args, edges, test, candidates
        )
        checkpoint_paths = []
        for hidden, seed, epochs in MEMBERS:
            net = train_full(feature, labels, hidden, epochs, args.batch, seed)
            path = args.output_dir / (
                f"multi_transformer2_set{hidden}_seed{seed}_jittor.npz"
            )
            save_checkpoint(path, net, hidden, seed, epochs)
            reloaded = load_checkpoint(path)
            before = set_v2.predict_set(net, feature[:1024], args.batch)
            after = set_v2.predict_set(reloaded, feature[:1024], args.batch)
            if not np.allclose(before, after, rtol=0, atol=1e-6):
                raise RuntimeError(f"checkpoint reload mismatch: {path}")
            checkpoint_paths.append(path)
            print(f"verified {path}", flush=True)
        del feature, labels

    feature = production.official_features(
        edges, test, args.prod_models, args.batch
    )
    jt.sync_all()
    if args.cpu_inference:
        jt.flags.use_cuda = 0
    nets = [load_checkpoint(path) for path in checkpoint_paths]
    predictions = [
        set_v2.predict_set(net, feature, args.batch) for net in nets
    ]
    set64 = pool.qnorm(np.mean(predictions[:3], axis=0))
    set96 = pool.qnorm(np.mean(predictions[3:], axis=0))
    baseline = pool.qnorm(0.5 * (set64 + set96))
    if args.warm_checkpoints:
        warm_nets = [load_checkpoint(path) for path in args.warm_checkpoints]
        warm_predictions = [
            set_v2.predict_set(net, feature, args.batch) for net in warm_nets
        ]
        warm64 = pool.qnorm(np.mean(warm_predictions[:3], axis=0))
        warm96 = pool.qnorm(np.mean(warm_predictions[3:], axis=0))
        warm_residual = pool.qnorm(0.5 * (warm64 + warm96))
        warm_candidates = np.isin(candidates, np.unique(edges[:, 1]))
        set_all = pool.qnorm(
            GROUP64_WEIGHT * set64 + (1.0 - GROUP64_WEIGHT) * set96
            + WARM_RESIDUAL_WEIGHT * (warm_residual - baseline)
            * warm_candidates
        )
    else:
        set_all = baseline
    for name, value in {
        "official_multislice_transformer2_set64.npy": set64,
        "official_multislice_transformer2_set96.npy": set96,
        "official_multislice_transformer2_set_all.npy": set_all,
    }.items():
        np.save(args.output_dir / name, np.round(value, 2).astype(np.float32))
    top1 = [np.argmax(value, axis=1) for value in predictions]
    diagnostics = {
        "kind": "jittor_transformer2_pool_meta_multislice",
        "training_rows": 306840,
        "training_slices": ["y2009", "strict"],
        "checkpoint_only": bool(args.checkpoints),
        "candidate_audit": audits,
        "checkpoints": [str(path) for path in checkpoint_paths],
        "warm_checkpoints": [
            str(path) for path in (args.warm_checkpoints or [])
        ],
        "group64_weight": GROUP64_WEIGHT if args.warm_checkpoints else 0.5,
        "warm_residual_weight": (
            WARM_RESIDUAL_WEIGHT if args.warm_checkpoints else 0.0
        ),
        "shape": list(set_all.shape),
        "finite": bool(np.isfinite(set_all).all()),
        "member_top1_agreement_with_first": [
            float(np.mean(top1[0] == value)) for value in top1[1:]
        ],
        "set64_set96_top1_agreement": float(
            np.mean(np.argmax(set64, axis=1) == np.argmax(set96, axis=1))
        ),
    }
    (args.output_dir / "production_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
