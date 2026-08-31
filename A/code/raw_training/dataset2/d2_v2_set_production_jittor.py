#!/usr/bin/env python3
"""Train/load the audited Jittor DeepSets-v2 ensemble and score official test."""

import argparse
import json
import zipfile
from pathlib import Path

import jittor as jt
import numpy as np
import pandas as pd
from jittor import nn

import d2_pool_ranker_eval_jittor as rank
import d2_pool_ranker_production_jittor as v1_production
import d2_temporal_pool_eval_jittor as pool
import d2_v2_set_incremental_audit_jittor as audit


MEMBERS = (
    (64, 20262725, 8),
    (64, 20262726, 7),
    (64, 20262727, 8),
    (96, 20263725, 8),
    (96, 20263726, 5),
    (96, 20263727, 8),
)


def train_full(feature, labels, hidden, epochs, batch, seed):
    np.random.seed(seed)
    jt.set_global_seed(seed)
    rng = np.random.default_rng(seed)
    net = audit.SetMetaRanker(feature.shape[-1], hidden)
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
            f"hidden={hidden} seed={seed} epoch={epoch + 1} "
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
        feature_count=np.asarray(len(audit.BASE_NAMES) + len(audit.EXTRA_NAMES)),
        seed=np.asarray(seed), epochs=np.asarray(epochs),
        kind=np.asarray("jittor_deepsets_pool_meta_v2"),
    )
    np.savez(path, **payload)


def load_checkpoint(path):
    saved = np.load(path, allow_pickle=False)
    feature_count = int(saved["feature_count"])
    expected = len(audit.BASE_NAMES) + len(audit.EXTRA_NAMES)
    if feature_count != expected:
        raise ValueError(f"feature mismatch in {path}: {feature_count} != {expected}")
    net = audit.SetMetaRanker(feature_count, int(saved["hidden"]))
    net.load_state_dict({
        str(name): jt.array(saved[f"state_{index}"])
        for index, name in enumerate(saved["state_names"])
    })
    return net


def official_features(edges, test, models, batch):
    base = v1_production.build_test(edges, test, models, batch)
    candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    target = np.zeros((len(test), 3), np.int64)
    target[:, 0] = test.src.to_numpy(np.int64, copy=False)
    target[:, 2] = test.time.to_numpy(np.int64, copy=False)
    values = {"target": target, "candidates": candidates, "feature": base}
    return audit.extend(values, edges, int(candidates.max()))["feature_v2"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=6)
    parser.add_argument("--cpu-inference", action="store_true")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--candidate-seed", type=int, default=20260726)
    args = parser.parse_args()

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
    nets, checkpoint_paths = [], []
    if args.checkpoints:
        for path in args.checkpoints:
            checkpoint_paths.append(str(path))
    else:
        strict = rank.build_slice(
            edges, test, args.models, "strict", pool.SLICES["strict"],
            153420, args.batch, args.candidate_seed,
        )
        feature = audit.extend(
            strict, edges, int(test.iloc[:, 2:].to_numpy(np.int64).max())
        )["feature_v2"]
        for hidden, seed, epochs in MEMBERS:
            net = train_full(
                feature, strict["labels"], hidden, epochs, args.batch, seed
            )
            path = args.output_dir / f"set{hidden}_seed{seed}_jittor.npz"
            save_checkpoint(path, net, hidden, seed, epochs)
            reloaded = load_checkpoint(path)
            before = audit.predict_set(net, feature[:1024], args.batch)
            after = audit.predict_set(reloaded, feature[:1024], args.batch)
            if not np.allclose(before, after, rtol=0, atol=1e-6):
                raise RuntimeError(f"checkpoint reload mismatch: {path}")
            checkpoint_paths.append(str(path))
            print(f"verified {path}", flush=True)
        del strict, feature

    feature = official_features(edges, test, args.models, args.batch)
    jt.sync_all()
    if args.cpu_inference:
        jt.flags.use_cuda = 0
    nets = [load_checkpoint(path) for path in checkpoint_paths]
    predictions = [audit.predict_set(net, feature, args.batch) for net in nets]
    set64 = pool.qnorm(np.mean(predictions[:3], axis=0))
    set96 = pool.qnorm(np.mean(predictions[3:], axis=0))
    set_all = pool.qnorm(0.5 * (set64 + set96))
    outputs = {
        "official_set64.npy": set64,
        "official_set96.npy": set96,
        "official_set_all.npy": set_all,
    }
    for name, value in outputs.items():
        np.save(args.output_dir / name, np.round(value, 2).astype(np.float32))
    top1 = [np.argmax(value, axis=1) for value in predictions]
    diagnostics = {
        "shape": list(set_all.shape),
        "finite": bool(np.isfinite(set_all).all()),
        "checkpoints": checkpoint_paths,
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
