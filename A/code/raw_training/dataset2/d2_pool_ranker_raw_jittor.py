#!/usr/bin/env python3
"""Train a Jittor candidate-level meta-ranker on rolling dataset2 slices."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt
from jittor import nn

import d2_temporal_pool_raw_jittor as pool


def item_signals(history, candidates, origin):
    capacity = int(max(history[:, 1].max(initial=0), candidates.max(initial=0)))
    dst = history[:, 1].astype(np.int64, copy=False)
    tim = history[:, 2].astype(np.float64, copy=False)
    output = []
    count = np.bincount(dst, minlength=capacity + 1)
    output.append(pool.qnorm(np.log1p(count[candidates]).astype(np.float32)))
    for days in (30, 90, 365):
        weight = np.exp(-(origin - tim) / (days * pool.DAY))
        trend = np.bincount(dst, weights=weight, minlength=capacity + 1)
        output.append(pool.qnorm(np.log1p(trend[candidates]).astype(np.float32)))
    last = np.full(capacity + 1, -1, np.int64)
    np.maximum.at(last, dst, history[:, 2])
    gap = np.where(
        last[candidates] >= 0,
        np.log1p(np.maximum(origin - last[candidates], 0) / pool.DAY),
        30.0,
    )
    output.append(pool.qnorm(-gap.astype(np.float32)))
    return output


def source_count_signal(src, candidates):
    keys = np.repeat(src, 100).astype(np.int64) * pool.KEY_BASE + candidates.ravel()
    _, inverse, count = np.unique(keys, return_inverse=True, return_counts=True)
    values = count[inverse].reshape(candidates.shape).astype(np.float32)
    return pool.qnorm(values)


def build_slice(
    edges,
    test,
    models,
    name,
    bounds,
    targets,
    batch,
    seed,
    replay_pool=None,
    replay_mode="uniform",
):
    origin, end = bounds
    history = edges[edges[:, 2] < origin]
    target = edges[(edges[:, 2] >= origin) & (edges[:, 2] < end)]
    target = target[np.argsort(target[:, 2], kind="stable")]
    rng = np.random.default_rng(seed + 100)
    if len(target) > targets:
        target = target[np.sort(rng.choice(len(target), targets, replace=False))]
        target = target[np.argsort(target[:, 2], kind="stable")]
    negative_items = int(test.iloc[:, 2:].to_numpy(np.int64).max())
    capacity = max(negative_items, int(edges[:, 1].max()))
    if replay_pool is None or replay_mode == "uniform":
        candidates, labels = pool.replay(target, negative_items, seed)
        candidate_audit = {
            "mode": "uniform",
            "seed": int(seed),
            "negative_item_count": int(negative_items),
            "candidates_sha256": pool.array_sha256(candidates),
            "labels_sha256": pool.array_sha256(labels),
        }
    elif replay_mode == "test-pool":
        candidates, labels, candidate_audit = pool.replay_from_pool(
            target, replay_pool, seed
        )
    else:
        raise ValueError(f"unsupported replay mode: {replay_mode}")
    old_keys = np.unique(history[:, 0] * pool.KEY_BASE + history[:, 1])
    scores = {}
    for model_name in ("multvae", "recvae", "bm25bpr"):
        scores[model_name] = pool.score_model(
            models / f"model_{model_name}_{name}_jittor.npz",
            target[:, 0],
            candidates,
            old_keys,
            batch,
        )
    global_set = pool.global_signals(candidates, capacity, negative_items)
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
    feature_list = [
        scores["multvae"],
        scores["recvae"],
        scores["bm25bpr"],
        pool.qnorm(current),
        global_set["global_log"],
        global_set["global_linear"],
        time60,
        source_count_signal(target[:, 0], candidates),
        *item_signals(history, candidates, origin),
    ]
    feature = np.stack(feature_list, axis=2).astype(np.float32)
    return {
        "target": target,
        "candidates": candidates,
        "labels": labels,
        "current": current.astype(np.float32),
        "feature": feature,
        "candidate_audit": candidate_audit,
    }


class Ranker(nn.Module):
    def __init__(self, dim, hidden):
        if hidden:
            self.layers = nn.Sequential(
                nn.Linear(dim, hidden), nn.Relu(), nn.Linear(hidden, 1)
            )
        else:
            self.layers = nn.Linear(dim, 1)

    def execute(self, values):
        return self.layers(values).squeeze(-1)


def predict(net, feature, batch):
    output = np.empty(feature.shape[:2], np.float32)
    net.eval()
    with jt.no_grad():
        for start in range(0, len(feature), batch):
            end = min(start + batch, len(feature))
            block = feature[start:end]
            score = net(jt.array(block.reshape(-1, block.shape[-1])))
            output[start:end] = np.asarray(
                score.reshape((len(block), 100)).data, np.float32
            )
    return pool.qnorm(output)


def train_ranker(feature, labels, split, hidden, epochs, batch, seed):
    np.random.seed(seed)
    jt.set_global_seed(seed)
    rng = np.random.default_rng(seed)
    net = Ranker(feature.shape[-1], hidden)
    optimizer = jt.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best = None
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(split)
        losses = []
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            block = feature[ids]
            score = net(jt.array(block.reshape(-1, block.shape[-1])))
            score = score.reshape((len(ids), 100))
            loss = nn.cross_entropy_loss(score, jt.array(labels[ids]))
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.data).item()))
        valid_score = predict(net, feature[split:], batch)
        valid_mrr = pool.mrr(valid_score, labels[split:])
        candidate = (valid_mrr, -epoch)
        if best is None or candidate > best[0]:
            state = {key: np.asarray(value.data).copy() for key, value in net.state_dict().items()}
            best = (candidate, state)
        print(
            f"hidden={hidden} epoch={epoch + 1} loss={np.mean(losses):.6f} "
            f"valid_mrr={valid_mrr:.6f}",
            flush=True,
        )
    net.load_state_dict({key: jt.array(value) for key, value in best[1].items()})
    return net, best[0][0], 1 - best[0][1]


def tune_blend(base, ranker, labels):
    best = None
    for weight in np.arange(0.0, 0.501, 0.02):
        value = pool.mrr(base + weight * ranker, labels)
        candidate = (value, -float(weight))
        if best is None or candidate > best:
            best = candidate
    return {"mrr": best[0], "weight": -best[1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=int, default=153420)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is required")
    with zipfile.ZipFile(args.data) as archive:
        train = pd.read_csv(
            archive.open("dataset2/train.csv"), usecols=["src", "dst", "time"]
        )
        test = pd.read_csv(archive.open("dataset2/test.csv"))
    edges = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    slices = {}
    for index, (name, bounds) in enumerate(pool.SLICES.items()):
        slices[name] = build_slice(
            edges, test, args.models, name, bounds, args.targets, args.batch,
            args.seed + index,
        )
        print(name, "features", slices[name]["feature"].shape, flush=True)

    y2008 = slices["y2008"]
    split = int(len(y2008["labels"]) * 0.8)
    results = {"models": {}, "slices": {}}
    trained = {}
    for hidden in (0, 32):
        net, valid_mrr, epoch = train_ranker(
            y2008["feature"], y2008["labels"], split, hidden,
            args.epochs, args.batch, args.seed + hidden,
        )
        valid_ranker = predict(net, y2008["feature"][split:], args.batch)
        blend = tune_blend(
            y2008["current"][split:], valid_ranker, y2008["labels"][split:]
        )
        results["models"][str(hidden)] = {
            "ranker_valid_mrr": valid_mrr,
            "epoch": epoch,
            "blend": blend,
        }
        trained[hidden] = (net, blend)

    selected_hidden = max(
        trained, key=lambda key: results["models"][str(key)]["blend"]["mrr"]
    )
    net, blend = trained[selected_hidden]
    results["selected_hidden"] = selected_hidden
    for name, values in slices.items():
        ranker = predict(net, values["feature"], args.batch)
        current_mrr = pool.mrr(values["current"], values["labels"])
        mixed_mrr = pool.mrr(
            values["current"] + blend["weight"] * ranker, values["labels"]
        )
        results["slices"][name] = {
            "current": current_mrr,
            "ranker": pool.mrr(ranker, values["labels"]),
            "mixed": mixed_mrr,
            "delta": mixed_mrr - current_mrr,
        }
        print(name, results["slices"][name], flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
