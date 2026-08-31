#!/usr/bin/env python3
"""Incremental DeepSets-v2 audit on top of the scored v1 pool ranker."""

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


BASE_NAMES = [
    "multvae", "recvae", "bm25bpr", "current", "global_log",
    "global_linear", "time60", "source_count", "history_count",
    "history_trend30", "history_trend90", "history_trend365",
    "history_last_seen",
]
EXTRA_NAMES = [
    "time14", "time30", "time90", "time180", "roll30", "roll60",
    "roll90", "past60", "future60", "excess_-1", "excess_0",
    "excess_1", "excess_2", "source_time60", "source_time180",
]


def source_time_signal(src, times, candidates, width_days, offset_days):
    bucket = np.floor_divide(
        times - int(times.min()) + offset_days * pool.DAY,
        width_days * pool.DAY,
    ).astype(np.int64)
    groups = src.astype(np.int64) * (int(bucket.max()) + 1) + bucket
    keys = np.repeat(groups, 100) * pool.KEY_BASE + candidates.ravel()
    _, inverse, count = np.unique(keys, return_inverse=True, return_counts=True)
    return pool.qnorm(np.log1p(count[inverse].reshape(candidates.shape)))


def extend(values, edges, negative_items):
    target = values["target"]
    candidates = values["candidates"]
    capacity = max(int(edges[:, 1].max()), int(candidates.max()))
    extras = []
    for width in (14, 30, 90, 180):
        extras.append(
            0.5 * (
                pool.temporal_signal(target, candidates, capacity, width, 0)
                + pool.temporal_signal(
                    target, candidates, capacity, width, width // 2
                )
            )
        )
    rolling = pool.rolling_signals(target, candidates, capacity)
    extras.extend(
        rolling[name]
        for name in ("roll_30", "roll_60", "roll_90", "past_60", "future_60")
    )
    global_set = pool.global_signals(candidates, capacity, negative_items)
    extras.extend(
        global_set[name]
        for name in ("excess_-1", "excess_0", "excess_1", "excess_2")
    )
    for width in (60, 180):
        extras.append(
            0.5 * (
                source_time_signal(
                    target[:, 0], target[:, 2], candidates, width, 0
                )
                + source_time_signal(
                    target[:, 0], target[:, 2], candidates, width, width // 2
                )
            )
        )
    values["feature_v2"] = np.concatenate(
        [values["feature"], np.stack(extras, axis=2).astype(np.float32)], axis=2
    )
    return values


class MetaRanker(nn.Module):
    def __init__(self, dim, architecture):
        if architecture == "h32":
            self.layers = nn.Sequential(
                nn.Linear(dim, 32), nn.Relu(), nn.Linear(32, 1)
            )
        elif architecture == "h64":
            self.layers = nn.Sequential(
                nn.Linear(dim, 64), nn.Relu(), nn.Linear(64, 1)
            )
        elif architecture == "deep":
            self.layers = nn.Sequential(
                nn.Linear(dim, 64), nn.Relu(), nn.Linear(64, 32), nn.Relu(),
                nn.Linear(32, 1),
            )
        else:
            raise ValueError(architecture)

    def execute(self, values):
        return self.layers(values).squeeze(-1)


class SetMetaRanker(nn.Module):
    def __init__(self, dim, hidden):
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.Relu(),
            nn.Linear(hidden, hidden), nn.Relu(),
        )
        self.pool = nn.Linear(hidden, 1)
        self.gate = nn.Linear(hidden, hidden)
        self.shift = nn.Linear(hidden, hidden)
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Relu(), nn.Linear(hidden, 1)
        )

    def execute(self, values):
        encoded = self.encoder(values)
        attention = nn.softmax(self.pool(encoded).squeeze(-1), dim=1)
        context = (attention.unsqueeze(-1) * encoded).sum(dim=1, keepdims=True)
        conditioned = (
            encoded * (1.0 + jt.sigmoid(self.gate(context)))
            + self.shift(context)
        )
        return self.output(conditioned).squeeze(-1)


def predict(net, feature, batch):
    output = np.empty(feature.shape[:2], np.float32)
    net.eval()
    with jt.no_grad():
        for start in range(0, len(feature), batch):
            block = feature[start : start + batch]
            score = net(jt.array(block.reshape(-1, block.shape[-1])))
            output[start : start + len(block)] = np.asarray(
                score.reshape((len(block), 100)).data, np.float32
            )
    return pool.qnorm(output)


def train(feature, labels, split, architecture, epochs, batch, seed):
    np.random.seed(seed)
    jt.set_global_seed(seed)
    rng = np.random.default_rng(seed)
    net = MetaRanker(feature.shape[-1], architecture)
    optimizer = jt.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best = None
    for epoch in range(epochs):
        net.train()
        losses = []
        order = rng.permutation(split)
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            block = feature[ids]
            score = net(jt.array(block.reshape(-1, block.shape[-1])))
            loss = nn.cross_entropy_loss(
                score.reshape((len(ids), 100)), jt.array(labels[ids])
            )
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.data).item()))
        valid = predict(net, feature[split:], batch)
        value = pool.mrr(valid, labels[split:])
        if best is None or value > best[0]:
            best = (
                value,
                epoch + 1,
                {k: np.asarray(v.data).copy() for k, v in net.state_dict().items()},
            )
        print(
            f"{architecture} epoch={epoch + 1} loss={np.mean(losses):.6f} "
            f"valid_mrr={value:.6f}", flush=True,
        )
    net.load_state_dict({k: jt.array(v) for k, v in best[2].items()})
    return net, best[:2]


def predict_set(net, feature, batch):
    output = np.empty(feature.shape[:2], np.float32)
    net.eval()
    with jt.no_grad():
        for start in range(0, len(feature), batch):
            block = feature[start:start + batch]
            output[start:start + len(block)] = np.asarray(
                net(jt.array(block)).data, np.float32
            )
    return pool.qnorm(output)


def train_set(feature, labels, split, hidden, epochs, batch, seed):
    np.random.seed(seed)
    jt.set_global_seed(seed)
    rng = np.random.default_rng(seed)
    net = SetMetaRanker(feature.shape[-1], hidden)
    optimizer = jt.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best = None
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(split)
        losses = []
        for start in range(0, len(order), batch):
            ids = order[start:start + batch]
            score = net(jt.array(feature[ids]))
            loss = nn.cross_entropy_loss(score, jt.array(labels[ids]))
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.data).item()))
        valid = predict_set(net, feature[split:], batch)
        value = pool.mrr(valid, labels[split:])
        if best is None or value > best[0]:
            best = (
                value, epoch + 1,
                {k: np.asarray(v.data).copy() for k, v in net.state_dict().items()},
            )
        print(
            f"set{hidden} epoch={epoch + 1} loss={np.mean(losses):.6f} "
            f"valid_mrr={value:.6f}", flush=True,
        )
    net.load_state_dict({k: jt.array(v) for k, v in best[2].items()})
    return net, best[:2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    jt.flags.use_cuda = 1
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is required")
    with zipfile.ZipFile(args.data) as archive:
        train_frame = pd.read_csv(
            archive.open("dataset2/train.csv"), usecols=["src", "dst", "time"]
        )
        test = pd.read_csv(archive.open("dataset2/test.csv"))
    edges = train_frame[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    negative_items = int(test.iloc[:, 2:].to_numpy(np.int64).max())
    slices = {}
    for index, name in enumerate(("y2009", "strict")):
        values = rank.build_slice(
            edges, test, args.models, name, pool.SLICES[name], 153420,
            args.batch, args.seed + index,
        )
        slices[name] = extend(values, edges, negative_items)
        print(name, slices[name]["feature_v2"].shape, flush=True)

    split = int(len(slices["y2009"]["labels"]) * 0.8)
    all_columns = np.arange(len(BASE_NAMES) + len(EXTRA_NAMES))
    specs = {
        "v1": (np.arange(13), "h32"),
        "set64": (all_columns, "set64"),
        "set96": (all_columns, "set96"),
    }
    predictions = {}
    training = {}
    for offset, (name, (columns, architecture)) in enumerate(specs.items()):
        valid_members, strict_members, members = [], [], []
        for member in range(3):
            seed = args.seed + 1000 * (offset + 1) + member
            feature = slices["y2009"]["feature_v2"][:, :, columns]
            if architecture.startswith("set"):
                net, best = train_set(
                    feature, slices["y2009"]["labels"], split,
                    int(architecture[3:]), 8, args.batch, seed,
                )
                predict_function = predict_set
            else:
                net, best = train(
                    feature, slices["y2009"]["labels"], split, architecture,
                    8, args.batch, seed,
                )
                predict_function = predict
            valid_members.append(predict_function(
                net,
                slices["y2009"]["feature_v2"][split:, :, columns],
                args.batch,
            ))
            strict_members.append(predict_function(
                net, slices["strict"]["feature_v2"][:, :, columns],
                args.batch,
            ))
            members.append({"seed": seed, "valid_mrr": best[0], "epoch": best[1]})
        predictions[name] = {
            "valid": pool.qnorm(np.mean(valid_members, axis=0)),
            "strict": pool.qnorm(np.mean(strict_members, axis=0)),
        }
        training[name] = members
        print(name, json.dumps(members), flush=True)

    predictions["set_all"] = {
        part: pool.qnorm(np.mean([
            predictions[name][part] for name in ("set64", "set96")
        ], axis=0))
        for part in ("valid", "strict")
    }
    valid_labels = slices["y2009"]["labels"][split:]
    strict_labels = slices["strict"]["labels"]
    valid_current = slices["y2009"]["current"][split:]
    strict_current = slices["strict"]["current"]
    valid_baseline = valid_current + 1.65 * predictions["v1"]["valid"]
    strict_baseline = strict_current + 1.65 * predictions["v1"]["strict"]
    baseline = {
        "valid": pool.mrr(valid_baseline, valid_labels),
        "strict": pool.mrr(strict_baseline, strict_labels),
    }
    result = {
        "feature_names": BASE_NAMES + EXTRA_NAMES,
        "training": training,
        "baseline": baseline,
        "incremental": {},
    }
    for name in ("set64", "set96", "set_all"):
        grid = {}
        for alpha in np.arange(0.0, 1.001, 0.05):
            meta = (
                (1.0 - alpha) * predictions["v1"]["valid"]
                + alpha * predictions[name]["valid"]
            )
            grid[float(alpha)] = pool.mrr(
                valid_current + 1.65 * meta, valid_labels
            )
        alpha = max(grid, key=lambda value: (grid[value], -value))
        strict_meta = (
            (1.0 - alpha) * predictions["v1"]["strict"]
            + alpha * predictions[name]["strict"]
        )
        strict_mrr = pool.mrr(
            strict_current + 1.65 * strict_meta, strict_labels
        )
        before_positive = strict_baseline[
            np.arange(len(strict_labels)), strict_labels
        ]
        after_score = strict_current + 1.65 * strict_meta
        after_positive = after_score[np.arange(len(strict_labels)), strict_labels]
        columns = np.arange(100)[None, :]
        before_rank = 1 + (strict_baseline > before_positive[:, None]).sum(1)
        before_rank += (
            (strict_baseline == before_positive[:, None])
            & (columns < strict_labels[:, None])
        ).sum(1)
        after_rank = 1 + (after_score > after_positive[:, None]).sum(1)
        after_rank += (
            (after_score == after_positive[:, None])
            & (columns < strict_labels[:, None])
        ).sum(1)
        paired = 1.0 / after_rank - 1.0 / before_rank
        result["incremental"][name] = {
            "alpha": alpha,
            "valid_mrr": grid[alpha],
            "valid_delta": grid[alpha] - baseline["valid"],
            "strict_mrr": strict_mrr,
            "strict_delta": strict_mrr - baseline["strict"],
            "strict_delta_se": float(
                paired.std(ddof=1) / np.sqrt(len(paired))
            ),
            "strict_positive_rows": float(np.mean(paired > 0)),
            "strict_negative_rows": float(np.mean(paired < 0)),
        }
        print(name, json.dumps(result["incremental"][name]), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
