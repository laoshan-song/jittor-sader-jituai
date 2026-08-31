#!/usr/bin/env python3
"""Train a time-decayed BM25-weighted matrix factorization model in Jittor."""

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt
from jittor import nn


DAY = 86400
ORIGINS = {
    "y2008": 1199145600,
    "y2009": 1230768000,
    "strict": 1262044800,
    "prod": 1296345600,
}


def mapped(values, sorted_ids):
    index = np.searchsorted(sorted_ids, values)
    valid = index < len(sorted_ids)
    valid[valid] &= sorted_ids[index[valid]] == values[valid]
    return index, valid


def load_pairs(data, origin, decay_days, bm25_k1, bm25_b):
    with zipfile.ZipFile(data) as archive:
        train = pd.read_csv(
            archive.open("dataset2/train.csv"), usecols=["src", "dst", "time"]
        )
    edges = train.loc[
        train.time < origin, ["src", "dst", "time"]
    ].to_numpy(np.int64, copy=False)
    users, items = np.unique(edges[:, 0]), np.unique(edges[:, 1])
    user_index, user_valid = mapped(edges[:, 0], users)
    item_index, item_valid = mapped(edges[:, 1], items)
    assert user_valid.all() and item_valid.all()
    raw = np.exp(-(origin - edges[:, 2]) / (decay_days * DAY)).astype(np.float32)
    keys = user_index.astype(np.int64) * len(items) + item_index
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    confidence = np.zeros(len(unique_keys), np.float32)
    np.add.at(confidence, inverse, raw)
    pair_user = unique_keys // len(items)
    pair_item = unique_keys % len(items)

    row_length = np.bincount(
        pair_user, weights=confidence, minlength=len(users)
    ).astype(np.float32)
    average_length = max(float(row_length.mean()), 1e-6)
    length_norm = (1.0 - bm25_b) + bm25_b * row_length / average_length
    item_df = np.bincount(pair_item, minlength=len(items)).astype(np.float32)
    idf = np.log(len(users)) - np.log1p(item_df)
    confidence = (
        confidence
        * (bm25_k1 + 1.0)
        / (bm25_k1 * length_norm[pair_user] + confidence)
        * idf[pair_item]
    )
    confidence = np.clip(confidence / max(float(confidence.mean()), 1e-6), 0.05, 20.0)
    return users, items, pair_user, pair_item, confidence.astype(np.float32)


class BPR(nn.Module):
    def __init__(self, users, items, factors):
        self.user = nn.Embedding(users, factors)
        self.item = nn.Embedding(items, factors)
        self.user.weight.assign(jt.randn(self.user.weight.shape) * 0.01)
        self.item.weight.assign(jt.randn(self.item.weight.shape) * 0.01)

    def loss(self, users, positive, negative, confidence, regularization):
        user_vector = self.user(users)
        positive_vector = self.item(positive)
        negative_vector = self.item(negative)
        positive_score = (user_vector * positive_vector).sum(dim=1)
        negative_score = (
            user_vector.unsqueeze(1) * negative_vector
        ).sum(dim=2)
        ranking = nn.softplus(-(positive_score.unsqueeze(1) - negative_score))
        ranking = (ranking * confidence.unsqueeze(1)).mean()
        penalty = regularization * (
            (user_vector * user_vector).mean()
            + (positive_vector * positive_vector).mean()
            + (negative_vector * negative_vector).mean()
        )
        return ranking + penalty


def scalar(value):
    return float(np.asarray(value.data).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slice", choices=ORIGINS)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factors", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32768)
    parser.add_argument("--negatives", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--decay-days", type=float, default=365.0)
    parser.add_argument("--bm25-k1", type=float, default=100.0)
    parser.add_argument("--bm25-b", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is required")
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    origin = ORIGINS[args.slice]
    users, items, pair_user, pair_item, confidence = load_pairs(
        args.data, origin, args.decay_days, args.bm25_k1, args.bm25_b
    )
    print(
        f"slice={args.slice} users={len(users)} items={len(items)} "
        f"pairs={len(pair_user)} factors={args.factors}",
        flush=True,
    )
    model = BPR(len(users), len(items), args.factors)
    optimizer = jt.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = rng.permutation(len(pair_user))
        total = 0.0
        for start in range(0, len(order), args.batch):
            ids = order[start : start + args.batch]
            negative = rng.integers(
                0, len(items), size=(len(ids), args.negatives), dtype=np.int64
            )
            loss = model.loss(
                jt.array(pair_user[ids]),
                jt.array(pair_item[ids]),
                jt.array(negative),
                jt.array(confidence[ids]),
                args.regularization,
            )
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.clip_grad_norm(5.0, 2)
            optimizer.step()
            total += scalar(loss) * len(ids)
        print(f"epoch={epoch + 1} loss={total / len(pair_user):.6f}", flush=True)

    np.savez_compressed(
        args.output,
        kind=np.array("bm25_bpr"),
        users=users,
        items=items,
        user=np.asarray(model.user.weight.data, np.float32),
        item=np.asarray(model.item.weight.data, np.float32),
        ibias=np.zeros(len(items), np.float32),
        history_end=np.int64(origin),
        decay_days=np.float64(args.decay_days),
        factors=np.int64(args.factors),
        epochs=np.int64(args.epochs),
        negatives=np.int64(args.negatives),
        seed=np.int64(args.seed),
        param__user__weight=np.asarray(model.user.weight.data, np.float32),
        param__item__weight=np.asarray(model.item.weight.data, np.float32),
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
