#!/usr/bin/env python3
"""Train the Dataset2 community representation with Jittor BPR only."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import jittor as jt
import numpy as np
import pandas as pd
from jittor import nn


DAY = 86400


def mapped(values: np.ndarray, sorted_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.searchsorted(sorted_ids, values)
    valid = index < len(sorted_ids)
    valid[valid] &= sorted_ids[index[valid]] == values[valid]
    return index, valid


def build_pairs(train: np.ndarray, origin: int, decay_days: float, bm25_k1: float, bm25_b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history = train[train[:, 2] < origin]
    if not len(history):
        raise ValueError("history is empty at the selected origin")
    users = np.unique(history[:, 0])
    items = np.unique(history[:, 1])
    user_index, user_valid = mapped(history[:, 0], users)
    item_index, item_valid = mapped(history[:, 1], items)
    if not user_valid.all() or not item_valid.all():
        raise RuntimeError("identifier mapping failed")
    raw = np.exp(-(origin - history[:, 2]) / (decay_days * DAY)).astype(np.float32)
    keys = user_index.astype(np.int64) * len(items) + item_index
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    confidence = np.zeros(len(unique_keys), dtype=np.float32)
    np.add.at(confidence, inverse, raw)
    pair_user = unique_keys // len(items)
    pair_item = unique_keys % len(items)
    row_length = np.bincount(pair_user, weights=confidence, minlength=len(users)).astype(np.float32)
    average_length = max(float(row_length.mean()), 1e-6)
    length_norm = (1.0 - bm25_b) + bm25_b * row_length / average_length
    item_df = np.bincount(pair_item, minlength=len(items)).astype(np.float32)
    idf = np.log(len(users)) - np.log1p(item_df)
    confidence = confidence * (bm25_k1 + 1.0) / (bm25_k1 * length_norm[pair_user] + confidence) * idf[pair_item]
    confidence = np.clip(confidence / max(float(confidence.mean()), 1e-6), 0.05, 20.0).astype(np.float32)
    if not np.isfinite(confidence).all():
        raise FloatingPointError("non-finite BM25 confidence")
    return users, items, pair_user, pair_item, confidence


class BPR(nn.Module):
    def __init__(self, users: int, items: int, factors: int):
        self.user = nn.Embedding(users, factors)
        self.item = nn.Embedding(items, factors)
        self.user.weight.assign(jt.randn(self.user.weight.shape) * 0.01)
        self.item.weight.assign(jt.randn(self.item.weight.shape) * 0.01)

    def loss(self, users, positive, negative, confidence, regularization: float):
        user_vector = self.user(users)
        positive_vector = self.item(positive)
        negative_vector = self.item(negative)
        positive_score = (user_vector * positive_vector).sum(dim=1)
        negative_score = (user_vector.unsqueeze(1) * negative_vector).sum(dim=2)
        ranking = (nn.softplus(-(positive_score.unsqueeze(1) - negative_score)) * confidence.unsqueeze(1)).mean()
        penalty = regularization * ((user_vector * user_vector).mean() + (positive_vector * positive_vector).mean() + (negative_vector * negative_vector).mean())
        return ranking + penalty


def scalar(value) -> float:
    return float(np.asarray(value.data).item())


def train(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {args.output}")
    if args.cuda:
        if not jt.has_cuda:
            raise RuntimeError("--cuda was requested but Jittor CUDA is unavailable")
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    with zipfile.ZipFile(args.data) as archive:
        frame = pd.read_csv(archive.open("dataset2/train.csv"), usecols=["src", "dst", "time"])
    train_rows = frame[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    if np.any(train_rows[1:, 2] < train_rows[:-1, 2]):
        raise ValueError("Dataset2 training rows are not time-stable")
    users, items, pair_user, pair_item, confidence = build_pairs(train_rows, args.origin, args.decay_days, args.bm25_k1, args.bm25_b)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    model = BPR(len(users), len(items), args.factors)
    optimizer = jt.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    generator = np.random.Generator(np.random.PCG64(args.seed))
    losses: list[float] = []
    for epoch in range(args.epochs):
        order = generator.permutation(len(pair_user))
        total = 0.0
        for start in range(0, len(order), args.batch):
            identifiers = order[start:start + args.batch]
            negative = generator.integers(0, len(items), size=(len(identifiers), args.negatives), dtype=np.int64)
            loss = model.loss(jt.array(pair_user[identifiers]), jt.array(pair_item[identifiers]), jt.array(negative), jt.array(confidence[identifiers]), args.regularization)
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.clip_grad_norm(args.gradient_clip_norm, 2)
            optimizer.step()
            total += scalar(loss) * len(identifiers)
        value = total / len(pair_user)
        if not math.isfinite(value):
            raise FloatingPointError("non-finite Jittor BPR training loss")
        losses.append(value)
        print(f"epoch={epoch + 1} loss={value:.8f}", flush=True)
    jt.sync_all()
    user = np.asarray(model.user.weight.data, dtype=np.float32).copy()
    item = np.asarray(model.item.weight.data, dtype=np.float32).copy()
    if not np.isfinite(user).all() or not np.isfinite(item).all():
        raise FloatingPointError("non-finite checkpoint values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "bm25_bpr",
        "history_end": args.origin,
        "factors": args.factors,
        "epochs": args.epochs,
        "batch": args.batch,
        "negatives": args.negatives,
        "learning_rate": args.lr,
        "regularization": args.regularization,
        "decay_days": args.decay_days,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "seed": args.seed,
        "device": "cuda" if args.cuda else "cpu",
        "epoch_losses": losses,
        "train_rows": int(len(train_rows)),
        "history_rows": int(np.sum(train_rows[:, 2] < args.origin)),
    }
    np.savez_compressed(args.output, kind=np.asarray("bm25_bpr"), lineage=np.asarray("track1_community_bpr_v1"), users=users.astype(np.int64), items=items.astype(np.int64), user=user, item=item, ibias=np.zeros(len(items), dtype=np.float32), history_end=np.asarray(args.origin, dtype=np.int64), factors=np.asarray(args.factors, dtype=np.int64), metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)), param__user__weight=user.copy(), param__item__weight=item.copy())
    print(json.dumps({"checkpoint": str(args.output), "users": int(len(users)), "items": int(len(items)), "losses": losses}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Jittor Dataset2 community BPR model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin", type=int, default=1296345600)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32768)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--regularization", type=float, default=0.0001)
    parser.add_argument("--decay-days", type=float, default=365.0)
    parser.add_argument("--bm25-k1", type=float, default=100.0)
    parser.add_argument("--bm25-b", type=float, default=0.8)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=202607254)
    parser.add_argument("--cuda", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
