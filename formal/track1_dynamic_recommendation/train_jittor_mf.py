"""Jittor matrix-factorization reranker for Track 1.

This is the reproducible Jittor implementation used for formal submissions.
It trains a BPR matrix-factorization model per scene, blends it with the
history/sequence baseline, and writes a valid result.zip.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import zipfile
from pathlib import Path

import jittor as jt
from jittor import nn
import numpy as np

from baseline import HistoryBaseline, probabilities
from rank_utils import rank_probabilities


HEURISTIC_WEIGHTS = {
    "pair_weight": 6.0,
    "pair_recency_weight": 4.0,
    "dst_pop_weight": 0.4,
    "dst_recency_weight": 0.2,
    "sequence_weight": 2.5,
    "repeat_recent_weight": 2.0,
}


class JittorMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.assign(jt.randn((num_users, dim)) * 0.02)
        self.item_emb.weight.assign(jt.randn((num_items, dim)) * 0.02)
        self.item_bias.weight.assign(jt.zeros((num_items, 1)))

    def score(self, users: jt.Var, items: jt.Var) -> jt.Var:
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=1) + self.item_bias(items).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Jittor MF reranker and write result.zip")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/track1/result.zip"))
    parser.add_argument("--scenes", default="dataset1,dataset2")
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--reg", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mf-weight", type=float, default=1.0)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--uniform-mix", type=float, default=0.02)
    parser.add_argument(
        "--probability-mode",
        choices=("softmax", "rank"),
        default="rank",
    )
    parser.add_argument("--limit-train-rows", type=int, default=0)
    parser.add_argument("--limit-test-rows", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    jt.misc.set_global_seed(seed)


def read_scene_rows(
    data_zip: zipfile.ZipFile,
    scene: str,
    limit_train_rows: int = 0,
) -> tuple[list[tuple[int, int, int]], set[int], set[int]]:
    rows: list[tuple[int, int, int]] = []
    users: set[int] = set()
    items: set[int] = set()

    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            src = int(row["src"])
            dst = int(row["dst"])
            time_value = int(row["time"])
            rows.append((src, dst, time_value))
            users.add(src)
            items.add(dst)
            if limit_train_rows and len(rows) >= limit_train_rows:
                break

    with open_csv(data_zip, f"{scene}/test.csv") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            users.add(int(row[0]))
            items.update(int(value) for value in row[2:])

    return rows, users, items


def build_heuristic(rows: list[tuple[int, int, int]]) -> HistoryBaseline:
    model = HistoryBaseline(**HEURISTIC_WEIGHTS)
    for src, dst, time_value in rows:
        model.update(src, dst, time_value)
    model.finalize()
    return model


def build_mappings(users: set[int], items: set[int]) -> tuple[dict[int, int], dict[int, int]]:
    return (
        {value: index for index, value in enumerate(sorted(users))},
        {value: index for index, value in enumerate(sorted(items))},
    )


def recency_cdf(times: np.ndarray) -> np.ndarray:
    span = max(1, int(times.max()) - int(times.min()))
    weights = 0.2 + 0.8 * ((times - times.min()) / span)
    cdf = np.cumsum(weights, dtype=np.float64)
    cdf /= cdf[-1]
    return cdf


def sample_indices(cdf: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    return np.searchsorted(cdf, rng.random(batch_size), side="right")


def sample_negative_items(
    train_items: np.ndarray,
    popular_items: np.ndarray,
    popular_probs: np.ndarray,
    shape: tuple[int, ...],
    hard_negative_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 0 <= hard_negative_ratio <= 1:
        raise ValueError("--hard-negative-ratio must be in [0, 1].")

    if hard_negative_ratio == 0 or len(popular_items) == 0:
        return rng.choice(train_items, size=shape, replace=True)
    if hard_negative_ratio == 1:
        return rng.choice(popular_items, size=shape, replace=True, p=popular_probs)

    uniform_neg = rng.choice(train_items, size=shape, replace=True)
    hard_neg = rng.choice(popular_items, size=shape, replace=True, p=popular_probs)
    mask = rng.random(shape) < hard_negative_ratio
    return np.where(mask, hard_neg, uniform_neg)


def mf_scores(
    src: int,
    candidates: list[int],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    item_bias: np.ndarray,
) -> np.ndarray:
    user_index = user_to_idx.get(src)
    if user_index is None:
        return np.zeros(len(candidates), dtype=np.float32)

    user_vec = user_emb[user_index]
    scores = np.zeros(len(candidates), dtype=np.float32)
    for index, dst in enumerate(candidates):
        item_index = item_to_idx.get(dst)
        if item_index is not None:
            scores[index] = float(user_vec @ item_emb[item_index] + item_bias[item_index])
    return scores


def blend_scores(heuristic_scores: list[float], mf_score_values: np.ndarray, mf_weight: float) -> list[float]:
    mf_std = float(mf_score_values.std())
    if mf_std > 1e-6:
        mf_score_values = (mf_score_values - float(mf_score_values.mean())) / mf_std
    else:
        mf_score_values = mf_score_values * 0.0
    return [h + mf_weight * float(m) for h, m in zip(heuristic_scores, mf_score_values)]


def output_probabilities(scores: list[float], args: argparse.Namespace) -> list[float]:
    if args.probability_mode == "rank":
        return rank_probabilities(scores)
    return probabilities(scores, args.temperature, args.uniform_mix)


def train_jittor_mf(
    rows: list[tuple[int, int, int]],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    args: argparse.Namespace,
) -> JittorMF:
    rng = np.random.default_rng(args.seed)
    model = JittorMF(len(user_to_idx), len(item_to_idx), args.dim)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    src_idx = np.array([user_to_idx[src] for src, _, _ in rows], dtype=np.int32)
    dst_idx = np.array([item_to_idx[dst] for _, dst, _ in rows], dtype=np.int32)
    times = np.array([time_value for _, _, time_value in rows], dtype=np.int64)
    cdf = recency_cdf(times)
    train_items = np.array(sorted(set(dst_idx.tolist())), dtype=np.int32)
    item_counts = np.bincount(dst_idx, minlength=len(item_to_idx)).astype(np.float64)
    popular_items = np.flatnonzero(item_counts)
    popular_probs = item_counts[popular_items] ** 0.75
    popular_probs /= popular_probs.sum()
    steps_per_epoch = max(1, math.ceil(len(rows) / args.batch_size))

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            batch = sample_indices(cdf, args.batch_size, rng)
            users = jt.array(src_idx[batch])
            pos_items = jt.array(dst_idx[batch])
            neg_items_np = sample_negative_items(
                train_items,
                popular_items,
                popular_probs,
                (len(batch), args.negatives_per_positive),
                args.hard_negative_ratio,
                rng,
            )
            neg_items = jt.array(neg_items_np.astype(np.int32, copy=False))

            pos_scores = model.score(users, pos_items)
            flat_users = users.reshape((-1, 1)).broadcast(neg_items.shape).reshape((-1,))
            flat_neg_items = neg_items.reshape((-1,))
            neg_scores = model.score(flat_users, flat_neg_items).reshape(neg_items.shape)
            margin = pos_scores.reshape((-1, 1)) - neg_scores
            loss = jt.log(1.0 + jt.exp(-margin)).mean()
            neg_emb = model.item_emb(flat_neg_items)
            reg = (
                (model.user_emb(users) * model.user_emb(users)).sum(dim=1)
                + (model.item_emb(pos_items) * model.item_emb(pos_items)).sum(dim=1)
                + (neg_emb * neg_emb)
                .sum(dim=1)
                .reshape(neg_items.shape)
                .mean(dim=1)
            ).mean()
            loss = loss + args.reg * reg
            optimizer.step(loss)
            total_loss += float(loss.item())

        print(f"epoch={epoch} loss={total_loss / steps_per_epoch:.6f}", flush=True)

    return model


def export_jittor_embeddings(model: JittorMF) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(model.user_emb.weight.numpy()),
        np.array(model.item_emb.weight.numpy()),
        np.array(model.item_bias.weight.numpy()).reshape(-1),
    )


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def write_scene(
    data_zip: zipfile.ZipFile,
    output_zip: zipfile.ZipFile,
    scene: str,
    heuristic: HistoryBaseline,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    embeddings: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> int:
    user_emb, item_emb, item_bias = embeddings
    rows = 0
    with open_csv(data_zip, f"{scene}/test.csv") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        with output_zip.open(f"{scene}.csv", "w") as raw_output:
            with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                writer = csv.writer(text_output, lineterminator="\n")
                for row in reader:
                    src = int(row[0])
                    time_value = int(row[1])
                    candidates = [int(value) for value in row[2:]]
                    heuristic_scores = [heuristic.score(src, dst, time_value) for dst in candidates]
                    model_scores = mf_scores(src, candidates, user_to_idx, item_to_idx, user_emb, item_emb, item_bias)
                    scores = blend_scores(heuristic_scores, model_scores, args.mf_weight)
                    probs = output_probabilities(scores, args)
                    writer.writerow([f"{value:.8f}" for value in probs])
                    rows += 1
                    if args.limit_test_rows and rows >= args.limit_test_rows:
                        break
    return rows


def main() -> None:
    args = parse_args()
    jt.flags.use_cuda = 0 if args.cpu or not jt.has_cuda else 1
    set_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.data_zip) as data_zip:
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
            scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
            for scene in scenes:
                print(f"[{scene}] loading", flush=True)
                train_rows, users, items = read_scene_rows(data_zip, scene, args.limit_train_rows)
                user_to_idx, item_to_idx = build_mappings(users, items)
                heuristic = build_heuristic(train_rows)

                print(
                    f"[{scene}] train_rows={len(train_rows)} "
                    f"users={len(users)} items={len(items)} cuda={jt.flags.use_cuda}",
                    flush=True,
                )
                model = train_jittor_mf(train_rows, user_to_idx, item_to_idx, args)
                embeddings = export_jittor_embeddings(model)
                rows = write_scene(
                    data_zip,
                    output_zip,
                    scene,
                    heuristic,
                    user_to_idx,
                    item_to_idx,
                    embeddings,
                    args,
                )
                print(f"[{scene}] wrote {rows} rows", flush=True)

    print(f"submission saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
