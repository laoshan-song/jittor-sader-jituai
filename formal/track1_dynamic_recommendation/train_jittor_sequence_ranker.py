"""Jittor sequence ranker for Track 1.

This is a stronger model-family experiment than rank fusion. It learns a
time-aware source-history representation from each source node's recent
destination sequence, then scores candidate destinations with a two-tower
dot-product model.

The model is trained with BPR loss on temporal interactions and evaluated on
train-only splits before a submission is considered.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import jittor as jt
from jittor import nn
import numpy as np

from baseline import HistoryBaseline
from rank_utils import rank_probabilities
from validate_heuristic import add_unique_top, load_test_candidate_pools


HEURISTIC_WEIGHTS = {
    "pair_weight": 6.0,
    "pair_recency_weight": 4.0,
    "dst_pop_weight": 0.4,
    "dst_recency_weight": 0.2,
    "sequence_weight": 2.5,
    "repeat_recent_weight": 2.0,
}


class SequenceRanker(nn.Module):
    def __init__(self, num_items: int, dim: int, history_len: int) -> None:
        super().__init__()
        self.history_len = history_len
        self.item_emb = nn.Embedding(num_items + 1, dim)
        self.item_bias = nn.Embedding(num_items + 1, 1)
        self.position_emb = nn.Embedding(history_len, dim)
        self.item_emb.weight.assign(jt.randn((num_items + 1, dim)) * 0.02)
        self.position_emb.weight.assign(jt.randn((history_len, dim)) * 0.02)
        self.item_bias.weight.assign(jt.zeros((num_items + 1, 1)))

    def encode(self, histories: jt.Var, masks: jt.Var) -> jt.Var:
        positions = jt.array(np.arange(self.history_len, dtype=np.int32)).reshape((1, -1))
        vectors = self.item_emb(histories) + self.position_emb(positions)
        weights = masks.reshape((masks.shape[0], masks.shape[1], 1))
        summed = (vectors * weights).sum(dim=1)
        denom = weights.sum(dim=1) + 1e-6
        return summed / denom

    def score(self, histories: jt.Var, masks: jt.Var, items: jt.Var) -> jt.Var:
        src_vec = self.encode(histories, masks)
        item_vec = self.item_emb(items)
        return (src_vec * item_vec).sum(dim=1) + self.item_bias(items).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Jittor sequence ranker")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", default="dataset1,dataset2")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model-weight", type=float, default=1.0)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--limit-train-rows", type=int, default=0)
    parser.add_argument("--limit-test-rows", type=int, default=0)
    parser.add_argument("--train-split", choices=("all", "zero"), default="all")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--valid-samples", type=int, default=5000)
    parser.add_argument("--candidate-strategy", choices=("hard", "test_pool", "mixed"), default="test_pool")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    jt.misc.set_global_seed(seed)


def read_scene_rows(
    data_zip: zipfile.ZipFile,
    scene: str,
    limit: int = 0,
    train_split: str = "all",
) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if train_split == "zero" and row.get("split") != "0":
                continue
            rows.append((int(row["src"]), int(row["dst"]), int(row["time"])))
            if limit and len(rows) >= limit:
                break
    rows.sort(key=lambda value: value[2])
    return rows


def build_mappings(rows: list[tuple[int, int, int]], test_items: set[int]) -> dict[int, int]:
    items = {dst for _, dst, _ in rows} | test_items
    return {dst: index + 1 for index, dst in enumerate(sorted(items))}


def load_test_items(data_zip: zipfile.ZipFile, scene: str) -> set[int]:
    items: set[int] = set()
    with open_csv(data_zip, f"{scene}/test.csv") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            items.update(int(value) for value in row[2:])
    return items


def build_examples(
    rows: list[tuple[int, int, int]],
    item_to_idx: dict[int, int],
    history_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories_by_src: dict[int, list[int]] = defaultdict(list)
    histories: list[list[int]] = []
    masks: list[list[float]] = []
    positives: list[int] = []
    for src, dst, _ in rows:
        history = histories_by_src[src][-history_len:]
        padded = [0] * (history_len - len(history)) + history
        mask = [0.0] * (history_len - len(history)) + [1.0] * len(history)
        histories.append(padded)
        masks.append(mask)
        positives.append(item_to_idx[dst])
        histories_by_src[src].append(item_to_idx[dst])
    return (
        np.asarray(histories, dtype=np.int32),
        np.asarray(masks, dtype=np.float32),
        np.asarray(positives, dtype=np.int32),
    )


def train_model(
    histories: np.ndarray,
    masks: np.ndarray,
    positives: np.ndarray,
    num_items: int,
    args: argparse.Namespace,
) -> SequenceRanker:
    rng = np.random.default_rng(args.seed)
    model = SequenceRanker(num_items, args.dim, args.history_len)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    steps = max(1, math.ceil(len(positives) / args.batch_size))
    item_pool = np.arange(1, num_items + 1, dtype=np.int32)

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(positives))
        total = 0.0
        for step in range(steps):
            batch = order[step * args.batch_size:(step + 1) * args.batch_size]
            h = jt.array(histories[batch])
            m = jt.array(masks[batch])
            pos = jt.array(positives[batch])
            neg_np = rng.choice(item_pool, size=(len(batch), args.negatives_per_positive), replace=True)
            neg = jt.array(neg_np)
            pos_scores = model.score(h, m, pos)
            flat_h = h.reshape((h.shape[0], 1, h.shape[1])).broadcast((h.shape[0], args.negatives_per_positive, h.shape[1])).reshape((-1, h.shape[1]))
            flat_m = m.reshape((m.shape[0], 1, m.shape[1])).broadcast((m.shape[0], args.negatives_per_positive, m.shape[1])).reshape((-1, m.shape[1]))
            neg_scores = model.score(flat_h, flat_m, neg.reshape((-1,))).reshape(neg.shape)
            loss = nn.softplus(-(pos_scores.reshape((-1, 1)) - neg_scores)).mean()
            optimizer.step(loss)
            total += float(loss.item())
        print(f"epoch={epoch} loss={total / steps:.6f}", flush=True)
    return model


def build_history_state(
    rows: list[tuple[int, int, int]],
    item_to_idx: dict[int, int],
    history_len: int,
) -> dict[int, list[int]]:
    histories: dict[int, list[int]] = defaultdict(list)
    for src, dst, _ in rows:
        histories[src].append(item_to_idx[dst])
        if len(histories[src]) > history_len:
            histories[src] = histories[src][-history_len:]
    return histories


def build_heuristic(rows: list[tuple[int, int, int]]) -> HistoryBaseline:
    model = HistoryBaseline(**HEURISTIC_WEIGHTS)
    for src, dst, time_value in rows:
        model.update(src, dst, time_value)
    model.finalize()
    return model


def export_arrays(model: SequenceRanker) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nan_to_num(np.asarray(model.item_emb.weight.numpy(), dtype=np.float32)),
        np.nan_to_num(np.asarray(model.position_emb.weight.numpy(), dtype=np.float32)),
        np.nan_to_num(np.asarray(model.item_bias.weight.numpy(), dtype=np.float32).reshape(-1)),
    )


def sequence_score(
    src_history: list[int],
    dst: int,
    item_to_idx: dict[int, int],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    history_len: int,
) -> float:
    item_emb, pos_emb, item_bias = arrays
    dst_idx = item_to_idx.get(dst, 0)
    if dst_idx == 0:
        return 0.0
    history = src_history[-history_len:]
    if not history:
        return float(item_bias[dst_idx])
    padded = [0] * (history_len - len(history)) + history
    mask = np.asarray([0.0] * (history_len - len(history)) + [1.0] * len(history), dtype=np.float32)
    vectors = item_emb[np.asarray(padded, dtype=np.int32)] + pos_emb[np.arange(history_len)]
    src_vec = (vectors * mask[:, None]).sum(axis=0) / (mask.sum() + 1e-6)
    return float(src_vec @ item_emb[dst_idx] + item_bias[dst_idx])


def source_vector(
    src_history: list[int],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    history_len: int,
) -> np.ndarray | None:
    item_emb, pos_emb, _ = arrays
    history = src_history[-history_len:]
    if not history:
        return None
    padded = [0] * (history_len - len(history)) + history
    mask = np.asarray([0.0] * (history_len - len(history)) + [1.0] * len(history), dtype=np.float32)
    vectors = item_emb[np.asarray(padded, dtype=np.int32)] + pos_emb[np.arange(history_len)]
    return (vectors * mask[:, None]).sum(axis=0) / (mask.sum() + 1e-6)


def write_scene(
    data_zip: zipfile.ZipFile,
    output_zip: zipfile.ZipFile,
    scene: str,
    heuristic: HistoryBaseline,
    histories_by_src: dict[int, list[int]],
    item_to_idx: dict[int, int],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> int:
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
                    history = histories_by_src.get(src, [])
                    src_vec = source_vector(history, arrays, args.history_len)
                    item_emb, _, item_bias = arrays
                    candidate_indices = np.asarray([item_to_idx.get(dst, 0) for dst in candidates], dtype=np.int32)
                    if src_vec is None:
                        sequence_scores = item_bias[candidate_indices]
                    else:
                        sequence_scores = item_emb[candidate_indices] @ src_vec + item_bias[candidate_indices]
                    scores = []
                    for index, dst in enumerate(candidates):
                        base = heuristic.score(src, dst, time_value)
                        seq = float(sequence_scores[index]) if candidate_indices[index] else 0.0
                        scores.append(args.base_weight * base + args.model_weight * seq)
                    writer.writerow([f"{value:.8f}" for value in rank_probabilities(scores)])
                    rows += 1
                    if args.limit_test_rows and rows >= args.limit_test_rows:
                        break
    return rows


def validation_rows(data_zip: zipfile.ZipFile, scene: str) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    rows: list[tuple[int, int, int, str | None]] = []
    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append((int(row["src"]), int(row["dst"]), int(row["time"]), row.get("split")))
    if any(split == "1" for *_, split in rows):
        train = [(src, dst, time) for src, dst, time, split in rows if split == "0"]
        valid = [(src, dst, time) for src, dst, time, split in rows if split != "0"]
        return train, valid
    ordered = sorted(rows, key=lambda value: value[2])
    cut = int(len(ordered) * 0.9)
    return (
        [(src, dst, time) for src, dst, time, _ in ordered[:cut]],
        [(src, dst, time) for src, dst, time, _ in ordered[cut:]],
    )


def build_validation_candidates(
    positive_dst: int,
    src: int,
    strategy: str,
    all_dsts: list[int],
    popular_dsts: list[int],
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    rng: random.Random,
) -> list[int]:
    candidates = [positive_dst]
    seen = {positive_dst}

    def add_random(pool: list[int], target_size: int) -> None:
        available = [dst for dst in pool if dst not in seen]
        need = target_size - len(candidates)
        if need <= 0 or not available:
            return
        chosen = available if len(available) <= need else rng.sample(available, need)
        seen.update(chosen)
        candidates.extend(chosen)

    if strategy == "hard":
        add_random(src_test_candidates.get(src, []), 61)
        add_unique_top(candidates, seen, popular_dsts, 81)
        add_random(all_test_candidates, 100)
    elif strategy == "test_pool":
        add_random(all_test_candidates, 80)
    else:
        add_unique_top(candidates, seen, popular_dsts, 30)
    add_random(all_dsts, 100)
    if len(candidates) != 100:
        raise RuntimeError(f"Only built {len(candidates)} candidates")
    rng.shuffle(candidates)
    return candidates


def reciprocal_rank(candidates: list[int], scores: list[float], positive_dst: int) -> float:
    for rank, index in enumerate(sorted(range(len(scores)), key=lambda value: scores[value], reverse=True), start=1):
        if candidates[index] == positive_dst:
            return 1.0 / rank
    return 0.0


def evaluate_sequence_model(
    data_zip: zipfile.ZipFile,
    scene: str,
    train_rows: list[tuple[int, int, int]],
    valid_rows: list[tuple[int, int, int]],
    heuristic: HistoryBaseline,
    histories_by_src: dict[int, list[int]],
    item_to_idx: dict[int, int],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> None:
    rng = random.Random(args.seed)
    if args.valid_samples and args.valid_samples < len(valid_rows):
        valid_rows = rng.sample(valid_rows, args.valid_samples)
    all_dsts = sorted({dst for _, dst, _ in train_rows})
    popular_dsts = [dst for dst, _ in Counter(dst for _, dst, _ in train_rows).most_common(10000)]
    src_test_candidates, all_test_candidates = load_test_candidate_pools(data_zip, scene)
    base_rr = 0.0
    seq_rr = 0.0
    blend_rr = 0.0
    for src, positive_dst, time_value in valid_rows:
        candidates = build_validation_candidates(
            positive_dst,
            src,
            args.candidate_strategy,
            all_dsts,
            popular_dsts,
            src_test_candidates,
            all_test_candidates,
            rng,
        )
        history = histories_by_src.get(src, [])
        src_vec = source_vector(history, arrays, args.history_len)
        item_emb, _, item_bias = arrays
        candidate_indices = np.asarray([item_to_idx.get(dst, 0) for dst in candidates], dtype=np.int32)
        if src_vec is None:
            sequence_scores = item_bias[candidate_indices]
        else:
            sequence_scores = item_emb[candidate_indices] @ src_vec + item_bias[candidate_indices]
        base_scores = [heuristic.score(src, dst, time_value) for dst in candidates]
        seq_scores = [float(value) if candidate_indices[index] else 0.0 for index, value in enumerate(sequence_scores)]
        blend_scores = [base + args.model_weight * seq for base, seq in zip(base_scores, seq_scores)]
        base_rr += reciprocal_rank(candidates, base_scores, positive_dst)
        seq_rr += reciprocal_rank(candidates, seq_scores, positive_dst)
        blend_rr += reciprocal_rank(candidates, blend_scores, positive_dst)
    n = len(valid_rows)
    print(
        f"[{scene}] validation strategy={args.candidate_strategy} "
        f"base_mrr={base_rr / n:.8f} seq_mrr={seq_rr / n:.8f} blend_mrr={blend_rr / n:.8f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    jt.flags.use_cuda = 0 if args.cpu or not jt.has_cuda else 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.data_zip) as data_zip:
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
            for scene in [scene.strip() for scene in args.scenes.split(",") if scene.strip()]:
                print(f"[{scene}] loading", flush=True)
                if args.validate:
                    rows, valid_rows = validation_rows(data_zip, scene)
                    if args.limit_train_rows:
                        rows = rows[:args.limit_train_rows]
                else:
                    rows = read_scene_rows(data_zip, scene, args.limit_train_rows, args.train_split)
                    valid_rows = []
                item_to_idx = build_mappings(rows, load_test_items(data_zip, scene))
                histories, masks, positives = build_examples(rows, item_to_idx, args.history_len)
                print(f"[{scene}] examples={len(positives)} items={len(item_to_idx)} cuda={jt.flags.use_cuda}", flush=True)
                model = train_model(histories, masks, positives, len(item_to_idx), args)
                arrays = export_arrays(model)
                heuristic = build_heuristic(rows)
                history_state = build_history_state(rows, item_to_idx, args.history_len)
                if args.validate:
                    evaluate_sequence_model(
                        data_zip,
                        scene,
                        rows,
                        valid_rows,
                        heuristic,
                        history_state,
                        item_to_idx,
                        arrays,
                        args,
                    )
                count = write_scene(data_zip, output_zip, scene, heuristic, history_state, item_to_idx, arrays, args)
                print(f"[{scene}] wrote {count} rows", flush=True)
    print(f"submission saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
