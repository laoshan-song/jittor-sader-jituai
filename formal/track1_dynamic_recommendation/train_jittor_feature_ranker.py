"""Jittor feature reranker for Track 1 candidates.

This script trains a compact supervised reranker on dataset2's official split.
It uses deterministic temporal-graph features inspired by TGN/TGAT/JODIE/CAW
and optimizes candidate ranking directly with binary cross entropy.

The model is intentionally small: a linear scorer over handcrafted temporal
features. That keeps code review reproducible while letting the validation
split decide whether any learned signal is strong enough to justify a
leaderboard submission.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
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


@dataclass
class FeatureContext:
    model: HistoryBaseline
    rows: list[tuple[int, int, int]]
    all_dsts: list[int]
    popular_dsts: list[int]
    train_pairs: set[tuple[int, int]]
    src_count: Counter[int]
    src_last_time: dict[int, int]
    co_motif: dict[int, Counter[int]]
    max_pair_log: float
    max_dst_log: float
    max_src_log: float
    max_co_log: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Jittor feature reranker")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--train-positives", type=int, default=80000)
    parser.add_argument("--valid-positives", type=int, default=30000)
    parser.add_argument("--negatives", type=int, default=24)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--learned-weight", type=float, default=1.0)
    parser.add_argument("--candidate-strategy", choices=("official", "hard"), default="official")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def read_rows(data_zip: zipfile.ZipFile, scene: str) -> list[tuple[int, int, int, str | None]]:
    rows: list[tuple[int, int, int, str | None]] = []
    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append((int(row["src"]), int(row["dst"]), int(row["time"]), row.get("split")))
    return rows


def build_context(rows: list[tuple[int, int, int]], all_dsts: list[int]) -> FeatureContext:
    model = HistoryBaseline(**HEURISTIC_WEIGHTS)
    src_count: Counter[int] = Counter()
    src_last_time: dict[int, int] = {}
    for src, dst, time_value in rows:
        model.update(src, dst, time_value)
        src_count[src] += 1
        src_last_time[src] = max(time_value, src_last_time.get(src, time_value))
    model.finalize()

    co_motif: dict[int, Counter[int]] = defaultdict(Counter)
    max_co_count = 1
    for history in model.src_history.values():
        history.sort()
        sequence = [dst for _, dst in history]
        for index, dst in enumerate(sequence):
            for prev_dst in sequence[max(0, index - 8):index]:
                if prev_dst == dst:
                    continue
                co_motif[prev_dst][dst] += 1
                max_co_count = max(max_co_count, co_motif[prev_dst][dst])

    return FeatureContext(
        model=model,
        rows=rows,
        all_dsts=all_dsts,
        popular_dsts=[dst for dst, _ in model.dst_count.most_common(10000)],
        train_pairs={(src, dst) for src, dst, _ in rows},
        src_count=src_count,
        src_last_time=src_last_time,
        co_motif=co_motif,
        max_pair_log=max((math.log1p(v) for counts in model.src_dst_count.values() for v in counts.values()), default=1.0),
        max_dst_log=max((math.log1p(v) for v in model.dst_count.values()), default=1.0),
        max_src_log=max((math.log1p(v) for v in src_count.values()), default=1.0),
        max_co_log=math.log1p(max_co_count),
    )


def row_features(context: FeatureContext, src: int, dst: int, query_time: int) -> list[float]:
    model = context.model
    src_counts = model.src_dst_count.get(src, {})
    pair_count = src_counts.get(dst, 0)
    pair_last = model.src_dst_last_time.get(src, {}).get(dst)
    dst_count = model.dst_count.get(dst, 0)
    dst_last = model.dst_last_time.get(dst)
    reverse_count = model.src_dst_count.get(dst, {}).get(src, 0)
    reverse_last = model.src_dst_last_time.get(dst, {}).get(src)
    recent_dsts = model.src_recent_dsts.get(src, [])

    motif = 0.0
    motif_recent = 0.0
    for offset, prev_dst in enumerate(reversed(recent_dsts[-12:]), start=1):
        count = context.co_motif.get(prev_dst, {}).get(dst, 0)
        if count:
            value = math.log1p(count) / context.max_co_log
            motif += value / math.sqrt(offset)
            motif_recent += value / offset

    repeat_distance = 0.0
    if dst in recent_dsts[-20:]:
        distance = len(recent_dsts) - 1 - max(index for index, value in enumerate(recent_dsts) if value == dst)
        repeat_distance = 1.0 / (1.0 + distance)

    base_score = model.score(src, dst, query_time)
    return [
        base_score / 20.0,
        math.log1p(pair_count) / context.max_pair_log if pair_count else 0.0,
        model._recency(query_time, pair_last),
        math.log1p(dst_count) / context.max_dst_log if dst_count else 0.0,
        model._recency(query_time, dst_last),
        math.log1p(reverse_count) / context.max_pair_log if reverse_count else 0.0,
        model._recency(query_time, reverse_last),
        math.log1p(context.src_count.get(dst, 0)) / context.max_src_log if dst in context.src_count else 0.0,
        motif,
        motif_recent,
        repeat_distance,
        1.0 if (src, dst) in context.train_pairs else 0.0,
    ]


def sampled_candidates(
    positive_dst: int,
    src: int,
    context: FeatureContext,
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    negatives: int,
    strategy: str,
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

    if strategy == "official":
        add_random(src_test_candidates.get(src, []), 1 + negatives // 2)
        add_unique_top(candidates, seen, context.popular_dsts, 1 + negatives * 3 // 4)
        add_random(all_test_candidates, 1 + negatives)
    else:
        add_random(src_test_candidates.get(src, []), 1 + negatives * 2 // 3)
        add_unique_top(candidates, seen, context.popular_dsts, 1 + negatives)

    add_random(context.all_dsts, 1 + negatives)
    if len(candidates) != 1 + negatives:
        raise RuntimeError(f"Only built {len(candidates)} candidates")
    rng.shuffle(candidates)
    return candidates


def build_examples(
    context: FeatureContext,
    positives: list[tuple[int, int, int]],
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    negatives: int,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    features: list[list[float]] = []
    labels: list[float] = []
    for src, dst, time_value in positives:
        candidates = sampled_candidates(
            dst,
            src,
            context,
            src_test_candidates,
            all_test_candidates,
            negatives,
            strategy,
            rng,
        )
        for candidate in candidates:
            features.append(row_features(context, src, candidate, time_value))
            labels.append(1.0 if candidate == dst else 0.0)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32).reshape(-1, 1)


def train_linear(features: np.ndarray, labels: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    jt.flags.use_cuda = 0 if args.cpu or not jt.has_cuda else 1
    weight = jt.zeros((features.shape[1], 1))
    bias = jt.zeros((1,))
    weight.start_grad()
    bias.start_grad()
    optimizer = nn.Adam([weight, bias], lr=args.lr)
    rng = np.random.default_rng(args.seed)
    steps = max(1, math.ceil(len(features) / args.batch_size))

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(features))
        total = 0.0
        for step in range(steps):
            batch = order[step * args.batch_size:(step + 1) * args.batch_size]
            x = jt.array(features[batch])
            y = jt.array(labels[batch])
            logits = x @ weight + bias
            loss = nn.binary_cross_entropy_with_logits(logits, y)
            optimizer.step(loss)
            total += float(loss.item())
        print(f"epoch={epoch} loss={total / steps:.6f}", flush=True)

    return np.asarray(weight.numpy()).reshape(-1), float(bias.numpy()[0])


def evaluate_mrr(
    context: FeatureContext,
    positives: list[tuple[int, int, int]],
    weight: np.ndarray,
    bias: float,
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    args: argparse.Namespace,
) -> tuple[float, float]:
    rng = random.Random(args.seed + 17)
    base_rr = 0.0
    learned_rr = 0.0
    for src, dst, time_value in positives:
        candidates = sampled_candidates(
            dst,
            src,
            context,
            src_test_candidates,
            all_test_candidates,
            99,
            args.candidate_strategy,
            rng,
        )
        base_scores = [context.model.score(src, cand, time_value) for cand in candidates]
        learned_scores = [
            base + args.learned_weight * float(np.asarray(row_features(context, src, cand, time_value)) @ weight + bias)
            for base, cand in zip(base_scores, candidates)
        ]
        base_rr += reciprocal_rank(candidates, base_scores, dst)
        learned_rr += reciprocal_rank(candidates, learned_scores, dst)
    n = len(positives)
    return base_rr / n, learned_rr / n


def reciprocal_rank(candidates: list[int], scores: list[float], positive_dst: int) -> float:
    for rank, index in enumerate(sorted(range(len(scores)), key=lambda value: scores[value], reverse=True), start=1):
        if candidates[index] == positive_dst:
            return 1.0 / rank
    return 0.0


def write_submission(
    data_zip: zipfile.ZipFile,
    output_path: Path,
    d1_context: FeatureContext,
    d2_context: FeatureContext,
    weight: np.ndarray,
    bias: float,
    args: argparse.Namespace,
) -> None:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        for scene, context in (("dataset1", d1_context), ("dataset2", d2_context)):
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
                            scores = []
                            for dst in candidates:
                                base = context.model.score(src, dst, time_value)
                                learned = float(np.asarray(row_features(context, src, dst, time_value)) @ weight + bias)
                                scores.append(args.base_weight * base + args.learned_weight * learned)
                            probs = rank_probabilities(scores)
                            writer.writerow([f"{value:.8f}" for value in probs])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.misc.set_global_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.data_zip) as data_zip:
        rows = read_rows(data_zip, "dataset2")
        train_rows = [(src, dst, time) for src, dst, time, split in rows if split == "0"]
        valid_rows = [(src, dst, time) for src, dst, time, split in rows if split != "0"]
        rng = random.Random(args.seed)
        rng.shuffle(valid_rows)
        train_pos = valid_rows[:args.train_positives]
        valid_pos = valid_rows[args.train_positives:args.train_positives + args.valid_positives]
        context = build_context(train_rows, sorted({dst for _, dst, _ in train_rows}))
        src_test_candidates, all_test_candidates = load_test_candidate_pools(data_zip, "dataset2")

        print(f"building examples positives={len(train_pos)} negatives={args.negatives}", flush=True)
        features, labels = build_examples(
            context,
            train_pos,
            src_test_candidates,
            all_test_candidates,
            args.negatives,
            args.candidate_strategy,
            args.seed,
        )
        weight, bias = train_linear(features, labels, args)
        base_mrr, learned_mrr = evaluate_mrr(
            context,
            valid_pos,
            weight,
            bias,
            src_test_candidates,
            all_test_candidates,
            args,
        )
        print(f"validation base_mrr={base_mrr:.8f} learned_mrr={learned_mrr:.8f}", flush=True)
        print("weights=" + ",".join(f"{value:.6f}" for value in weight) + f" bias={bias:.6f}", flush=True)

        full_d2_rows = [(src, dst, time) for src, dst, time, _ in rows]
        d2_context = build_context(full_d2_rows, sorted({dst for _, dst, _ in full_d2_rows}))
        d1_rows = [(src, dst, time) for src, dst, time, _ in read_rows(data_zip, "dataset1")]
        d1_context = build_context(d1_rows, sorted({dst for _, dst, _ in d1_rows}))
        write_submission(data_zip, args.output, d1_context, d2_context, weight, bias, args)
    print(f"submission saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
