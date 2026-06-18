"""Local MRR validation for Track 1 heuristic ranking.

This script uses `dataset2/train.csv` because it contains an official-like
time split. Rows with `split=0` are used as history, and rows with `split=1`
are treated as positive future interactions. For each positive edge, the
script samples negative candidates and computes MRR@100.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from baseline import HistoryBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Track 1 heuristic MRR")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--scene", default="dataset2")
    parser.add_argument("--sample-positives", type=int, default=20000)
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("--src-test-negatives", type=int, default=60)
    parser.add_argument("--popular-negatives", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pair-weight", type=float, default=4.0)
    parser.add_argument("--pair-recency-weight", type=float, default=3.0)
    parser.add_argument("--dst-pop-weight", type=float, default=0.8)
    parser.add_argument("--dst-recency-weight", type=float, default=0.4)
    parser.add_argument("--sequence-weight", type=float, default=1.2)
    parser.add_argument("--repeat-recent-weight", type=float, default=1.0)
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def build_model_and_holdout(
    data_zip: zipfile.ZipFile,
    scene: str,
    args: argparse.Namespace,
) -> tuple[HistoryBaseline, list[tuple[int, int, int]], list[int]]:
    model = HistoryBaseline(
        pair_weight=args.pair_weight,
        pair_recency_weight=args.pair_recency_weight,
        dst_pop_weight=args.dst_pop_weight,
        dst_recency_weight=args.dst_recency_weight,
        sequence_weight=args.sequence_weight,
        repeat_recent_weight=args.repeat_recent_weight,
    )
    positives: list[tuple[int, int, int]] = []
    all_dsts: set[int] = set()

    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            src = int(row["src"])
            dst = int(row["dst"])
            time_value = int(row["time"])
            all_dsts.add(dst)
            if row.get("split", "0") == "0":
                model.update(src, dst, time_value)
            else:
                positives.append((src, dst, time_value))

    model.finalize()
    return model, positives, sorted(all_dsts)


def load_test_candidate_pools(
    data_zip: zipfile.ZipFile,
    scene: str,
) -> tuple[dict[int, list[int]], list[int]]:
    src_candidates: dict[int, set[int]] = defaultdict(set)
    all_candidates: set[int] = set()

    with open_csv(data_zip, f"{scene}/test.csv") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            src = int(row[0])
            candidates = [int(value) for value in row[2:]]
            src_candidates[src].update(candidates)
            all_candidates.update(candidates)

    return {src: sorted(values) for src, values in src_candidates.items()}, sorted(all_candidates)


def add_unique_random(
    candidates: list[int],
    seen: set[int],
    pool: list[int],
    target_size: int,
    rng: random.Random,
) -> None:
    if not pool:
        return

    attempts = 0
    max_attempts = max(1000, target_size * 100)
    while len(candidates) < target_size and attempts < max_attempts:
        attempts += 1
        dst = rng.choice(pool)
        if dst not in seen:
            seen.add(dst)
            candidates.append(dst)


def add_unique_top(
    candidates: list[int],
    seen: set[int],
    pool: list[int],
    target_size: int,
) -> None:
    for dst in pool:
        if len(candidates) >= target_size:
            return
        if dst not in seen:
            seen.add(dst)
            candidates.append(dst)


def sampled_candidates(
    positive_dst: int,
    all_dsts: list[int],
    src_pool: list[int],
    all_test_candidates: list[int],
    popular_dsts: list[int],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[int]:
    candidates = [positive_dst]
    seen = {positive_dst}

    src_target = min(args.negatives + 1, 1 + args.src_test_negatives)
    add_unique_random(candidates, seen, src_pool, src_target, rng)

    pop_target = min(args.negatives + 1, len(candidates) + args.popular_negatives)
    add_unique_top(candidates, seen, popular_dsts, pop_target)

    add_unique_random(candidates, seen, all_test_candidates, args.negatives + 1, rng)
    add_unique_random(candidates, seen, all_dsts, args.negatives + 1, rng)

    if len(candidates) != args.negatives + 1:
        raise RuntimeError(f"Only built {len(candidates)} candidates.")

    rng.shuffle(candidates)
    return candidates


def reciprocal_rank(
    model: HistoryBaseline,
    src: int,
    positive_dst: int,
    time_value: int,
    candidates: list[int],
) -> float:
    scored = [(model.score(src, dst, time_value), dst) for dst in candidates]
    scored.sort(reverse=True)
    for rank, (_, dst) in enumerate(scored, start=1):
        if dst == positive_dst:
            return 1.0 / rank
    return 0.0


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    with zipfile.ZipFile(args.data_zip) as data_zip:
        model, positives, all_dsts = build_model_and_holdout(data_zip, args.scene, args)
        src_test_candidates, all_test_candidates = load_test_candidate_pools(
            data_zip,
            args.scene,
        )

    if args.sample_positives and args.sample_positives < len(positives):
        positives = rng.sample(positives, args.sample_positives)

    popular_dsts = [
        dst
        for dst, _ in Counter(model.dst_count).most_common(max(args.popular_negatives * 50, 5000))
    ]

    total_rr = 0.0
    hit_at_1 = 0
    hit_at_10 = 0
    for index, (src, positive_dst, time_value) in enumerate(positives, start=1):
        candidates = sampled_candidates(
            positive_dst,
            all_dsts,
            src_test_candidates.get(src, []),
            all_test_candidates,
            popular_dsts,
            args,
            rng,
        )
        rr = reciprocal_rank(model, src, positive_dst, time_value, candidates)
        total_rr += rr
        if rr == 1.0:
            hit_at_1 += 1
        if rr >= 0.1:
            hit_at_10 += 1
        if index % 5000 == 0:
            print(f"processed={index} mrr={total_rr / index:.6f}")

    n = len(positives)
    print(f"scene={args.scene}")
    print(f"positives={n}")
    print(f"mrr={total_rr / n:.8f}")
    print(f"hit@1={hit_at_1 / n:.8f}")
    print(f"hit@10={hit_at_10 / n:.8f}")
    print(
        "weights="
        f"{args.pair_weight},"
        f"{args.pair_recency_weight},"
        f"{args.dst_pop_weight},"
        f"{args.dst_recency_weight},"
        f"{args.sequence_weight},"
        f"{args.repeat_recent_weight}"
    )


if __name__ == "__main__":
    main()
