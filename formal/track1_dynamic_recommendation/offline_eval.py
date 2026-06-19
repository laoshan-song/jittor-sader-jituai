"""Offline proxy evaluation for Track 1 scoring functions.

This is a labeled validation harness built from train.csv only. It evaluates a
scorer on several time splits and candidate-generation strategies so that a
single misleading proxy is less likely to dominate decisions.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from baseline import HistoryBaseline
from temporal_motif_rerank import TemporalMotifReranker
from validate_heuristic import add_unique_top, load_test_candidate_pools


DEFAULT_WEIGHTS = {
    "pair_weight": 6.0,
    "pair_recency_weight": 4.0,
    "dst_pop_weight": 0.4,
    "dst_recency_weight": 0.2,
    "sequence_weight": 2.5,
    "repeat_recent_weight": 2.0,
}


@dataclass
class EvalResult:
    scene: str
    split: str
    candidate_strategy: str
    positives: int
    mrr: float
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    mean_rank: float
    repeat_pair_ratio: float
    seen_dst_ratio: float
    median_positive_dst_count: float


@dataclass
class EvalContext:
    scene: str
    split: str
    model: HistoryBaseline
    valid_rows: list[tuple[int, int, int]]
    all_dsts: list[int]
    popular_dsts: list[int]
    src_test_candidates: dict[int, list[int]]
    all_test_candidates: list[int]
    train_pairs: set[tuple[int, int]]
    train_dsts: set[int]
    dst_count: Counter[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run labeled offline proxy evaluation")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--scenes", default="dataset1,dataset2")
    parser.add_argument("--splits", default="temporal,official")
    parser.add_argument("--candidate-strategies", default="random,popular,mixed,test_pool,hard")
    parser.add_argument("--sample-positives", type=int, default=20000)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--scorer", choices=("history", "motif"), default="history")
    parser.add_argument("--reverse-weight", type=float, default=0.0)
    parser.add_argument("--reverse-recency-weight", type=float, default=0.0)
    parser.add_argument("--co-motif-weight", type=float, default=0.0)
    parser.add_argument("--co-motif-recent-weight", type=float, default=0.0)
    parser.add_argument("--source-pop-weight", type=float, default=0.0)
    parser.add_argument("--local-repeat-boost", type=float, default=0.0)
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


def split_rows(
    rows: list[tuple[int, int, int, str | None]],
    split_name: str,
    valid_fraction: float,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    if split_name == "official":
        if not any(split == "1" for *_, split in rows):
            return [], []
        train = [(src, dst, time) for src, dst, time, split in rows if split == "0"]
        valid = [(src, dst, time) for src, dst, time, split in rows if split != "0"]
        return train, valid

    if split_name == "temporal":
        ordered = sorted(rows, key=lambda value: value[2])
        cut = int(len(ordered) * (1.0 - valid_fraction))
        train = [(src, dst, time) for src, dst, time, _ in ordered[:cut]]
        valid = [(src, dst, time) for src, dst, time, _ in ordered[cut:]]
        return train, valid

    raise ValueError(f"Unknown split: {split_name}")


def build_model(rows: list[tuple[int, int, int]], args: argparse.Namespace) -> HistoryBaseline | TemporalMotifReranker:
    if args.scorer == "motif":
        model = TemporalMotifReranker(
            reverse_weight=args.reverse_weight,
            reverse_recency_weight=args.reverse_recency_weight,
            co_motif_weight=args.co_motif_weight,
            co_motif_recent_weight=args.co_motif_recent_weight,
            source_pop_weight=args.source_pop_weight,
            local_repeat_boost=args.local_repeat_boost,
        )
    else:
        model = HistoryBaseline(**DEFAULT_WEIGHTS)
    for src, dst, time_value in rows:
        model.update(src, dst, time_value)
    model.finalize()
    return model


def build_candidates(
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
        if len(available) <= need:
            chosen = available
        else:
            chosen = rng.sample(available, need)
        seen.update(chosen)
        candidates.extend(chosen)

    if strategy == "random":
        add_random(all_dsts, 100)
    elif strategy == "popular":
        add_unique_top(candidates, seen, popular_dsts, 80)
        add_random(all_dsts, 100)
    elif strategy == "mixed":
        add_unique_top(candidates, seen, popular_dsts, 30)
        add_random(all_dsts, 100)
    elif strategy == "test_pool":
        add_random(all_test_candidates, 80)
        add_random(all_dsts, 100)
    elif strategy == "hard":
        add_random(src_test_candidates.get(src, []), 61)
        add_unique_top(candidates, seen, popular_dsts, 81)
        add_random(all_test_candidates, 100)
        add_random(all_dsts, 100)
    else:
        raise ValueError(f"Unknown candidate strategy: {strategy}")

    if len(candidates) != 100:
        raise RuntimeError(f"Only built {len(candidates)} candidates for strategy={strategy}")
    rng.shuffle(candidates)
    return candidates


def reciprocal_rank(
    model: HistoryBaseline,
    src: int,
    positive_dst: int,
    time_value: int,
    candidates: list[int],
) -> tuple[float, int]:
    scored = [(model.score(src, dst, time_value), dst) for dst in candidates]
    scored.sort(reverse=True)
    for rank, (_, dst) in enumerate(scored, start=1):
        if dst == positive_dst:
            return 1.0 / rank, rank
    return 0.0, 101


def build_context(
    data_zip: zipfile.ZipFile,
    scene: str,
    split_name: str,
    args: argparse.Namespace,
) -> EvalContext | None:
    rng = random.Random(args.seed)
    rows = read_rows(data_zip, scene)
    train_rows, valid_rows = split_rows(rows, split_name, args.valid_fraction)
    if not train_rows or not valid_rows:
        return None
    if args.sample_positives and args.sample_positives < len(valid_rows):
        valid_rows = rng.sample(valid_rows, args.sample_positives)

    model = build_model(train_rows, args)
    all_dsts = sorted({dst for _, dst, _ in train_rows})
    popular_dsts = [dst for dst, _ in Counter(dst for _, dst, _ in train_rows).most_common(10000)]
    src_test_candidates, all_test_candidates = load_test_candidate_pools(data_zip, scene)

    train_pairs = {(src, dst) for src, dst, _ in train_rows}
    train_dsts = {dst for _, dst, _ in train_rows}
    dst_count = Counter(dst for _, dst, _ in train_rows)

    return EvalContext(
        scene=scene,
        split=split_name,
        model=model,
        valid_rows=valid_rows,
        all_dsts=all_dsts,
        popular_dsts=popular_dsts,
        src_test_candidates=src_test_candidates,
        all_test_candidates=all_test_candidates,
        train_pairs=train_pairs,
        train_dsts=train_dsts,
        dst_count=dst_count,
    )


def evaluate_strategy(context: EvalContext, strategy: str, seed: int) -> EvalResult:
    rng = random.Random(seed)
    rr_total = 0.0
    hit1 = 0
    hit5 = 0
    hit10 = 0
    ranks: list[int] = []
    repeat_pair = 0
    seen_dst = 0
    positive_dst_counts: list[int] = []

    for src, positive_dst, time_value in context.valid_rows:
        candidates = build_candidates(
            positive_dst,
            src,
            strategy,
            context.all_dsts,
            context.popular_dsts,
            context.src_test_candidates,
            context.all_test_candidates,
            rng,
        )
        rr, rank = reciprocal_rank(context.model, src, positive_dst, time_value, candidates)
        rr_total += rr
        hit1 += int(rank <= 1)
        hit5 += int(rank <= 5)
        hit10 += int(rank <= 10)
        ranks.append(rank)
        repeat_pair += int((src, positive_dst) in context.train_pairs)
        seen_dst += int(positive_dst in context.train_dsts)
        positive_dst_counts.append(context.dst_count[positive_dst])

    n = len(context.valid_rows)
    return EvalResult(
        scene=context.scene,
        split=context.split,
        candidate_strategy=strategy,
        positives=n,
        mrr=rr_total / n,
        hit_at_1=hit1 / n,
        hit_at_5=hit5 / n,
        hit_at_10=hit10 / n,
        mean_rank=statistics.fmean(ranks),
        repeat_pair_ratio=repeat_pair / n,
        seen_dst_ratio=seen_dst / n,
        median_positive_dst_count=statistics.median(positive_dst_counts),
    )


def print_results(results: list[EvalResult]) -> None:
    print(
        "scene,split,candidate_strategy,positives,mrr,hit@1,hit@5,hit@10,"
        "mean_rank,repeat_pair,seen_dst,median_pos_dst_count"
    )
    for result in results:
        print(
            f"{result.scene},{result.split},{result.candidate_strategy},{result.positives},"
            f"{result.mrr:.8f},{result.hit_at_1:.8f},{result.hit_at_5:.8f},"
            f"{result.hit_at_10:.8f},{result.mean_rank:.4f},"
            f"{result.repeat_pair_ratio:.4f},{result.seen_dst_ratio:.4f},"
            f"{result.median_positive_dst_count:.1f}"
        )


def main() -> None:
    args = parse_args()
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    strategies = [strategy.strip() for strategy in args.candidate_strategies.split(",") if strategy.strip()]

    results: list[EvalResult] = []
    with zipfile.ZipFile(args.data_zip) as data_zip:
        for scene in scenes:
            for split_name in splits:
                context = build_context(
                    data_zip,
                    scene,
                    split_name,
                    args,
                )
                if context is None:
                    continue
                for strategy in strategies:
                    results.append(evaluate_strategy(context, strategy, args.seed))

    print_results(results)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
