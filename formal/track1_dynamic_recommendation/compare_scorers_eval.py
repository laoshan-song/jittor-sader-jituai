"""Compare Track 1 scorers on matched local validation cases.

This script is stricter than the old proxy because it compares a candidate
scorer against the current strong baseline on exactly the same sampled cases,
across dataset1 temporal split and dataset2 official/temporal splits.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from baseline import HistoryBaseline
from calibrated_rerank import calibrated_score
from offline_eval import build_candidates, split_rows
from validate_heuristic import load_test_candidate_pools


HEURISTIC_WEIGHTS = {
    "pair_weight": 6.0,
    "pair_recency_weight": 4.0,
    "dst_pop_weight": 0.4,
    "dst_recency_weight": 0.2,
    "sequence_weight": 2.5,
    "repeat_recent_weight": 2.0,
}


@dataclass
class CompareResult:
    scene: str
    split: str
    candidate_strategy: str
    positives: int
    baseline_mrr: float
    candidate_mrr: float
    relative_gain: float
    baseline_hit1: float
    candidate_hit1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and calibrated scorer")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--scenes", default="dataset1,dataset2")
    parser.add_argument("--splits", default="temporal,official")
    parser.add_argument("--candidate-strategies", default="hard,test_pool,mixed")
    parser.add_argument("--sample-positives", type=int, default=5000)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pair-penalty", type=float, default=-2.0)
    parser.add_argument("--dst-pop-penalty", type=float, default=-1.0)
    parser.add_argument("--dst-recency-weight", type=float, default=3.0)
    parser.add_argument("--local-repeat-penalty", type=float, default=-2.0)
    parser.add_argument("--output-json", type=Path)
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


def fit_model(rows: list[tuple[int, int, int]]) -> HistoryBaseline:
    model = HistoryBaseline(**HEURISTIC_WEIGHTS)
    for src, dst, time_value in rows:
        model.update(src, dst, time_value)
    model.finalize()
    return model


def reciprocal_rank(candidates: list[int], scores: list[float], positive_dst: int) -> tuple[float, int]:
    for rank, index in enumerate(sorted(range(len(scores)), key=lambda value: scores[value], reverse=True), start=1):
        if candidates[index] == positive_dst:
            return 1.0 / rank, rank
    return 0.0, 101


def compare_one(
    data_zip: zipfile.ZipFile,
    scene: str,
    split_name: str,
    strategy: str,
    args: argparse.Namespace,
) -> CompareResult | None:
    rows = read_rows(data_zip, scene)
    train_rows, valid_rows = split_rows(rows, split_name, args.valid_fraction)
    if not train_rows or not valid_rows:
        return None
    rng = random.Random(args.seed + hash((scene, split_name, strategy)) % 100000)
    if args.sample_positives and args.sample_positives < len(valid_rows):
        valid_rows = rng.sample(valid_rows, args.sample_positives)

    model = fit_model(train_rows)
    all_dsts = sorted({dst for _, dst, _ in train_rows})
    popular_dsts = [dst for dst, _ in Counter(dst for _, dst, _ in train_rows).most_common(10000)]
    src_test_candidates, all_test_candidates = load_test_candidate_pools(data_zip, scene)

    baseline_rr = 0.0
    candidate_rr = 0.0
    baseline_hit1 = 0
    candidate_hit1 = 0
    for src, positive_dst, time_value in valid_rows:
        candidates = build_candidates(
            positive_dst,
            src,
            strategy,
            all_dsts,
            popular_dsts,
            src_test_candidates,
            all_test_candidates,
            rng,
        )
        baseline_scores = [model.score(src, dst, time_value) for dst in candidates]
        candidate_scores = [calibrated_score(model, src, dst, time_value, args) for dst in candidates]
        rr, rank = reciprocal_rank(candidates, baseline_scores, positive_dst)
        baseline_rr += rr
        baseline_hit1 += int(rank == 1)
        rr, rank = reciprocal_rank(candidates, candidate_scores, positive_dst)
        candidate_rr += rr
        candidate_hit1 += int(rank == 1)

    n = len(valid_rows)
    baseline_mrr = baseline_rr / n
    candidate_mrr = candidate_rr / n
    return CompareResult(
        scene=scene,
        split=split_name,
        candidate_strategy=strategy,
        positives=n,
        baseline_mrr=baseline_mrr,
        candidate_mrr=candidate_mrr,
        relative_gain=(candidate_mrr / baseline_mrr - 1.0) if baseline_mrr else 0.0,
        baseline_hit1=baseline_hit1 / n,
        candidate_hit1=candidate_hit1 / n,
    )


def main() -> None:
    args = parse_args()
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    strategies = [strategy.strip() for strategy in args.candidate_strategies.split(",") if strategy.strip()]
    results: list[CompareResult] = []
    with zipfile.ZipFile(args.data_zip) as data_zip:
        for scene in scenes:
            for split_name in splits:
                for strategy in strategies:
                    result = compare_one(data_zip, scene, split_name, strategy, args)
                    if result:
                        results.append(result)

    print("scene,split,strategy,n,baseline_mrr,candidate_mrr,relative_gain,baseline_hit1,candidate_hit1")
    for result in results:
        print(
            f"{result.scene},{result.split},{result.candidate_strategy},{result.positives},"
            f"{result.baseline_mrr:.8f},{result.candidate_mrr:.8f},{result.relative_gain:.6f},"
            f"{result.baseline_hit1:.8f},{result.candidate_hit1:.8f}"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
