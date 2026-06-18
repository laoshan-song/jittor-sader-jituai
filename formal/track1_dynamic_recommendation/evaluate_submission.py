"""Audit Track 1 submission zips on official test candidates.

This script has no hidden labels, so it does not estimate leaderboard MRR. It
checks whether a submission is well formed, whether probabilities preserve
enough ranking signal after 8-decimal rounding, and how far rankings drift from
a trusted baseline submission.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


SCENES = ("dataset1", "dataset2")


@dataclass
class SceneAudit:
    rows: int
    expected_rows: int
    bad_rows: int
    avg_distinct: float
    median_distinct: float
    avg_min_tie: float
    avg_max_probability: float
    avg_normalized_entropy: float
    min_sum: float
    max_sum: float
    top1_seen_dst_ratio: float
    top1_repeated_pair_ratio: float
    top10_seen_dst_ratio: float
    top10_repeated_pair_ratio: float


@dataclass
class AgreementAudit:
    top1_agreement: float
    top10_jaccard: float


@dataclass
class SceneContext:
    expected_rows: int
    train_dsts: set[int]
    train_pairs: set[tuple[int, int]]
    test_rows: list[tuple[int, list[int]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Track 1 submission zips")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--submissions", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline-zip", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--limit-rows", type=int, default=0)
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def load_train_sets(data_zip: zipfile.ZipFile, scene: str) -> tuple[set[int], set[tuple[int, int]]]:
    dsts: set[int] = set()
    pairs: set[tuple[int, int]] = set()
    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            src = int(row["src"])
            dst = int(row["dst"])
            dsts.add(dst)
            pairs.add((src, dst))
    return dsts, pairs


def load_test_rows(
    data_zip: zipfile.ZipFile,
    scene: str,
    limit_rows: int,
) -> list[tuple[int, list[int]]]:
    rows: list[tuple[int, list[int]]] = []
    with open_csv(data_zip, f"{scene}/test.csv") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            rows.append((int(row[0]), [int(value) for value in row[2:]]))
            if limit_rows and len(rows) >= limit_rows:
                break
    return rows


def probability_entropy(values: list[float]) -> float:
    if not values:
        return 0.0
    entropy = 0.0
    for value in values:
        if value > 0:
            entropy -= value * math.log(value)
    return entropy / math.log(len(values))


def read_orders(zip_path: Path, scene: str, limit_rows: int) -> list[list[int]]:
    orders: list[list[int]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"{scene}.csv") as file:
            reader = csv.reader(line.decode("utf-8") for line in file)
            for row in reader:
                values = [float(value) for value in row if value != ""]
                orders.append(sorted(range(len(values)), key=lambda index: values[index], reverse=True)[:10])
                if limit_rows and len(orders) >= limit_rows:
                    break
    return orders


def load_contexts(data_zip: zipfile.ZipFile, limit_rows: int) -> dict[str, SceneContext]:
    contexts: dict[str, SceneContext] = {}
    for scene in SCENES:
        train_dsts, train_pairs = load_train_sets(data_zip, scene)
        test_rows = load_test_rows(data_zip, scene, limit_rows)
        contexts[scene] = SceneContext(
            expected_rows=len(test_rows),
            train_dsts=train_dsts,
            train_pairs=train_pairs,
            test_rows=test_rows,
        )
    return contexts


def audit_scene(zip_path: Path, scene: str, context: SceneContext, limit_rows: int) -> SceneAudit:
    train_dsts = context.train_dsts
    train_pairs = context.train_pairs
    test_rows = context.test_rows

    distinct_counts: list[int] = []
    min_ties: list[int] = []
    max_probabilities: list[float] = []
    entropies: list[float] = []
    sums: list[float] = []
    bad_rows = 0
    rows = 0
    top1_seen_dst = 0
    top1_repeated_pair = 0
    top10_seen_dst = 0
    top10_repeated_pair = 0
    top10_total = 0

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        member = f"{scene}.csv"
        if member not in names:
            return SceneAudit(0, context.expected_rows, context.expected_rows, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        with archive.open(member) as file:
            reader = csv.reader(line.decode("utf-8") for line in file)
            for row_index, row in enumerate(reader):
                if limit_rows and row_index >= limit_rows:
                    break
                rows += 1
                values = [float(value) for value in row if value != ""]
                if row_index >= len(test_rows):
                    bad_rows += 1
                    continue
                src, candidates = test_rows[row_index]
                if len(values) != 100 or len(candidates) != 100 or any(value < 0 or value > 1 for value in values):
                    bad_rows += 1
                    continue

                order = sorted(range(100), key=lambda index: values[index], reverse=True)
                top1_dst = candidates[order[0]]
                top1_seen_dst += int(top1_dst in train_dsts)
                top1_repeated_pair += int((src, top1_dst) in train_pairs)

                for index in order[:10]:
                    dst = candidates[index]
                    top10_total += 1
                    top10_seen_dst += int(dst in train_dsts)
                    top10_repeated_pair += int((src, dst) in train_pairs)

                row_sum = sum(values)
                sums.append(row_sum)
                distinct_counts.append(len(set(values)))
                min_value = min(values)
                min_ties.append(sum(value == min_value for value in values))
                max_probabilities.append(max(values))
                entropies.append(probability_entropy(values))

    if rows != len(test_rows):
        bad_rows += abs(rows - len(test_rows))

    return SceneAudit(
        rows=rows,
        expected_rows=context.expected_rows,
        bad_rows=bad_rows,
        avg_distinct=statistics.fmean(distinct_counts) if distinct_counts else 0.0,
        median_distinct=statistics.median(distinct_counts) if distinct_counts else 0.0,
        avg_min_tie=statistics.fmean(min_ties) if min_ties else 0.0,
        avg_max_probability=statistics.fmean(max_probabilities) if max_probabilities else 0.0,
        avg_normalized_entropy=statistics.fmean(entropies) if entropies else 0.0,
        min_sum=min(sums) if sums else 0.0,
        max_sum=max(sums) if sums else 0.0,
        top1_seen_dst_ratio=top1_seen_dst / max(1, rows),
        top1_repeated_pair_ratio=top1_repeated_pair / max(1, rows),
        top10_seen_dst_ratio=top10_seen_dst / max(1, top10_total),
        top10_repeated_pair_ratio=top10_repeated_pair / max(1, top10_total),
    )


def compare_orders(candidate_zip: Path, baseline_zip: Path, limit_rows: int) -> dict[str, AgreementAudit]:
    agreement: dict[str, AgreementAudit] = {}
    for scene in SCENES:
        candidate_orders = read_orders(candidate_zip, scene, limit_rows)
        baseline_orders = read_orders(baseline_zip, scene, limit_rows)
        pairs = list(zip(candidate_orders, baseline_orders))
        if not pairs:
            agreement[scene] = AgreementAudit(0.0, 0.0)
            continue
        top1 = sum(candidate[0] == baseline[0] for candidate, baseline in pairs) / len(pairs)
        jaccard = 0.0
        for candidate, baseline in pairs:
            candidate_set = set(candidate)
            baseline_set = set(baseline)
            jaccard += len(candidate_set & baseline_set) / len(candidate_set | baseline_set)
        agreement[scene] = AgreementAudit(top1, jaccard / len(pairs))
    return agreement


def print_report(results: dict[str, object]) -> None:
    for submission, payload in results.items():
        print(f"\n## {submission}")
        scenes = payload["scenes"]
        for scene in SCENES:
            stats = scenes[scene]
            print(
                f"{scene}: rows={stats['rows']}/{stats['expected_rows']} "
                f"bad={stats['bad_rows']} distinct_avg={stats['avg_distinct']:.2f} "
                f"distinct_med={stats['median_distinct']:.1f} min_tie_avg={stats['avg_min_tie']:.2f} "
                f"max_p_avg={stats['avg_max_probability']:.6f} entropy={stats['avg_normalized_entropy']:.4f} "
                f"sum=[{stats['min_sum']:.8f},{stats['max_sum']:.8f}] "
                f"top1_seen_dst={stats['top1_seen_dst_ratio']:.4f} "
                f"top1_repeat_pair={stats['top1_repeated_pair_ratio']:.4f}"
            )
        if payload.get("agreement"):
            print("agreement_vs_baseline:")
            for scene, stats in payload["agreement"].items():
                print(
                    f"{scene}: top1={stats['top1_agreement']:.4f} "
                    f"top10_jaccard={stats['top10_jaccard']:.4f}"
                )


def main() -> None:
    args = parse_args()
    results: dict[str, object] = {}
    with zipfile.ZipFile(args.data_zip) as data_zip:
        contexts = load_contexts(data_zip, args.limit_rows)
        for submission in args.submissions:
            scenes = {
                scene: asdict(audit_scene(submission, scene, contexts[scene], args.limit_rows))
                for scene in SCENES
            }
            payload: dict[str, object] = {"scenes": scenes}
            if args.baseline_zip and submission != args.baseline_zip:
                payload["agreement"] = {
                    scene: asdict(stats)
                    for scene, stats in compare_orders(submission, args.baseline_zip, args.limit_rows).items()
                }
            results[str(submission)] = payload

    print_report(results)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
