"""Tree-based candidate reranker for Track 1.

LightGBM is ideal for this style of tabular ranking, but the offline server may
not have the package installed. This script uses sklearn's histogram gradient
boosting as a drop-in exploration model and keeps the feature pipeline reusable
for LightGBM once available.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

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
class EvalRow:
    scene: str
    split: str
    strategy: str
    positives: int
    baseline_mrr: float
    tree_mrr: float
    blend_mrr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tree candidate reranker")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="dataset2")
    parser.add_argument("--train-positives", type=int, default=50000)
    parser.add_argument("--valid-positives", type=int, default=10000)
    parser.add_argument("--negatives", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--strategy", choices=("hard", "test_pool", "mixed"), default="test_pool")
    parser.add_argument("--blend-weight", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--report-json", type=Path)
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


def split_rows(rows: list[tuple[int, int, int, str | None]]) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
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


def fit_history(rows: list[tuple[int, int, int]]) -> HistoryBaseline:
    model = HistoryBaseline(**HEURISTIC_WEIGHTS)
    for src, dst, time_value in rows:
        model.update(src, dst, time_value)
    model.finalize()
    return model


def make_candidates(
    positive_dst: int,
    src: int,
    strategy: str,
    all_dsts: list[int],
    popular_dsts: list[int],
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    target_size: int,
    rng: random.Random,
) -> list[int]:
    candidates = [positive_dst]
    seen = {positive_dst}

    def add_random(pool: list[int], size: int) -> None:
        available = [dst for dst in pool if dst not in seen]
        need = size - len(candidates)
        if need <= 0 or not available:
            return
        chosen = available if len(available) <= need else rng.sample(available, need)
        seen.update(chosen)
        candidates.extend(chosen)

    if strategy == "hard":
        add_random(src_test_candidates.get(src, []), min(target_size, 1 + target_size * 2 // 3))
        add_unique_top(candidates, seen, popular_dsts, min(target_size, 1 + target_size * 4 // 5))
        add_random(all_test_candidates, target_size)
    elif strategy == "test_pool":
        add_random(all_test_candidates, min(target_size, 1 + target_size * 4 // 5))
    else:
        add_unique_top(candidates, seen, popular_dsts, min(target_size, 1 + target_size // 3))
    add_random(all_dsts, target_size)
    if len(candidates) != target_size:
        raise RuntimeError(f"Only built {len(candidates)} candidates")
    rng.shuffle(candidates)
    return candidates


def features(model: HistoryBaseline, src: int, dst: int, time_value: int) -> list[float]:
    src_counts = model.src_dst_count.get(src, {})
    pair_count = src_counts.get(dst, 0)
    pair_last = model.src_dst_last_time.get(src, {}).get(dst)
    dst_count = model.dst_count.get(dst, 0)
    dst_last = model.dst_last_time.get(dst)
    recent = model.src_recent_dsts.get(src, [])
    transition_score = 0.0
    for offset, prev_dst in enumerate(reversed(recent[-10:]), start=1):
        transition_score += model.transition_count.get(prev_dst, {}).get(dst, 0) / offset
    base = model.score(src, dst, time_value)
    return [
        base,
        math.log1p(pair_count),
        model._recency(time_value, pair_last),
        math.log1p(dst_count),
        model._recency(time_value, dst_last),
        1.0 if dst in recent[-1:] else 0.0,
        1.0 if dst in recent[-5:] else 0.0,
        1.0 if dst in recent[-20:] else 0.0,
        math.log1p(transition_score),
        len(recent),
    ]


def build_dataset(
    model: HistoryBaseline,
    positives: list[tuple[int, int, int]],
    all_dsts: list[int],
    popular_dsts: list[int],
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    args: argparse.Namespace,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(args.seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    for src, dst, time_value in positives:
        candidates = make_candidates(
            dst,
            src,
            args.strategy,
            all_dsts,
            popular_dsts,
            src_test_candidates,
            all_test_candidates,
            target_size,
            rng,
        )
        for candidate in candidates:
            xs.append(features(model, src, candidate, time_value))
            ys.append(1 if candidate == dst else 0)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)


def reciprocal_rank(candidates: list[int], scores: list[float], positive_dst: int) -> float:
    for rank, index in enumerate(sorted(range(len(scores)), key=lambda value: scores[value], reverse=True), start=1):
        if candidates[index] == positive_dst:
            return 1.0 / rank
    return 0.0


def evaluate(
    clf: HistGradientBoostingClassifier,
    model: HistoryBaseline,
    positives: list[tuple[int, int, int]],
    all_dsts: list[int],
    popular_dsts: list[int],
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    args: argparse.Namespace,
    strategy: str,
) -> EvalRow:
    rng = random.Random(args.seed + 19)
    base_rr = 0.0
    tree_rr = 0.0
    blend_rr = 0.0
    for src, dst, time_value in positives:
        candidates = make_candidates(
            dst,
            src,
            strategy,
            all_dsts,
            popular_dsts,
            src_test_candidates,
            all_test_candidates,
            100,
            rng,
        )
        x = np.asarray([features(model, src, candidate, time_value) for candidate in candidates], dtype=np.float32)
        tree_scores = clf.predict_proba(x)[:, 1]
        base_scores = x[:, 0]
        blend_scores = base_scores + args.blend_weight * tree_scores
        base_rr += reciprocal_rank(candidates, base_scores.tolist(), dst)
        tree_rr += reciprocal_rank(candidates, tree_scores.tolist(), dst)
        blend_rr += reciprocal_rank(candidates, blend_scores.tolist(), dst)
    n = len(positives)
    return EvalRow(
        scene=args.scene,
        split="local",
        strategy=strategy,
        positives=n,
        baseline_mrr=base_rr / n,
        tree_mrr=tree_rr / n,
        blend_mrr=blend_rr / n,
    )


def write_submission(
    data_zip: zipfile.ZipFile,
    output: Path,
    primary_zip: Path,
    clf: HistGradientBoostingClassifier,
    model: HistoryBaseline,
    args: argparse.Namespace,
) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        with zipfile.ZipFile(primary_zip) as primary:
            for scene in ("dataset1", "dataset2"):
                if scene != args.scene:
                    output_zip.writestr(f"{scene}.csv", primary.read(f"{scene}.csv"))
                    continue
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
                                x = np.asarray(
                                    [features(model, src, dst, time_value) for dst in candidates],
                                    dtype=np.float32,
                                )
                                scores = x[:, 0] + args.blend_weight * clf.predict_proba(x)[:, 1]
                                writer.writerow([f"{value:.8f}" for value in rank_probabilities(scores.tolist())])


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.data_zip) as data_zip:
        rows = read_rows(data_zip, args.scene)
        train_rows, valid_rows = split_rows(rows)
        model = fit_history(train_rows)
        all_dsts = sorted({dst for _, dst, _ in train_rows})
        popular_dsts = [dst for dst, _ in Counter(dst for _, dst, _ in train_rows).most_common(10000)]
        src_test_candidates, all_test_candidates = load_test_candidate_pools(data_zip, args.scene)
        rng.shuffle(valid_rows)
        train_pos = valid_rows[:args.train_positives]
        valid_pos = valid_rows[args.train_positives:args.train_positives + args.valid_positives]
        print(f"building tree data positives={len(train_pos)} negatives={args.negatives}", flush=True)
        x_train, y_train = build_dataset(
            model,
            train_pos,
            all_dsts,
            popular_dsts,
            src_test_candidates,
            all_test_candidates,
            args,
            1 + args.negatives,
        )
        clf = HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2,
            random_state=args.seed,
        )
        clf.fit(x_train, y_train)
        reports = [
            evaluate(clf, model, valid_pos, all_dsts, popular_dsts, src_test_candidates, all_test_candidates, args, "test_pool"),
            evaluate(clf, model, valid_pos, all_dsts, popular_dsts, src_test_candidates, all_test_candidates, args, "hard"),
        ]
        for report in reports:
            print(
                f"{report.scene},{report.strategy},n={report.positives},"
                f"base={report.baseline_mrr:.8f},tree={report.tree_mrr:.8f},blend={report.blend_mrr:.8f}",
                flush=True,
            )
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps([asdict(row) for row in reports], indent=2), encoding="utf-8")
        if args.output:
            # The current anchor is expected at outputs/track1/result.zip.
            write_submission(data_zip, args.output, Path("outputs/track1/result.zip"), clf, fit_history([(src, dst, t) for src, dst, t, _ in rows]), args)
            print(f"submission saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
