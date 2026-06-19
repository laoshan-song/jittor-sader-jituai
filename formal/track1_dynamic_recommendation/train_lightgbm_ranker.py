"""LightGBM LambdaMART reranker for Track 1.

This is an exploration path for stronger tabular ranking. The competition audit
path still needs a Jittor implementation or a reproducible distillation, so this
script writes candidates outside outputs/ by default and should be gated before
any submission is handed over.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import warnings
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

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


FEATURE_NAMES = [
    "base_score",
    "base_rank_recip",
    "base_percentile",
    "base_centered",
    "base_z",
    "pair_log",
    "pair_recency",
    "dst_log",
    "dst_log_norm",
    "dst_recency",
    "recent_dst_log_norm",
    "in_recent_1",
    "in_recent_3",
    "in_recent_5",
    "in_recent_10",
    "in_recent_20",
    "recent_distance_recip",
    "transition_log",
    "transition_log_norm",
    "src_history_log",
    "src_unique_log",
    "src_repeat_ratio",
    "dst_pop_rank_recip",
    "dst_is_seen",
    "query_time_norm",
]


@dataclass
class EvalRow:
    scene: str
    split: str
    strategy: str
    positives: int
    baseline_mrr: float
    ranker_mrr: float
    blend_mrr: float
    baseline_hit1: float
    ranker_hit1: float
    blend_hit1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaMART reranker")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--primary-zip", type=Path, default=Path("outputs/track1/result.zip"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/result_lgbm_ranker.zip"))
    parser.add_argument("--scene", default="dataset2")
    parser.add_argument("--strategy", choices=("hard", "test_pool", "mixed"), default="test_pool")
    parser.add_argument("--train-queries", type=int, default=30000)
    parser.add_argument("--valid-queries", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--blend-weight", type=float, default=0.35)
    parser.add_argument("--score-mode", choices=("ranker", "blend"), default="ranker")
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--n-estimators", type=int, default=450)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--no-write-submission", action="store_true")
    parser.add_argument("--write-batch-size", type=int, default=1024)
    parser.add_argument("--report-json", type=Path, default=Path("/tmp/lightgbm_ranker_report.json"))
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


def split_for_scene(
    rows: list[tuple[int, int, int, str | None]],
    scene: str,
    valid_fraction: float,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    if scene == "dataset2" and any(split == "1" for *_, split in rows):
        history = [(src, dst, time) for src, dst, time, split in rows if split == "0"]
        positives = [(src, dst, time) for src, dst, time, split in rows if split != "0"]
        return history, positives

    ordered = sorted(rows, key=lambda value: value[2])
    cut = int(len(ordered) * (1.0 - valid_fraction))
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
    elif strategy == "mixed":
        add_unique_top(candidates, seen, popular_dsts, min(target_size, 1 + target_size // 3))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    add_random(all_dsts, target_size)
    if len(candidates) != target_size:
        raise RuntimeError(f"Only built {len(candidates)} candidates")
    rng.shuffle(candidates)
    return candidates


def raw_features(model: HistoryBaseline, src: int, dst: int, time_value: int) -> list[float]:
    src_counts = model.src_dst_count.get(src, {})
    pair_count = src_counts.get(dst, 0)
    pair_last = model.src_dst_last_time.get(src, {}).get(dst)
    dst_count = model.dst_count.get(dst, 0)
    dst_last = model.dst_last_time.get(dst)
    recent = model.src_recent_dsts.get(src, [])
    src_unique = len(src_counts)
    src_len = len(model.src_history.get(src, []))
    transition_score = 0.0
    for offset, prev_dst in enumerate(reversed(recent[-10:]), start=1):
        transition_score += model.transition_count.get(prev_dst, {}).get(dst, 0) / offset
    recent_distance = 0.0
    if dst in recent:
        recent_distance = 1.0 / (1.0 + len(recent) - 1 - max(index for index, value in enumerate(recent) if value == dst))
    dst_pop_rank = 1.0 / (1.0 + math.log1p(dst_count))
    time_span = max(1, (model.max_time or time_value) - (model.min_time or time_value))
    query_time_norm = (time_value - (model.min_time or time_value)) / time_span
    transition_log = math.log1p(transition_score)
    return [
        model.score(src, dst, time_value),
        math.log1p(pair_count),
        model._recency(time_value, pair_last),
        math.log1p(dst_count),
        math.log1p(dst_count) / model.max_dst_log if model.max_dst_log else 0.0,
        model._recency(time_value, dst_last),
        math.log1p(model.recent_dst_count.get(dst, 0)) / model.max_recent_dst_log if model.max_recent_dst_log else 0.0,
        1.0 if dst in recent[-1:] else 0.0,
        1.0 if dst in recent[-3:] else 0.0,
        1.0 if dst in recent[-5:] else 0.0,
        1.0 if dst in recent[-10:] else 0.0,
        1.0 if dst in recent[-20:] else 0.0,
        recent_distance,
        transition_log,
        transition_log / model.max_transition_log if model.max_transition_log else 0.0,
        math.log1p(src_len),
        math.log1p(src_unique),
        (src_len / src_unique) if src_unique else 0.0,
        dst_pop_rank,
        1.0 if dst_count else 0.0,
        query_time_norm,
    ]


def query_features(model: HistoryBaseline, src: int, candidates: list[int], time_value: int) -> np.ndarray:
    raw = np.asarray([raw_features(model, src, dst, time_value) for dst in candidates], dtype=np.float32)
    base = raw[:, 0]
    order = np.argsort(-base)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    centered = base - float(base.mean())
    std = float(base.std())
    if std <= 1e-8:
        z = np.zeros_like(base)
    else:
        z = centered / std
    relative = np.column_stack(
        [
            base,
            1.0 / (1.0 + ranks.astype(np.float32)),
            1.0 - ranks.astype(np.float32) / max(1, len(candidates) - 1),
            centered,
            z,
        ]
    ).astype(np.float32)
    return np.concatenate([relative, raw[:, 1:]], axis=1)


def build_rank_data(
    model: HistoryBaseline,
    positives: list[tuple[int, int, int]],
    all_dsts: list[int],
    popular_dsts: list[int],
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = random.Random(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[int] = []
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
        xs.append(query_features(model, src, candidates, time_value))
        ys.append(np.asarray([1 if candidate == dst else 0 for candidate in candidates], dtype=np.int32))
        groups.append(len(candidates))
    return np.vstack(xs).astype(np.float32), np.concatenate(ys), groups


def reciprocal_rank(candidates: list[int], scores: np.ndarray, positive_dst: int) -> tuple[float, int]:
    for rank, index in enumerate(np.argsort(-scores), start=1):
        if candidates[int(index)] == positive_dst:
            return 1.0 / rank, rank
    return 0.0, 101


def predict_ranker(ranker, x: np.ndarray) -> np.ndarray:
    if hasattr(ranker, "booster_"):
        return np.asarray(ranker.booster_.predict(x), dtype=np.float32)
    return np.asarray(ranker.predict(x), dtype=np.float32)


def evaluate(
    ranker,
    model: HistoryBaseline,
    positives: list[tuple[int, int, int]],
    all_dsts: list[int],
    popular_dsts: list[int],
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    scene: str,
    split: str,
    strategy: str,
    seed: int,
    blend_weight: float,
) -> EvalRow:
    rng = random.Random(seed)
    base_rr = 0.0
    ranker_rr = 0.0
    blend_rr = 0.0
    base_hit1 = 0
    ranker_hit1 = 0
    blend_hit1 = 0
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
        x = query_features(model, src, candidates, time_value)
        base_scores = x[:, 0]
        ranker_scores = predict_ranker(ranker, x)
        blend_scores = ranker_scores if blend_weight == 0.0 else base_scores + blend_weight * ranker_scores
        rr, rank = reciprocal_rank(candidates, base_scores, dst)
        base_rr += rr
        base_hit1 += int(rank == 1)
        rr, rank = reciprocal_rank(candidates, ranker_scores, dst)
        ranker_rr += rr
        ranker_hit1 += int(rank == 1)
        rr, rank = reciprocal_rank(candidates, blend_scores, dst)
        blend_rr += rr
        blend_hit1 += int(rank == 1)
    n = len(positives)
    return EvalRow(
        scene=scene,
        split=split,
        strategy=strategy,
        positives=n,
        baseline_mrr=base_rr / n,
        ranker_mrr=ranker_rr / n,
        blend_mrr=blend_rr / n,
        baseline_hit1=base_hit1 / n,
        ranker_hit1=ranker_hit1 / n,
        blend_hit1=blend_hit1 / n,
    )


def write_submission(
    data_zip: zipfile.ZipFile,
    output: Path,
    primary_zip: Path,
    ranker,
    model: HistoryBaseline,
    scene: str,
    blend_weight: float,
    score_mode: str,
    batch_size: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        with zipfile.ZipFile(primary_zip) as primary:
            for current_scene in ("dataset1", "dataset2"):
                if current_scene != scene:
                    output_zip.writestr(f"{current_scene}.csv", primary.read(f"{current_scene}.csv"))
                    continue
                with open_csv(data_zip, f"{current_scene}/test.csv") as input_file:
                    reader = csv.reader(input_file)
                    next(reader)
                    with output_zip.open(f"{current_scene}.csv", "w") as raw_output:
                        with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                            writer = csv.writer(text_output, lineterminator="\n")
                            features_batch: list[np.ndarray] = []
                            counts: list[int] = []
                            row_count = 0

                            def flush_batch() -> None:
                                nonlocal row_count
                                if not features_batch:
                                    return
                                x_batch = np.vstack(features_batch).astype(np.float32)
                                ranker_batch = predict_ranker(ranker, x_batch)
                                offset = 0
                                for x_row, count in zip(features_batch, counts):
                                    scores = ranker_batch[offset : offset + count]
                                    if score_mode == "blend":
                                        scores = x_row[:, 0] + blend_weight * scores
                                    writer.writerow([f"{value:.8f}" for value in rank_probabilities(scores.tolist())])
                                    offset += count
                                    row_count += 1
                                    if row_count % 10000 == 0:
                                        print(f"wrote {current_scene} rows={row_count}", flush=True)
                                features_batch.clear()
                                counts.clear()

                            for row in reader:
                                src = int(row[0])
                                time_value = int(row[1])
                                candidates = [int(value) for value in row[2:]]
                                x = query_features(model, src, candidates, time_value)
                                features_batch.append(x)
                                counts.append(len(candidates))
                                if len(features_batch) >= batch_size:
                                    flush_batch()
                            flush_batch()


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    try:
        from lightgbm import LGBMRanker
    except ImportError as exc:
        raise SystemExit("LightGBM is not installed in this environment.") from exc

    args = parse_args()
    rng = random.Random(args.seed)
    with zipfile.ZipFile(args.data_zip) as data_zip:
        rows = read_rows(data_zip, args.scene)
        history_rows, positive_rows = split_for_scene(rows, args.scene, args.valid_fraction)
        rng.shuffle(positive_rows)
        train_pos = positive_rows[: args.train_queries]
        valid_pos = positive_rows[args.train_queries : args.train_queries + args.valid_queries]
        if not train_pos or not valid_pos:
            raise RuntimeError("Not enough positive rows for training and validation.")

        history_model = fit_history(history_rows)
        all_dsts = sorted({dst for _, dst, _ in history_rows})
        popular_dsts = [dst for dst, _ in Counter(dst for _, dst, _ in history_rows).most_common(10000)]
        src_test_candidates, all_test_candidates = load_test_candidate_pools(data_zip, args.scene)

        print(
            f"building rank data scene={args.scene} train_queries={len(train_pos)} "
            f"valid_queries={len(valid_pos)} strategy={args.strategy}",
            flush=True,
        )
        x_train, y_train, group_train = build_rank_data(
            history_model,
            train_pos,
            all_dsts,
            popular_dsts,
            src_test_candidates,
            all_test_candidates,
            args.strategy,
            args.seed,
        )
        x_valid, y_valid, group_valid = build_rank_data(
            history_model,
            valid_pos,
            all_dsts,
            popular_dsts,
            src_test_candidates,
            all_test_candidates,
            args.strategy,
            args.seed + 1,
        )

        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            subsample_freq=1,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            label_gain=[0, 1],
            verbose=-1,
        )
        ranker.fit(
            x_train,
            y_train,
            group=group_train,
            eval_set=[(x_valid, y_valid)],
            eval_group=[group_valid],
            eval_at=[1, 5, 10],
        )

        reports = []
        for strategy in ("test_pool", "hard", "mixed"):
            reports.append(
                evaluate(
                    ranker,
                    history_model,
                    valid_pos,
                    all_dsts,
                    popular_dsts,
                    src_test_candidates,
                    all_test_candidates,
                    args.scene,
                    "local",
                    strategy,
                    args.seed + 19,
                    0.0 if args.score_mode == "ranker" else args.blend_weight,
                )
            )

        print("scene,split,strategy,n,baseline_mrr,ranker_mrr,blend_mrr,baseline_hit1,ranker_hit1,blend_hit1")
        for report in reports:
            print(
                f"{report.scene},{report.split},{report.strategy},{report.positives},"
                f"{report.baseline_mrr:.8f},{report.ranker_mrr:.8f},{report.blend_mrr:.8f},"
                f"{report.baseline_hit1:.8f},{report.ranker_hit1:.8f},{report.blend_hit1:.8f}",
                flush=True,
            )

        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(
                json.dumps([asdict(report) for report in reports], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if not args.no_write_submission:
            full_model = fit_history([(src, dst, time_value) for src, dst, time_value, _ in rows])
            write_submission(
                data_zip,
                args.output,
                args.primary_zip,
                ranker,
                full_model,
                args.scene,
                args.blend_weight,
                args.score_mode,
                args.write_batch_size,
            )
            print(f"submission candidate saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
