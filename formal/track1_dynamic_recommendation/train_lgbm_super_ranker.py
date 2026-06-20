"""High-capacity LightGBM ranker for Track 1.

This route is intentionally not a submission fusion. It trains a fresh
LambdaMART model per scene with richer candidate-in-query features and writes a
complete result.zip.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from baseline import HistoryBaseline
from rank_utils import rank_probabilities
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
class EvalRow:
    scene: str
    strategy: str
    positives: int
    base_mrr: float
    model_mrr: float
    hit1: float


class FeatureStore:
    def __init__(self, rows: list[tuple[int, int, int]], test_src_candidates: dict[int, list[int]], all_test_candidates: list[int]) -> None:
        self.model = HistoryBaseline(**HEURISTIC_WEIGHTS)
        self.src_history: dict[int, list[int]] = defaultdict(list)
        self.src_unique: dict[int, set[int]] = defaultdict(set)
        self.dst_sources: dict[int, set[int]] = defaultdict(set)
        self.node_count: Counter[int] = Counter()
        self.test_freq: Counter[int] = Counter(all_test_candidates)
        self.src_test_freq = {src: Counter(cands) for src, cands in test_src_candidates.items()}
        self.co_recent: dict[int, Counter[int]] = defaultdict(Counter)

        for src, dst, time_value in sorted(rows, key=lambda value: value[2]):
            self.model.update(src, dst, time_value)
            recent = self.src_history[src][-12:]
            for prev in recent:
                if prev != dst:
                    self.co_recent[prev][dst] += 1
            self.src_history[src].append(dst)
            self.src_unique[src].add(dst)
            self.dst_sources[dst].add(src)
            self.node_count[src] += 1
            self.node_count[dst] += 1
        self.model.finalize()
        self.max_test_log = max((math.log1p(v) for v in self.test_freq.values()), default=1.0)
        self.max_node_log = max((math.log1p(v) for v in self.node_count.values()), default=1.0)
        self.max_co_log = max((math.log1p(v) for counts in self.co_recent.values() for v in counts.values()), default=1.0)

    def raw_feature(self, src: int, dst: int, time_value: int) -> list[float]:
        model = self.model
        src_counts = model.src_dst_count.get(src, {})
        pair_count = src_counts.get(dst, 0)
        pair_last = model.src_dst_last_time.get(src, {}).get(dst)
        dst_count = model.dst_count.get(dst, 0)
        dst_last = model.dst_last_time.get(dst)
        recent = model.src_recent_dsts.get(src, [])
        transition = 0.0
        motif = 0.0
        motif_recent = 0.0
        for offset, prev in enumerate(reversed(recent[-20:]), start=1):
            transition += model.transition_count.get(prev, {}).get(dst, 0) / offset
            count = self.co_recent.get(prev, {}).get(dst, 0)
            if count:
                value = math.log1p(count) / self.max_co_log
                motif += value / math.sqrt(offset)
                motif_recent += value / offset
        repeat_distance = 0.0
        if dst in recent:
            repeat_distance = 1.0 / (1.0 + len(recent) - 1 - max(i for i, value in enumerate(recent) if value == dst))
        dst_source_overlap = 1.0 if src in self.dst_sources.get(dst, set()) else 0.0
        src_test_count = self.src_test_freq.get(src, Counter()).get(dst, 0)
        time_span = max(1, (model.max_time or time_value) - (model.min_time or time_value))
        return [
            model.score(src, dst, time_value),
            math.log1p(pair_count),
            model._recency(time_value, pair_last),
            math.log1p(dst_count),
            math.log1p(dst_count) / model.max_dst_log if model.max_dst_log else 0.0,
            model._recency(time_value, dst_last),
            1.0 if dst in recent[-1:] else 0.0,
            1.0 if dst in recent[-3:] else 0.0,
            1.0 if dst in recent[-10:] else 0.0,
            1.0 if dst in recent[-30:] else 0.0,
            repeat_distance,
            math.log1p(transition),
            motif,
            motif_recent,
            math.log1p(len(recent)),
            math.log1p(len(src_counts)),
            len(recent) / max(1, len(src_counts)),
            math.log1p(self.node_count.get(dst, 0)) / self.max_node_log if self.node_count.get(dst, 0) else 0.0,
            dst_source_overlap,
            math.log1p(self.test_freq.get(dst, 0)) / self.max_test_log if self.test_freq.get(dst, 0) else 0.0,
            math.log1p(src_test_count),
            (time_value - (model.min_time or time_value)) / time_span,
        ]

    def query_features(self, src: int, candidates: list[int], time_value: int) -> np.ndarray:
        raw = np.asarray([self.raw_feature(src, dst, time_value) for dst in candidates], dtype=np.float32)
        parts = [raw]
        for col in range(raw.shape[1]):
            values = raw[:, col]
            order = np.argsort(-values)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(values))
            centered = values - float(values.mean())
            std = float(values.std())
            z = centered / std if std > 1e-8 else np.zeros_like(values)
            parts.append(np.column_stack([
                1.0 / (1.0 + ranks.astype(np.float32)),
                1.0 - ranks.astype(np.float32) / max(1, len(values) - 1),
                centered,
                z,
            ]).astype(np.float32))
        return np.concatenate(parts, axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train high-capacity LGBM ranker")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", default="dataset1,dataset2")
    parser.add_argument("--train-queries", type=int, default=30000)
    parser.add_argument("--valid-queries", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--strategy", choices=("test_pool", "src_pool", "mixed"), default="test_pool")
    parser.add_argument("--n-estimators", type=int, default=520)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--no-write-submission", action="store_true")
    parser.add_argument("--limit-test-rows", type=int, default=0)
    parser.add_argument("--report-json", type=Path, default=Path("/tmp/super_ranker_report.json"))
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def read_rows(data_zip: zipfile.ZipFile, scene: str) -> list[tuple[int, int, int, str | None]]:
    rows = []
    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append((int(row["src"]), int(row["dst"]), int(row["time"]), row.get("split")))
    return rows


def split_rows(rows: list[tuple[int, int, int, str | None]], scene: str) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    if scene == "dataset2" and any(split == "1" for *_, split in rows):
        return (
            [(s, d, t) for s, d, t, split in rows if split == "0"],
            [(s, d, t) for s, d, t, split in rows if split != "0"],
        )
    ordered = sorted(rows, key=lambda value: value[2])
    cut = int(len(ordered) * 0.85)
    return (
        [(s, d, t) for s, d, t, _ in ordered[:cut]],
        [(s, d, t) for s, d, t, _ in ordered[cut:]],
    )


def make_candidates(
    positive: int,
    src: int,
    all_dsts: list[int],
    src_pool: dict[int, list[int]],
    all_test: list[int],
    popular: list[int],
    strategy: str,
    rng: random.Random,
) -> list[int]:
    candidates = [positive]
    seen = {positive}

    def add_random(pool: list[int], target: int) -> None:
        if len(candidates) >= target or not pool:
            return
        attempts = 0
        while len(candidates) < target and attempts < max(1000, target * 50):
            attempts += 1
            dst = rng.choice(pool)
            if dst not in seen:
                seen.add(dst)
                candidates.append(dst)
        for dst in pool:
            if len(candidates) >= target:
                break
            if dst not in seen:
                seen.add(dst)
                candidates.append(dst)

    def add_top(pool: list[int], target: int) -> None:
        for dst in pool:
            if len(candidates) >= target:
                return
            if dst not in seen:
                seen.add(dst)
                candidates.append(dst)

    if strategy == "src_pool":
        add_random(src_pool.get(src, []), 70)
        add_random(all_test, 92)
    elif strategy == "mixed":
        add_random(src_pool.get(src, []), 45)
        add_top(popular, 70)
        add_random(all_test, 92)
    else:
        add_random(all_test, 85)
    add_random(all_dsts, 100)
    if len(candidates) != 100:
        raise RuntimeError(f"Only built {len(candidates)} candidates")
    rng.shuffle(candidates)
    return candidates


def build_rank_data(
    store: FeatureStore,
    positives: list[tuple[int, int, int]],
    all_dsts: list[int],
    src_pool: dict[int, list[int]],
    all_test: list[int],
    popular: list[int],
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = random.Random(seed)
    xs, ys, groups = [], [], []
    for src, dst, time_value in positives:
        candidates = make_candidates(dst, src, all_dsts, src_pool, all_test, popular, strategy, rng)
        xs.append(store.query_features(src, candidates, time_value))
        ys.append(np.asarray([1 if c == dst else 0 for c in candidates], dtype=np.int32))
        groups.append(100)
    return np.vstack(xs).astype(np.float32), np.concatenate(ys), groups


def reciprocal_rank(candidates: list[int], scores: np.ndarray, positive: int) -> tuple[float, int]:
    for rank, idx in enumerate(np.argsort(-scores), start=1):
        if candidates[int(idx)] == positive:
            return 1.0 / rank, rank
    return 0.0, 101


def evaluate(ranker, store: FeatureStore, positives: list[tuple[int, int, int]], all_dsts: list[int], src_pool: dict[int, list[int]], all_test: list[int], popular: list[int], strategy: str, seed: int, scene: str) -> EvalRow:
    rng = random.Random(seed)
    base_rr = model_rr = 0.0
    hit1 = 0
    for src, dst, time_value in positives:
        candidates = make_candidates(dst, src, all_dsts, src_pool, all_test, popular, strategy, rng)
        x = store.query_features(src, candidates, time_value)
        scores = np.asarray(ranker.booster_.predict(x), dtype=np.float32)
        rr, rank = reciprocal_rank(candidates, x[:, 0], dst)
        base_rr += rr
        rr, rank = reciprocal_rank(candidates, scores, dst)
        model_rr += rr
        hit1 += int(rank == 1)
    n = len(positives)
    return EvalRow(scene=scene, strategy=strategy, positives=n, base_mrr=base_rr / n, model_mrr=model_rr / n, hit1=hit1 / n)


def write_scene(data_zip: zipfile.ZipFile, output_zip: zipfile.ZipFile, scene: str, store: FeatureStore, ranker, limit_rows: int, batch_size: int = 2048) -> None:
    with open_csv(data_zip, f"{scene}/test.csv") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        with output_zip.open(f"{scene}.csv", "w") as raw_output:
            with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                writer = csv.writer(text_output, lineterminator="\n")
                features_batch: list[np.ndarray] = []
                rows = 0
                for row in reader:
                    src = int(row[0])
                    time_value = int(row[1])
                    candidates = [int(value) for value in row[2:]]
                    features_batch.append(store.query_features(src, candidates, time_value))
                    rows += 1
                    if len(features_batch) >= batch_size:
                        flush(writer, features_batch, ranker)
                    if limit_rows and rows >= limit_rows:
                        break
                flush(writer, features_batch, ranker)


def flush(writer: csv.writer, features_batch: list[np.ndarray], ranker) -> None:
    if not features_batch:
        return
    x = np.vstack(features_batch).astype(np.float32)
    pred = np.asarray(ranker.booster_.predict(x), dtype=np.float32)
    offset = 0
    for features in features_batch:
        scores = pred[offset:offset + len(features)]
        writer.writerow([f"{value:.8f}" for value in rank_probabilities(scores.tolist())])
        offset += len(features)
    features_batch.clear()


def main() -> None:
    from lightgbm import LGBMRanker

    args = parse_args()
    rng = random.Random(args.seed)
    reports: list[EvalRow] = []
    scene_models = {}
    with zipfile.ZipFile(args.data_zip) as data_zip:
        for scene in [value.strip() for value in args.scenes.split(",") if value.strip()]:
            rows = read_rows(data_zip, scene)
            history, positives = split_rows(rows, scene)
            rng.shuffle(positives)
            train_pos = positives[: args.train_queries]
            valid_pos = positives[args.train_queries:args.train_queries + args.valid_queries]
            src_pool, all_test = load_test_candidate_pools(data_zip, scene)
            store = FeatureStore(history, src_pool, all_test)
            all_dsts = sorted({dst for _, dst, _ in history})
            popular = [dst for dst, _ in store.model.dst_count.most_common(20000)]
            print(f"[{scene}] build rank data train={len(train_pos)} valid={len(valid_pos)}", flush=True)
            x_train, y_train, group_train = build_rank_data(store, train_pos, all_dsts, src_pool, all_test, popular, args.strategy, args.seed)
            x_valid, y_valid, group_valid = build_rank_data(store, valid_pos, all_dsts, src_pool, all_test, popular, args.strategy, args.seed + 1)
            ranker = LGBMRanker(
                objective="rank_xendcg",
                metric="ndcg",
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                min_child_samples=60,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_lambda=0.5,
                random_state=args.seed,
                n_jobs=args.n_jobs,
                label_gain=[0, 1],
                verbose=-1,
            )
            ranker.fit(x_train, y_train, group=group_train, eval_set=[(x_valid, y_valid)], eval_group=[group_valid], eval_at=[1, 5, 10])
            for strategy in ("test_pool", "src_pool", "mixed"):
                reports.append(evaluate(ranker, store, valid_pos, all_dsts, src_pool, all_test, popular, strategy, args.seed + 19, scene))
            full_rows = [(s, d, t) for s, d, t, _ in rows]
            full_store = FeatureStore(full_rows, src_pool, all_test)
            scene_models[scene] = (full_store, ranker)

        if not args.no_write_submission:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
                for scene, (store, ranker) in scene_models.items():
                    print(f"[{scene}] write submission", flush=True)
                    write_scene(data_zip, output_zip, scene, store, ranker, args.limit_test_rows)

    print("scene,strategy,n,base_mrr,model_mrr,hit1")
    for row in reports:
        print(f"{row.scene},{row.strategy},{row.positives},{row.base_mrr:.8f},{row.model_mrr:.8f},{row.hit1:.8f}")
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps([asdict(row) for row in reports], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
