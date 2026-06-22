"""Stacked ranker for Track 1.

Trains multiple XGBoost base rankers on different negative distributions, then
learns a meta XGBoost ranker from their per-query scores/ranks plus handcrafted
features. This is a leaderboard-first route.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xgboost as xgb

from rank_utils import rank_probabilities
from train_lgbm_super_ranker import (
    EvalRow,
    FeatureStore,
    build_rank_data,
    load_test_candidate_pools,
    make_candidates,
    reciprocal_rank,
    read_rows,
    split_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stacked XGBoost ranker")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="dataset2")
    parser.add_argument("--base-train-queries", type=int, default=12000)
    parser.add_argument("--meta-train-queries", type=int, default=8000)
    parser.add_argument("--valid-queries", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2200)
    parser.add_argument("--base-rounds", type=int, default=260)
    parser.add_argument("--meta-rounds", type=int, default=360)
    parser.add_argument("--nthread", type=int, default=12)
    parser.add_argument("--eval-strategies", default="test_pool,src_pool,mixed")
    parser.add_argument("--no-write-submission", action="store_true")
    parser.add_argument("--report-json", type=Path, default=Path("/tmp/stack_ranker_report.json"))
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def train_xgb(x_train: np.ndarray, y_train: np.ndarray, groups: list[int], rounds: int, seed: int, nthread: int, depth: int, eta: float) -> xgb.Booster:
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtrain.set_group(groups)
    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "tree_method": "hist",
        "max_depth": depth,
        "eta": eta,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "seed": seed,
        "nthread": nthread,
    }
    return xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False)


def predict(booster: xgb.Booster, x: np.ndarray) -> np.ndarray:
    return np.asarray(booster.predict(xgb.DMatrix(x)), dtype=np.float32)


def rank_features(scores_by_model: list[np.ndarray], raw_x: np.ndarray) -> np.ndarray:
    parts = [raw_x[:, :24]]
    for scores in scores_by_model:
        order = np.argsort(-scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores))
        centered = scores - float(scores.mean())
        std = float(scores.std())
        z = centered / std if std > 1e-8 else np.zeros_like(scores)
        parts.append(
            np.column_stack(
                [
                    scores,
                    1.0 / (1.0 + ranks.astype(np.float32)),
                    1.0 - ranks.astype(np.float32) / max(1, len(scores) - 1),
                    centered,
                    z,
                ]
            ).astype(np.float32)
        )
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_meta_data(store: FeatureStore, boosters: list[xgb.Booster], positives, all_dsts, src_pool, all_test, popular, strategy: str, seed: int):
    rng = random.Random(seed)
    xs, ys, groups = [], [], []
    for src, dst, time_value in positives:
        candidates = make_candidates(dst, src, all_dsts, src_pool, all_test, popular, strategy, rng)
        raw_x = store.query_features(src, candidates, time_value)
        score_parts = [raw_x[:, 0]] + [predict(booster, raw_x) for booster in boosters]
        xs.append(rank_features(score_parts, raw_x))
        ys.append(np.asarray([1 if c == dst else 0 for c in candidates], dtype=np.int32))
        groups.append(100)
    return np.vstack(xs).astype(np.float32), np.concatenate(ys), groups


def evaluate_stack(meta: xgb.Booster, boosters: list[xgb.Booster], store: FeatureStore, positives, all_dsts, src_pool, all_test, popular, strategy: str, seed: int, scene: str) -> EvalRow:
    rng = random.Random(seed)
    base_rr = model_rr = 0.0
    hit1 = 0
    for src, dst, time_value in positives:
        candidates = make_candidates(dst, src, all_dsts, src_pool, all_test, popular, strategy, rng)
        raw_x = store.query_features(src, candidates, time_value)
        score_parts = [raw_x[:, 0]] + [predict(booster, raw_x) for booster in boosters]
        meta_x = rank_features(score_parts, raw_x)
        scores = predict(meta, meta_x)
        rr, _ = reciprocal_rank(candidates, raw_x[:, 0], dst)
        base_rr += rr
        rr, rank = reciprocal_rank(candidates, scores, dst)
        model_rr += rr
        hit1 += int(rank == 1)
    n = len(positives)
    return EvalRow(scene=scene, strategy=strategy, positives=n, base_mrr=base_rr / n, model_mrr=model_rr / n, hit1=hit1 / n)


def write_scene(data_zip: zipfile.ZipFile, output_zip: zipfile.ZipFile, scene: str, store: FeatureStore, boosters: list[xgb.Booster], meta: xgb.Booster, batch_size: int = 1024) -> None:
    with open_csv(data_zip, f"{scene}/test.csv") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        with output_zip.open(f"{scene}.csv", "w") as raw_output:
            with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                writer = csv.writer(text_output, lineterminator="\n")
                rows = 0
                for row in reader:
                    src = int(row[0])
                    time_value = int(row[1])
                    candidates = [int(value) for value in row[2:]]
                    raw_x = store.query_features(src, candidates, time_value)
                    score_parts = [raw_x[:, 0]] + [predict(booster, raw_x) for booster in boosters]
                    meta_x = rank_features(score_parts, raw_x)
                    scores = predict(meta, meta_x)
                    writer.writerow([f"{value:.8f}" for value in rank_probabilities(scores.tolist())])
                    rows += 1
                    if rows % 10000 == 0:
                        print(f"[{scene}] wrote rows={rows}", flush=True)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    reports: list[EvalRow] = []
    with zipfile.ZipFile(args.data_zip) as data_zip:
        rows = read_rows(data_zip, args.scene)
        history, positives = split_rows(rows, args.scene)
        rng.shuffle(positives)
        base_pos = positives[: args.base_train_queries]
        meta_pos = positives[args.base_train_queries:args.base_train_queries + args.meta_train_queries]
        valid_pos = positives[args.base_train_queries + args.meta_train_queries:args.base_train_queries + args.meta_train_queries + args.valid_queries]
        src_pool, all_test = load_test_candidate_pools(data_zip, args.scene)
        store = FeatureStore(history, src_pool, all_test)
        all_dsts = sorted({dst for _, dst, _ in history})
        popular = [dst for dst, _ in store.model.dst_count.most_common(20000)]

        boosters = []
        for index, strategy in enumerate(("test_pool", "src_pool", "mixed")):
            print(f"[{args.scene}] train base {strategy}", flush=True)
            x_base, y_base, g_base = build_rank_data(store, base_pos, all_dsts, src_pool, all_test, popular, strategy, args.seed + index)
            boosters.append(train_xgb(x_base, y_base, g_base, args.base_rounds, args.seed + index, args.nthread, 8, 0.045))

        print(f"[{args.scene}] build/train meta", flush=True)
        x_meta, y_meta, g_meta = build_meta_data(store, boosters, meta_pos, all_dsts, src_pool, all_test, popular, "test_pool", args.seed + 10)
        meta = train_xgb(x_meta, y_meta, g_meta, args.meta_rounds, args.seed + 99, args.nthread, 6, 0.04)

        for strategy in [value.strip() for value in args.eval_strategies.split(",") if value.strip()]:
            reports.append(evaluate_stack(meta, boosters, store, valid_pos, all_dsts, src_pool, all_test, popular, strategy, args.seed + 19, args.scene))

        if not args.no_write_submission:
            full_store = FeatureStore([(s, d, t) for s, d, t, _ in rows], src_pool, all_test)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
                write_scene(data_zip, output_zip, args.scene, full_store, boosters, meta)

    print("scene,strategy,n,base_mrr,model_mrr,hit1")
    for row in reports:
        print(f"{row.scene},{row.strategy},{row.positives},{row.base_mrr:.8f},{row.model_mrr:.8f},{row.hit1:.8f}")
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps([asdict(row) for row in reports], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
