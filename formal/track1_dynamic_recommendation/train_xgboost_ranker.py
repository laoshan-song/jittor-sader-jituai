"""XGBoost LambdaRank-style ranker for Track 1.

This is a fresh model family route, not a fusion of previous submissions.
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
    parser = argparse.ArgumentParser(description="Train XGBoost ranker for Track 1")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", default="dataset1,dataset2")
    parser.add_argument("--train-queries", type=int, default=25000)
    parser.add_argument("--valid-queries", type=int, default=5000)
    parser.add_argument("--strategy", choices=("test_pool", "src_pool", "mixed"), default="test_pool")
    parser.add_argument("--seed", type=int, default=2050)
    parser.add_argument("--num-boost-round", type=int, default=420)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.045)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--nthread", type=int, default=12)
    parser.add_argument("--no-write-submission", action="store_true")
    parser.add_argument("--report-json", type=Path, default=Path("/tmp/xgb_ranker_report.json"))
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def predict(booster: xgb.Booster, x: np.ndarray) -> np.ndarray:
    return np.asarray(booster.predict(xgb.DMatrix(x)), dtype=np.float32)


def write_scene(data_zip: zipfile.ZipFile, output_zip: zipfile.ZipFile, scene: str, store: FeatureStore, booster: xgb.Booster, batch_size: int = 1024) -> None:
    with open_csv(data_zip, f"{scene}/test.csv") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        with output_zip.open(f"{scene}.csv", "w") as raw_output:
            with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                writer = csv.writer(text_output, lineterminator="\n")
                batch: list[np.ndarray] = []
                rows = 0

                def flush() -> None:
                    nonlocal rows
                    if not batch:
                        return
                    x = np.vstack(batch).astype(np.float32)
                    scores = predict(booster, x)
                    offset = 0
                    for features in batch:
                        row_scores = scores[offset:offset + len(features)]
                        writer.writerow([f"{value:.8f}" for value in rank_probabilities(row_scores.tolist())])
                        offset += len(features)
                        rows += 1
                        if rows % 10000 == 0:
                            print(f"[{scene}] wrote rows={rows}", flush=True)
                    batch.clear()

                for row in reader:
                    src = int(row[0])
                    time_value = int(row[1])
                    candidates = [int(value) for value in row[2:]]
                    batch.append(store.query_features(src, candidates, time_value))
                    if len(batch) >= batch_size:
                        flush()
                flush()


def evaluate_xgb(booster: xgb.Booster, store: FeatureStore, positives, all_dsts, src_pool, all_test, popular, strategy: str, seed: int, scene: str) -> EvalRow:
    rng = random.Random(seed)
    base_rr = model_rr = 0.0
    hit1 = 0
    for src, dst, time_value in positives:
        candidates = make_candidates(dst, src, all_dsts, src_pool, all_test, popular, strategy, rng)
        x = store.query_features(src, candidates, time_value)
        scores = predict(booster, x)
        rr, _ = reciprocal_rank(candidates, x[:, 0], dst)
        base_rr += rr
        rr, rank = reciprocal_rank(candidates, scores, dst)
        model_rr += rr
        hit1 += int(rank == 1)
    n = len(positives)
    return EvalRow(scene=scene, strategy=strategy, positives=n, base_mrr=base_rr / n, model_mrr=model_rr / n, hit1=hit1 / n)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    reports: list[EvalRow] = []
    scene_models: dict[str, tuple[FeatureStore, xgb.Booster]] = {}
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
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtrain.set_group(group_train)
            dvalid = xgb.DMatrix(x_valid, label=y_valid)
            dvalid.set_group(group_valid)
            params = {
                "objective": "rank:ndcg",
                "eval_metric": "ndcg@10",
                "tree_method": "hist",
                "max_depth": args.max_depth,
                "eta": args.eta,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "lambda": 1.0,
                "seed": args.seed,
                "nthread": args.nthread,
            }
            booster = xgb.train(params, dtrain, num_boost_round=args.num_boost_round, evals=[(dvalid, "valid")], verbose_eval=50)
            for strategy in ("test_pool", "src_pool", "mixed"):
                reports.append(evaluate_xgb(booster, store, valid_pos, all_dsts, src_pool, all_test, popular, strategy, args.seed + 19, scene))
            full_store = FeatureStore([(s, d, t) for s, d, t, _ in rows], src_pool, all_test)
            scene_models[scene] = (full_store, booster)

        if not args.no_write_submission:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
                for scene, (store, booster) in scene_models.items():
                    print(f"[{scene}] write submission", flush=True)
                    write_scene(data_zip, output_zip, scene, store, booster)

    print("scene,strategy,n,base_mrr,model_mrr,hit1")
    for row in reports:
        print(f"{row.scene},{row.strategy},{row.positives},{row.base_mrr:.8f},{row.model_mrr:.8f},{row.hit1:.8f}")
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps([asdict(row) for row in reports], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
