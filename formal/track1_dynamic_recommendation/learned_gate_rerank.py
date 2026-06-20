"""Learn a local gate for applying a strong secondary Track 1 ranker.

The gate is trained on labeled train-only validation queries. It decides when a
secondary model should replace the online-verified anchor top ranks. This keeps
the high-variance secondary signal focused on rows where it is expected to gain
reciprocal rank.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rank_utils import rank_probabilities
from train_lightgbm_ranker import (
    StructuralContext,
    build_rank_data,
    fit_history,
    make_candidates,
    predict_ranker,
    query_features,
    read_rows,
    reciprocal_rank,
    split_for_scene,
)
from validate_heuristic import load_test_candidate_pools


@dataclass
class GateReport:
    mode: str
    threshold: float
    validation_rows: int
    changed_frac: float
    base_mrr: float
    gated_mrr: float
    rr_gain: float
    test_changed_frac: float
    test_top1_changed_frac: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learned gate for secondary rerank")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="dataset2")
    parser.add_argument("--strategy", choices=("test_pool", "hard", "mixed", "temporal_struct_hard"), default="test_pool")
    parser.add_argument("--feature-set", choices=("base", "cheap", "structural"), default="base")
    parser.add_argument("--train-queries", type=int, default=30000)
    parser.add_argument("--valid-queries", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-change-frac", type=float, default=0.35)
    parser.add_argument("--modes", default="promote,topk")
    parser.add_argument("--report-json", type=Path, default=Path("/tmp/learned_gate_report.json"))
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def read_submission_rows(zip_path: Path, scene: str) -> list[list[float]]:
    rows: list[list[float]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"{scene}.csv") as file:
            reader = csv.reader(line.decode("utf-8") for line in file)
            for row in reader:
                rows.append([float(value) for value in row if value != ""])
    return rows


def order(values: list[float] | np.ndarray) -> list[int]:
    return sorted(range(len(values)), key=lambda index: values[index], reverse=True)


def ranks(values: list[float] | np.ndarray) -> list[int]:
    ranked = order(values)
    result = [0] * len(ranked)
    for rank, index in enumerate(ranked, start=1):
        result[index] = rank
    return result


def gate_features(anchor_scores: np.ndarray, secondary_scores: np.ndarray) -> list[float]:
    anchor_order = order(anchor_scores)
    secondary_order = order(secondary_scores)
    anchor_rank = ranks(anchor_scores)
    secondary_rank = ranks(secondary_scores)
    anchor_top = anchor_order[0]
    secondary_top = secondary_order[0]
    anchor_margin = anchor_scores[anchor_order[0]] - anchor_scores[anchor_order[1]]
    secondary_margin = secondary_scores[secondary_order[0]] - secondary_scores[secondary_order[1]]
    return [
        1.0 if anchor_top != secondary_top else 0.0,
        anchor_rank[secondary_top],
        secondary_rank[anchor_top],
        float(anchor_margin),
        float(secondary_margin),
        float(secondary_scores[secondary_top] - secondary_scores[anchor_top]),
        float(anchor_scores[secondary_top] - anchor_scores[anchor_top]),
        len(set(anchor_order[:10]) & set(secondary_order[:10])) / 10.0,
    ]


def apply_mode(anchor_scores: np.ndarray, secondary_scores: np.ndarray, mode: str, top_k: int) -> np.ndarray:
    anchor_order = order(anchor_scores)
    secondary_order = order(secondary_scores)
    if mode == "promote":
        secondary_top = secondary_order[0]
        final_order = [secondary_top] + [index for index in anchor_order if index != secondary_top]
    elif mode == "topk":
        top_set = set(anchor_order[:top_k])
        reranked = [index for index in secondary_order if index in top_set]
        final_order = reranked + [index for index in anchor_order if index not in top_set]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    scores = np.zeros_like(anchor_scores)
    for rank, index in enumerate(final_order, start=1):
        scores[index] = len(anchor_scores) + 1 - rank
    return scores


def build_validation_examples(
    ranker,
    history_model,
    structural_context: StructuralContext,
    positives: list[tuple[int, int, int]],
    all_dsts: list[int],
    popular_dsts: list[int],
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]], list[dict[str, float]]]:
    rng = random.Random(args.seed + 101)
    xs: list[list[float]] = []
    ys: list[float] = []
    mode_gains: dict[str, list[float]] = {mode: [] for mode in args.modes.split(",") if mode}
    rows: list[dict[str, float]] = []
    for src, dst, time_value in positives:
        candidates = make_candidates(
            dst,
            src,
            args.strategy,
            all_dsts,
            popular_dsts,
            src_test_candidates,
            all_test_candidates,
            100,
            rng,
        )
        features = query_features(history_model, src, candidates, time_value, structural_context, args.feature_set)
        anchor_scores = features[:, 0]
        secondary_scores = predict_ranker(ranker, features)
        _, base_rank = reciprocal_rank(candidates, anchor_scores, dst)
        base_rr = 1.0 / base_rank
        best_gain = 0.0
        per_mode_gain: dict[str, float] = {}
        for mode in mode_gains:
            gated_scores = apply_mode(anchor_scores, secondary_scores, mode, args.top_k)
            _, gated_rank = reciprocal_rank(candidates, gated_scores, dst)
            gain = 1.0 / gated_rank - base_rr
            per_mode_gain[mode] = gain
            mode_gains[mode].append(gain)
            best_gain = max(best_gain, gain)
        xs.append(gate_features(anchor_scores, secondary_scores))
        ys.append(1.0 if best_gain > 0.0 else 0.0)
        rows.append({"base_rr": base_rr, **per_mode_gain})
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32), mode_gains, rows


def choose_threshold(probabilities: np.ndarray, gains: list[float], max_change_frac: float) -> tuple[float, float, float]:
    order_idx = np.argsort(-probabilities)
    best_threshold = 1.0
    best_gain = -1e9
    best_frac = 0.0
    limit = max(1, int(len(probabilities) * max_change_frac))
    total = 0.0
    for rank, index in enumerate(order_idx[:limit], start=1):
        total += gains[int(index)]
        avg_gain = total / len(probabilities)
        if avg_gain > best_gain:
            best_gain = avg_gain
            best_threshold = float(probabilities[int(index)])
            best_frac = rank / len(probabilities)
    return best_threshold, best_gain, best_frac


def write_gated_submission(
    output: Path,
    anchor: Path,
    secondary: Path,
    scene: str,
    gate_model,
    threshold: float,
    mode: str,
    top_k: int,
    max_change_frac: float,
) -> tuple[float, float]:
    output.parent.mkdir(parents=True, exist_ok=True)
    anchor_rows = read_submission_rows(anchor, scene)
    secondary_rows = read_submission_rows(secondary, scene)
    features = np.asarray([gate_features(np.asarray(a), np.asarray(s)) for a, s in zip(anchor_rows, secondary_rows)], dtype=np.float32)
    probabilities = np.asarray(gate_model.predict_proba(features)[:, 1], dtype=np.float32)
    limit = int(len(anchor_rows) * max_change_frac)
    selected = set(np.argsort(-probabilities)[:limit][probabilities[np.argsort(-probabilities)[:limit]] >= threshold])
    changed = 0
    top1_changed = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        with zipfile.ZipFile(anchor) as anchor_zip:
            for current_scene in ("dataset1", "dataset2"):
                if current_scene != scene:
                    output_zip.writestr(f"{current_scene}.csv", anchor_zip.read(f"{current_scene}.csv"))
                    continue
                with output_zip.open(f"{current_scene}.csv", "w") as raw_output:
                    with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                        writer = csv.writer(text_output, lineterminator="\n")
                        for index, (anchor_values, secondary_values) in enumerate(zip(anchor_rows, secondary_rows)):
                            anchor_np = np.asarray(anchor_values, dtype=np.float32)
                            secondary_np = np.asarray(secondary_values, dtype=np.float32)
                            if index in selected:
                                scores = apply_mode(anchor_np, secondary_np, mode, top_k)
                                changed += 1
                                top1_changed += int(order(anchor_np)[0] != order(scores)[0])
                            else:
                                scores = anchor_np
                            writer.writerow([f"{value:.8f}" for value in rank_probabilities(scores.tolist())])
    denom = max(1, len(anchor_rows))
    return changed / denom, top1_changed / denom


def main() -> None:
    args = parse_args()
    from lightgbm import LGBMClassifier, LGBMRanker

    rng = random.Random(args.seed)
    with zipfile.ZipFile(args.data_zip) as data_zip:
        rows = read_rows(data_zip, args.scene)
        history_rows, positive_rows = split_for_scene(rows, args.scene, 0.15)
        rng.shuffle(positive_rows)
        ranker_pos = positive_rows[: args.train_queries]
        gate_pos = positive_rows[args.train_queries : args.train_queries + args.valid_queries]
        if not ranker_pos or not gate_pos:
            raise RuntimeError("Not enough positives for ranker and gate.")

        src_test_candidates, all_test_candidates = load_test_candidate_pools(data_zip, args.scene)
        history_model = fit_history(history_rows)
        structural_context = None
        if args.feature_set != "base":
            structural_context = StructuralContext(history_rows, src_test_candidates, all_test_candidates)
        all_dsts = sorted({dst for _, dst, _ in history_rows})
        popular_dsts = [dst for dst, count in history_model.dst_count.most_common(10000)]

        x_rank, y_rank, group_rank = build_rank_data(
            history_model,
            structural_context,
            ranker_pos,
            all_dsts,
            popular_dsts,
            src_test_candidates,
            all_test_candidates,
            args.strategy,
            args.seed,
            args.feature_set,
        )
        ranker = LGBMRanker(
            objective="rank_xendcg",
            metric="ndcg",
            n_estimators=420,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=args.seed,
            n_jobs=12,
            label_gain=[0, 1],
            verbose=-1,
        )
        ranker.fit(x_rank, y_rank, group=group_rank)

        gate_x, gate_y, mode_gains, validation_rows = build_validation_examples(
            ranker,
            history_model,
            structural_context,
            gate_pos,
            all_dsts,
            popular_dsts,
            src_test_candidates,
            all_test_candidates,
            args,
        )
        gate = LGBMClassifier(
            objective="binary",
            n_estimators=220,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=60,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=args.seed,
            n_jobs=12,
            verbose=-1,
        )
        gate.fit(gate_x, gate_y)
        probabilities = np.asarray(gate.predict_proba(gate_x)[:, 1], dtype=np.float32)
        base_mrr = sum(row["base_rr"] for row in validation_rows) / len(validation_rows)

        reports: list[GateReport] = []
        best: tuple[float, str, float] | None = None
        for mode, gains in mode_gains.items():
            threshold, gain, changed_frac = choose_threshold(probabilities, gains, args.max_change_frac)
            gated_mrr = base_mrr + gain
            if best is None or gain > best[0]:
                best = (gain, mode, threshold)
            reports.append(
                GateReport(
                    mode=mode,
                    threshold=threshold,
                    validation_rows=len(validation_rows),
                    changed_frac=changed_frac,
                    base_mrr=base_mrr,
                    gated_mrr=gated_mrr,
                    rr_gain=gain,
                    test_changed_frac=0.0,
                    test_top1_changed_frac=0.0,
                )
            )

        if best is None:
            raise RuntimeError("No gate modes configured.")
        _, best_mode, best_threshold = best
        test_changed, test_top1_changed = write_gated_submission(
            args.output,
            args.anchor,
            args.secondary,
            args.scene,
            gate,
            best_threshold,
            best_mode,
            args.top_k,
            args.max_change_frac,
        )
        for index, report in enumerate(reports):
            if report.mode == best_mode and abs(report.threshold - best_threshold) < 1e-9:
                reports[index] = GateReport(
                    **{
                        **asdict(report),
                        "test_changed_frac": test_changed,
                        "test_top1_changed_frac": test_top1_changed,
                    }
                )

        for report in reports:
            print(
                f"{report.mode}: threshold={report.threshold:.6f} changed={report.changed_frac:.4f} "
                f"base={report.base_mrr:.8f} gated={report.gated_mrr:.8f} gain={report.rr_gain:.8f} "
                f"test_changed={report.test_changed_frac:.4f} test_top1_changed={report.test_top1_changed_frac:.4f}",
                flush=True,
            )
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps([asdict(report) for report in reports], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
