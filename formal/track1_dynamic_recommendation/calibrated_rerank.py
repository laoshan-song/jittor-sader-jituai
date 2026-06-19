"""Calibrated reranker found by official-split validation.

The calibration is deliberately simple and reproducible. It keeps the strong
history/sequence baseline, then applies a dataset2-only correction discovered
on dataset2 split=1 hard-candidate validation:

- penalize repeated source-target candidates that overfit the validation split;
- penalize global popularity slightly;
- strongly boost destination recency;
- penalize very recent local repeats.

This is not used blindly: generate candidates to /tmp first and audit before
copying to outputs/track1/result.zip.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import zipfile
from pathlib import Path

from baseline import HistoryBaseline
from rank_utils import rank_probabilities


HEURISTIC_WEIGHTS = {
    "pair_weight": 6.0,
    "pair_recency_weight": 4.0,
    "dst_pop_weight": 0.4,
    "dst_recency_weight": 0.2,
    "sequence_weight": 2.5,
    "repeat_recent_weight": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate calibrated Track 1 rerank submission")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True, help="Current trusted submission zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-penalty", type=float, default=-4.0)
    parser.add_argument("--dst-pop-penalty", type=float, default=-2.0)
    parser.add_argument("--dst-recency-weight", type=float, default=4.0)
    parser.add_argument("--local-repeat-penalty", type=float, default=-4.0)
    parser.add_argument("--dataset1-mode", choices=("primary", "calibrated"), default="primary")
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def fit_model(data_zip: zipfile.ZipFile, scene: str) -> HistoryBaseline:
    model = HistoryBaseline(**HEURISTIC_WEIGHTS)
    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            model.update(int(row["src"]), int(row["dst"]), int(row["time"]))
    model.finalize()
    return model


def calibrated_score(
    model: HistoryBaseline,
    src: int,
    dst: int,
    query_time: int,
    args: argparse.Namespace,
) -> float:
    score = model.score(src, dst, query_time)
    if dst in model.src_dst_count.get(src, {}):
        score += args.pair_penalty

    dst_count = model.dst_count.get(dst, 0)
    if dst_count:
        score += args.dst_pop_penalty * math.log1p(dst_count) / model.max_dst_log
        score += args.dst_recency_weight * model._recency(query_time, model.dst_last_time.get(dst))

    recent = model.src_recent_dsts.get(src, [])
    if dst in recent[-10:]:
        score += args.local_repeat_penalty
    return score


def copy_scene(primary_zip: zipfile.ZipFile, output_zip: zipfile.ZipFile, scene: str) -> None:
    output_zip.writestr(f"{scene}.csv", primary_zip.read(f"{scene}.csv"))


def write_calibrated_scene(
    data_zip: zipfile.ZipFile,
    output_zip: zipfile.ZipFile,
    scene: str,
    args: argparse.Namespace,
) -> None:
    model = fit_model(data_zip, scene)
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
                    scores = [calibrated_score(model, src, dst, time_value, args) for dst in candidates]
                    probs = rank_probabilities(scores)
                    writer.writerow([f"{value:.8f}" for value in probs])


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.data_zip) as data_zip:
        with zipfile.ZipFile(args.primary) as primary_zip:
            with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
                if args.dataset1_mode == "primary":
                    copy_scene(primary_zip, output_zip, "dataset1")
                else:
                    write_calibrated_scene(data_zip, output_zip, "dataset1", args)
                write_calibrated_scene(data_zip, output_zip, "dataset2", args)
    print(f"submission saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
