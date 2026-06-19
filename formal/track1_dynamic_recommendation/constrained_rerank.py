"""Constrain a strong secondary model to rerank inside anchor top-k buckets."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

from rank_utils import rank_probabilities


SCENES = ("dataset1", "dataset2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained top-k rerank")
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="dataset2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-secondary-margin", type=float, default=0.0)
    parser.add_argument("--lock-top1", action="store_true")
    parser.add_argument("--dataset1-mode", choices=("anchor", "rerank"), default="anchor")
    return parser.parse_args()


def read_rows(zip_path: Path, scene: str) -> list[list[float]]:
    rows: list[list[float]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"{scene}.csv") as file:
            reader = csv.reader(line.decode("utf-8") for line in file)
            for row in reader:
                rows.append([float(value) for value in row if value != ""])
    return rows


def constrained_scores(
    anchor_values: list[float],
    secondary_values: list[float],
    top_k: int,
    min_secondary_margin: float,
    lock_top1: bool,
) -> list[float]:
    anchor_order = sorted(range(len(anchor_values)), key=lambda index: anchor_values[index], reverse=True)
    top_set = set(anchor_order[:top_k])
    if lock_top1:
        locked = anchor_order[0]
        rerank_set = top_set - {locked}
        secondary_top = [locked] + sorted(rerank_set, key=lambda index: secondary_values[index], reverse=True)
    else:
        secondary_top = sorted(top_set, key=lambda index: secondary_values[index], reverse=True)
    if not lock_top1 and secondary_top[0] != anchor_order[0]:
        margin = secondary_values[secondary_top[0]] - secondary_values[anchor_order[0]]
        if margin < min_secondary_margin:
            secondary_top.remove(anchor_order[0])
            secondary_top.insert(0, anchor_order[0])
    final_order = secondary_top + [index for index in anchor_order if index not in top_set]
    scores = [0.0] * len(anchor_values)
    for rank, index in enumerate(final_order, start=1):
        scores[index] = len(anchor_values) + 1 - rank
    return scores


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        with zipfile.ZipFile(args.anchor) as anchor_zip:
            for scene in SCENES:
                if scene != args.scene or (scene == "dataset1" and args.dataset1_mode == "anchor"):
                    output_zip.writestr(f"{scene}.csv", anchor_zip.read(f"{scene}.csv"))
                    continue
                anchor_rows = read_rows(args.anchor, scene)
                secondary_rows = read_rows(args.secondary, scene)
                with output_zip.open(f"{scene}.csv", "w") as raw_output:
                    with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                        writer = csv.writer(text_output, lineterminator="\n")
                        for anchor_values, secondary_values in zip(anchor_rows, secondary_rows):
                            scores = constrained_scores(
                                anchor_values,
                                secondary_values,
                                args.top_k,
                                args.min_secondary_margin,
                                args.lock_top1,
                            )
                            writer.writerow([f"{value:.8f}" for value in rank_probabilities(scores)])
    print(f"constrained rerank saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
