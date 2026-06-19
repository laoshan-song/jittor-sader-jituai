"""Promote a secondary model only on high-confidence rows.

The online-calibrated anchor remains the default. A secondary model may promote
its top candidate to rank 1 only when the secondary top candidate is already
near the anchor top ranks and the anchor top candidate is weak under the
secondary model. This keeps drift measurable and deliberately small.
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

from rank_utils import rank_probabilities


SCENES = ("dataset1", "dataset2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gated top candidate promotion")
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="dataset2")
    parser.add_argument("--max-anchor-rank", type=int, default=3)
    parser.add_argument("--min-secondary-rank-of-anchor-top1", type=int, default=20)
    parser.add_argument("--max-promote-frac", type=float, default=0.03)
    return parser.parse_args()


def read_rows(zip_path: Path, scene: str) -> list[list[float]]:
    rows: list[list[float]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"{scene}.csv") as file:
            reader = csv.reader(line.decode("utf-8") for line in file)
            for row in reader:
                rows.append([float(value) for value in row if value != ""])
    return rows


def ranks(values: list[float]) -> tuple[list[int], list[int]]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    rank_by_index = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        rank_by_index[index] = rank
    return order, rank_by_index


def promoted_scores(anchor_values: list[float], promote_index: int) -> list[float]:
    anchor_order, _ = ranks(anchor_values)
    final_order = [promote_index] + [index for index in anchor_order if index != promote_index]
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
                if scene != args.scene:
                    output_zip.writestr(f"{scene}.csv", anchor_zip.read(f"{scene}.csv"))
                    continue

                anchor_rows = read_rows(args.anchor, scene)
                secondary_rows = read_rows(args.secondary, scene)
                limit = int(len(anchor_rows) * args.max_promote_frac)
                promoted = 0
                with output_zip.open(f"{scene}.csv", "w") as raw_output:
                    with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                        writer = csv.writer(text_output, lineterminator="\n")
                        for anchor_values, secondary_values in zip(anchor_rows, secondary_rows):
                            anchor_order, anchor_rank = ranks(anchor_values)
                            secondary_order, secondary_rank = ranks(secondary_values)
                            secondary_top = secondary_order[0]
                            anchor_top = anchor_order[0]
                            should_promote = (
                                promoted < limit
                                and secondary_top != anchor_top
                                and anchor_rank[secondary_top] <= args.max_anchor_rank
                                and secondary_rank[anchor_top] >= args.min_secondary_rank_of_anchor_top1
                            )
                            if should_promote:
                                scores = promoted_scores(anchor_values, secondary_top)
                                promoted += 1
                            else:
                                scores = anchor_values
                            writer.writerow([f"{value:.8f}" for value in rank_probabilities(scores)])
                print(f"{scene}: promoted={promoted}/{len(anchor_rows)}", flush=True)


if __name__ == "__main__":
    main()
