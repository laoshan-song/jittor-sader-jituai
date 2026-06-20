"""Fast confidence gate between an online anchor and secondary submission."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np

from rank_utils import rank_probabilities


SCENES = ("dataset1", "dataset2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast gated rerank using submission confidence features")
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="dataset2")
    parser.add_argument("--mode", choices=("promote", "topk"), default="promote")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--change-frac", type=float, default=0.15)
    parser.add_argument("--min-anchor-rank", type=int, default=5)
    parser.add_argument("--min-secondary-rank-of-anchor-top1", type=int, default=8)
    return parser.parse_args()


def read_rows(zip_path: Path, scene: str) -> list[list[float]]:
    rows: list[list[float]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"{scene}.csv") as file:
            reader = csv.reader(line.decode("utf-8") for line in file)
            for row in reader:
                rows.append([float(value) for value in row if value != ""])
    return rows


def order(values: list[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda index: values[index], reverse=True)


def rank_by_index(values: list[float]) -> list[int]:
    result = [0] * len(values)
    for rank, index in enumerate(order(values), start=1):
        result[index] = rank
    return result


def confidence(anchor: list[float], secondary: list[float], min_anchor_rank: int, min_secondary_rank_of_anchor_top1: int) -> float:
    anchor_order = order(anchor)
    secondary_order = order(secondary)
    anchor_rank = rank_by_index(anchor)
    secondary_rank = rank_by_index(secondary)
    secondary_top = secondary_order[0]
    anchor_top = anchor_order[0]
    if secondary_top == anchor_top:
        return -1.0
    if anchor_rank[secondary_top] > min_anchor_rank:
        return -1.0
    if secondary_rank[anchor_top] < min_secondary_rank_of_anchor_top1:
        return -1.0
    top10_overlap = len(set(anchor_order[:10]) & set(secondary_order[:10])) / 10.0
    sec_margin = secondary[secondary_order[0]] - secondary[secondary_order[1]]
    anchor_disagree_margin = secondary[secondary_top] - secondary[anchor_top]
    anchor_acceptance = anchor[secondary_top] - anchor[anchor_top]
    return float(2.0 * sec_margin + anchor_disagree_margin + 0.25 * anchor_acceptance + top10_overlap)


def gated_scores(anchor: list[float], secondary: list[float], mode: str, top_k: int) -> list[float]:
    anchor_order = order(anchor)
    secondary_order = order(secondary)
    if mode == "promote":
        promote = secondary_order[0]
        final_order = [promote] + [index for index in anchor_order if index != promote]
    else:
        top_set = set(anchor_order[:top_k])
        reranked = [index for index in secondary_order if index in top_set]
        final_order = reranked + [index for index in anchor_order if index not in top_set]
    scores = [0.0] * len(anchor)
    for rank, index in enumerate(final_order, start=1):
        scores[index] = len(anchor) + 1 - rank
    return scores


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    anchor_rows = read_rows(args.anchor, args.scene)
    secondary_rows = read_rows(args.secondary, args.scene)
    scores = [
        confidence(anchor, secondary, args.min_anchor_rank, args.min_secondary_rank_of_anchor_top1)
        for anchor, secondary in zip(anchor_rows, secondary_rows)
    ]
    limit = int(len(scores) * args.change_frac)
    selected = {
        int(index)
        for index in np.argsort(-np.asarray(scores))[:limit]
        if scores[int(index)] > 0.0
    }
    changed = 0
    top1_changed = 0
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        with zipfile.ZipFile(args.anchor) as anchor_zip:
            for scene in SCENES:
                if scene != args.scene:
                    output_zip.writestr(f"{scene}.csv", anchor_zip.read(f"{scene}.csv"))
                    continue
                with output_zip.open(f"{scene}.csv", "w") as raw_output:
                    with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                        writer = csv.writer(text_output, lineterminator="\n")
                        for index, (anchor, secondary) in enumerate(zip(anchor_rows, secondary_rows)):
                            if index in selected:
                                row_scores = gated_scores(anchor, secondary, args.mode, args.top_k)
                                changed += 1
                                top1_changed += int(order(anchor)[0] != order(row_scores)[0])
                            else:
                                row_scores = anchor
                            writer.writerow([f"{value:.8f}" for value in rank_probabilities(row_scores)])
    total = max(1, len(anchor_rows))
    print(
        f"changed={changed}/{total} changed_frac={changed/total:.4f} "
        f"top1_changed_frac={top1_changed/total:.4f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
