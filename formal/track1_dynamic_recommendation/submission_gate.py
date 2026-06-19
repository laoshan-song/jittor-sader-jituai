"""Gate candidate submissions using online-calibrated drift guardrails.

This is a conservative pre-submit check. It does not estimate hidden MRR.
Instead, it blocks candidates that resemble known failed submissions: valid
probability files whose rankings drift too far from the best online-verified
anchor submission.
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path


SCENES = ("dataset1", "dataset2")


@dataclass
class SceneAgreement:
    top1: float
    top10_jaccard: float
    top10_exact: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate Track 1 candidate submissions")
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--min-d1-top1", type=float, default=0.99)
    parser.add_argument("--min-d1-top10", type=float, default=0.95)
    parser.add_argument("--min-d2-top1", type=float, default=0.90)
    parser.add_argument("--min-d2-top10", type=float, default=0.75)
    parser.add_argument("--min-d2-top10-exact", type=float, default=0.95)
    return parser.parse_args()


def read_orders(zip_path: Path, scene: str) -> list[list[int]]:
    rows: list[list[int]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"{scene}.csv") as file:
            reader = csv.reader(line.decode("utf-8") for line in file)
            for row in reader:
                values = [float(value) for value in row if value != ""]
                rows.append(sorted(range(len(values)), key=lambda index: values[index], reverse=True)[:10])
    return rows


def agreement(anchor: Path, candidate: Path, scene: str) -> SceneAgreement:
    anchor_orders = read_orders(anchor, scene)
    candidate_orders = read_orders(candidate, scene)
    pairs = list(zip(anchor_orders, candidate_orders))
    if not pairs or len(anchor_orders) != len(candidate_orders):
        return SceneAgreement(0.0, 0.0)
    top1 = sum(a[0] == c[0] for a, c in pairs) / len(pairs)
    top10_exact = sum(a == c for a, c in pairs) / len(pairs)
    jaccard = 0.0
    for anchor_order, candidate_order in pairs:
        a = set(anchor_order)
        c = set(candidate_order)
        jaccard += len(a & c) / len(a | c)
    return SceneAgreement(top1=top1, top10_jaccard=jaccard / len(pairs), top10_exact=top10_exact)


def main() -> None:
    args = parse_args()
    results = {scene: agreement(args.anchor, args.candidate, scene) for scene in SCENES}
    for scene, stats in results.items():
        print(
            f"{scene}: top1={stats.top1:.4f} "
            f"top10_jaccard={stats.top10_jaccard:.4f} "
            f"top10_exact={stats.top10_exact:.4f}"
        )

    failures: list[str] = []
    if results["dataset1"].top1 < args.min_d1_top1:
        failures.append("dataset1 top1 drift")
    if results["dataset1"].top10_jaccard < args.min_d1_top10:
        failures.append("dataset1 top10 drift")
    if results["dataset2"].top1 < args.min_d2_top1:
        failures.append("dataset2 top1 drift")
    if results["dataset2"].top10_jaccard < args.min_d2_top10:
        failures.append("dataset2 top10 drift")
    if results["dataset2"].top10_exact < args.min_d2_top10_exact:
        failures.append("dataset2 top10 order drift")

    if failures:
        print("BLOCK: " + ", ".join(failures))
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
