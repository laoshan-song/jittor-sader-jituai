"""Route Track 1 rows between XGBoost expert submissions."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


SCENES = ("dataset1", "dataset2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route dataset2 rows between expert submissions")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="Full submission used for dataset1 and default dataset2")
    parser.add_argument("--src-expert", type=Path, required=True)
    parser.add_argument("--mixed-expert", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--src-candidate-threshold", type=int, default=80)
    parser.add_argument("--unseen-ratio-threshold", type=float, default=0.45)
    parser.add_argument("--popular-hit-threshold", type=int, default=12)
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def read_rows(zip_path: Path, scene: str) -> list[bytes]:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(f"{scene}.csv").splitlines()


def train_stats(data_zip: zipfile.ZipFile, scene: str) -> tuple[set[int], Counter[int]]:
    dsts: set[int] = set()
    counts: Counter[int] = Counter()
    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dst = int(row["dst"])
            dsts.add(dst)
            counts[dst] += 1
    return dsts, counts


def source_candidate_sizes(data_zip: zipfile.ZipFile, scene: str) -> dict[int, int]:
    pools: dict[int, set[int]] = defaultdict(set)
    with open_csv(data_zip, f"{scene}/test.csv") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            pools[int(row[0])].update(int(value) for value in row[2:])
    return {src: len(cands) for src, cands in pools.items()}


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.data_zip) as data_zip:
        seen_dsts, dst_counts = train_stats(data_zip, "dataset2")
        popular = {dst for dst, _ in dst_counts.most_common(2000)}
        src_pool_sizes = source_candidate_sizes(data_zip, "dataset2")
        base_d2 = read_rows(args.base, "dataset2")
        src_d2 = read_rows(args.src_expert, "dataset2")
        mixed_d2 = read_rows(args.mixed_expert, "dataset2")
        src_used = 0
        mixed_used = 0
        total = 0
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
            with zipfile.ZipFile(args.base) as base_zip:
                output_zip.writestr("dataset1.csv", base_zip.read("dataset1.csv"))
            with open_csv(data_zip, "dataset2/test.csv") as test_file:
                reader = csv.reader(test_file)
                next(reader)
                with output_zip.open("dataset2.csv", "w") as raw_output:
                    for row_index, row in enumerate(reader):
                        src = int(row[0])
                        candidates = [int(value) for value in row[2:]]
                        unseen_ratio = sum(dst not in seen_dsts for dst in candidates) / len(candidates)
                        popular_hits = sum(dst in popular for dst in candidates)
                        pool_size = src_pool_sizes.get(src, 0)
                        if pool_size >= args.src_candidate_threshold and unseen_ratio < args.unseen_ratio_threshold:
                            raw_output.write(src_d2[row_index] + b"\n")
                            src_used += 1
                        elif popular_hits >= args.popular_hit_threshold or unseen_ratio < args.unseen_ratio_threshold / 2:
                            raw_output.write(mixed_d2[row_index] + b"\n")
                            mixed_used += 1
                        else:
                            raw_output.write(base_d2[row_index] + b"\n")
                        total += 1
    print(f"routed total={total} src_used={src_used} mixed_used={mixed_used}", flush=True)


if __name__ == "__main__":
    main()
