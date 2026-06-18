"""Feature-gated fusion of two Track 1 submissions.

The gate is intentionally simple and auditable: rows with many repeated
source-destination candidates stay close to the stable submission, while rows
with fewer repeated candidates borrow more ranking signal from the secondary
submission.
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

from train_mf_rerank import rank_probabilities


SCENES = ("dataset1", "dataset2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive fusion for Track 1 submissions")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-repeat-weight", type=float, default=1.0)
    parser.add_argument("--high-repeat-weight", type=float, default=0.0)
    parser.add_argument("--repeat-threshold", type=int, default=2)
    parser.add_argument("--primary-weight", type=float, default=3.0)
    return parser.parse_args()


def open_text(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def load_train_pairs(data_zip: zipfile.ZipFile, scene: str) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    with open_text(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            pairs.add((int(row["src"]), int(row["dst"])))
    return pairs


def load_test_rows(data_zip: zipfile.ZipFile, scene: str) -> list[tuple[int, list[int]]]:
    rows: list[tuple[int, list[int]]] = []
    with open_text(data_zip, f"{scene}/test.csv") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            rows.append((int(row[0]), [int(value) for value in row[2:]]))
    return rows


def read_submission_rows(zip_path: Path, scene: str) -> list[list[float]]:
    rows: list[list[float]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"{scene}.csv") as file:
            reader = csv.reader(line.decode("utf-8") for line in file)
            for row in reader:
                rows.append([float(value) for value in row if value != ""])
    return rows


def rank_scores(values: list[float], weight: float) -> list[float]:
    count = len(values)
    scores = [0.0] * count
    order = sorted(range(count), key=lambda index: values[index], reverse=True)
    for rank, index in enumerate(order, start=1):
        scores[index] = weight * (count + 1 - rank)
    return scores


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.data_zip) as data_zip:
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
            for scene in SCENES:
                train_pairs = load_train_pairs(data_zip, scene)
                test_rows = load_test_rows(data_zip, scene)
                primary_rows = read_submission_rows(args.primary, scene)
                secondary_rows = read_submission_rows(args.secondary, scene)
                if len(primary_rows) != len(test_rows) or len(secondary_rows) != len(test_rows):
                    raise ValueError(f"Row count mismatch for {scene}.")

                with output_zip.open(f"{scene}.csv", "w") as raw_output:
                    with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                        writer = csv.writer(text_output, lineterminator="\n")
                        for row_index, (src, candidates) in enumerate(test_rows):
                            repeat_count = sum((src, dst) in train_pairs for dst in candidates)
                            secondary_weight = (
                                args.high_repeat_weight
                                if repeat_count >= args.repeat_threshold
                                else args.low_repeat_weight
                            )
                            scores = rank_scores(primary_rows[row_index], args.primary_weight)
                            secondary_scores = rank_scores(secondary_rows[row_index], secondary_weight)
                            fused = [a + b for a, b in zip(scores, secondary_scores)]
                            probs = rank_probabilities(fused)
                            writer.writerow([f"{value:.8f}" for value in probs])

    print(f"adaptive fused submission saved to {args.output}")


if __name__ == "__main__":
    main()
