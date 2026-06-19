"""Stratified rank fusion for Track 1 submissions.

Rows with different repeated-pair counts behave differently. This tool fuses a
trusted primary submission with a secondary signal using a separate secondary
rank weight for each repeat-count bucket.
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
    parser = argparse.ArgumentParser(description="Stratified rank fusion")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-weight", type=float, default=8.0)
    parser.add_argument(
        "--repeat-weights",
        default="0:3,1:3,2:0,3:0,4:0,5:0",
        help="Comma-separated bucket weights. Last key also applies to larger counts.",
    )
    parser.add_argument("--dataset1-mode", choices=("primary", "fuse"), default="primary")
    return parser.parse_args()


def parse_repeat_weights(value: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    for chunk in value.split(","):
        if not chunk.strip():
            continue
        key, weight = chunk.split(":", 1)
        weights[int(key)] = float(weight)
    if not weights:
        raise ValueError("--repeat-weights cannot be empty")
    return weights


def weight_for_repeat(repeat_count: int, weights: dict[int, float]) -> float:
    if repeat_count in weights:
        return weights[repeat_count]
    return weights[max(key for key in weights if key <= repeat_count)]


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
    for rank, index in enumerate(sorted(range(count), key=lambda item: values[item], reverse=True), start=1):
        scores[index] = weight * (count + 1 - rank)
    return scores


def copy_scene(primary_zip: zipfile.ZipFile, output_zip: zipfile.ZipFile, scene: str) -> None:
    output_zip.writestr(f"{scene}.csv", primary_zip.read(f"{scene}.csv"))


def write_fused_scene(
    data_zip: zipfile.ZipFile,
    output_zip: zipfile.ZipFile,
    scene: str,
    primary_rows: list[list[float]],
    secondary_rows: list[list[float]],
    primary_weight: float,
    repeat_weights: dict[int, float],
) -> None:
    train_pairs = load_train_pairs(data_zip, scene)
    test_rows = load_test_rows(data_zip, scene)
    if len(primary_rows) != len(test_rows) or len(secondary_rows) != len(test_rows):
        raise ValueError(f"Row count mismatch for {scene}.")

    with output_zip.open(f"{scene}.csv", "w") as raw_output:
        with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
            writer = csv.writer(text_output, lineterminator="\n")
            for row_index, (src, candidates) in enumerate(test_rows):
                repeat_count = sum((src, dst) in train_pairs for dst in candidates)
                secondary_weight = weight_for_repeat(repeat_count, repeat_weights)
                primary_scores = rank_scores(primary_rows[row_index], primary_weight)
                secondary_scores = rank_scores(secondary_rows[row_index], secondary_weight)
                fused = [a + b for a, b in zip(primary_scores, secondary_scores)]
                writer.writerow([f"{value:.8f}" for value in rank_probabilities(fused)])


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    repeat_weights = parse_repeat_weights(args.repeat_weights)
    with zipfile.ZipFile(args.data_zip) as data_zip:
        with zipfile.ZipFile(args.primary) as primary_zip:
            with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
                for scene in SCENES:
                    if scene == "dataset1" and args.dataset1_mode == "primary":
                        copy_scene(primary_zip, output_zip, scene)
                        continue
                    primary_rows = read_submission_rows(args.primary, scene)
                    secondary_rows = read_submission_rows(args.secondary, scene)
                    write_fused_scene(
                        data_zip,
                        output_zip,
                        scene,
                        primary_rows,
                        secondary_rows,
                        args.primary_weight,
                        repeat_weights,
                    )
    print(f"stratified fused submission saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
