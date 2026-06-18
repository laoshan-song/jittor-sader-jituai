"""Fuse Track 1 submission rankings into a rank-preserving submission."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

from train_mf_rerank import rank_probabilities


SCENES = ("dataset1", "dataset2")


def parse_weighted_zip(value: str) -> tuple[Path, float]:
    if ":" not in value:
        return Path(value), 1.0
    path, weight = value.rsplit(":", 1)
    return Path(path), float(weight)


def parse_scene_weights(value: str | None) -> dict[str, list[float]]:
    if not value:
        return {}
    result: dict[str, list[float]] = {}
    for chunk in value.split(";"):
        if not chunk.strip():
            continue
        scene, weights = chunk.split("=", 1)
        result[scene.strip()] = [float(weight) for weight in weights.split(",") if weight.strip()]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse submission zips by candidate rank")
    parser.add_argument("--inputs", nargs="+", required=True, help="zip[:weight] entries")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("borda", "rrf"),
        default="borda",
        help="Borda uses 100-rank; RRF uses 1/(k+rank).",
    )
    parser.add_argument("--rrf-k", type=float, default=20.0)
    parser.add_argument(
        "--scene-weights",
        help="Optional overrides like 'dataset1=1,0;dataset2=3,1'.",
    )
    return parser.parse_args()


def read_scene_rows(zip_path: Path, scene: str) -> list[list[float]]:
    rows: list[list[float]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"{scene}.csv") as file:
            reader = csv.reader(line.decode("utf-8") for line in file)
            for row in reader:
                rows.append([float(value) for value in row if value != ""])
    return rows


def fuse_scores(values_by_model: list[list[float]], weights: list[float], method: str, rrf_k: float) -> list[float]:
    count = len(values_by_model[0])
    scores = [0.0] * count
    for values, weight in zip(values_by_model, weights):
        order = sorted(range(count), key=lambda index: values[index], reverse=True)
        for rank, index in enumerate(order, start=1):
            if method == "borda":
                scores[index] += weight * (count + 1 - rank)
            else:
                scores[index] += weight / (rrf_k + rank)
    return scores


def main() -> None:
    args = parse_args()
    weighted_inputs = [parse_weighted_zip(value) for value in args.inputs]
    paths = [path for path, _ in weighted_inputs]
    weights = [weight for _, weight in weighted_inputs]
    scene_weights = parse_scene_weights(args.scene_weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        for scene in SCENES:
            active_weights = scene_weights.get(scene, weights)
            if len(active_weights) != len(paths):
                raise ValueError(f"{scene} has {len(active_weights)} weights for {len(paths)} inputs.")
            scene_rows = [read_scene_rows(path, scene) for path in paths]
            row_count = len(scene_rows[0])
            if any(len(rows) != row_count for rows in scene_rows):
                raise ValueError(f"Input row count mismatch for {scene}.")

            with output_zip.open(f"{scene}.csv", "w") as raw_output:
                with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                    writer = csv.writer(text_output, lineterminator="\n")
                    for row_index in range(row_count):
                        values_by_model = [rows[row_index] for rows in scene_rows]
                        fused = fuse_scores(values_by_model, active_weights, args.method, args.rrf_k)
                        probs = rank_probabilities(fused)
                        writer.writerow([f"{value:.8f}" for value in probs])

    print(f"fused submission saved to {args.output}")


if __name__ == "__main__":
    main()
