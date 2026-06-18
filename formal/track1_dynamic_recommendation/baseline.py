"""Heuristic baseline for Track 1 dynamic recommendation.

The submission format contains one CSV file per scene. Each row has 100
probabilities, matching the 100 candidate destinations in the test row.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SCENE_PATTERN = re.compile(r"([^/]+)/(?P<kind>train|test)\.csv$")


@dataclass
class SceneStats:
    train_rows: int
    test_rows: int


class HistoryBaseline:
    """Score candidates from repeated interactions, recency, and popularity."""

    def __init__(
        self,
        pair_weight: float,
        pair_recency_weight: float,
        dst_pop_weight: float,
        dst_recency_weight: float,
    ) -> None:
        self.pair_weight = pair_weight
        self.pair_recency_weight = pair_recency_weight
        self.dst_pop_weight = dst_pop_weight
        self.dst_recency_weight = dst_recency_weight
        self.src_dst_count: dict[int, Counter[int]] = defaultdict(Counter)
        self.src_dst_last_time: dict[int, dict[int, int]] = defaultdict(dict)
        self.dst_count: Counter[int] = Counter()
        self.dst_last_time: dict[int, int] = {}
        self.min_time: int | None = None
        self.max_time: int | None = None
        self.max_dst_log = 1.0
        self.time_scale = 1.0
        self.train_rows = 0

    def update(self, src: int, dst: int, time_value: int) -> None:
        self.train_rows += 1
        self.src_dst_count[src][dst] += 1
        self.src_dst_last_time[src][dst] = max(
            time_value,
            self.src_dst_last_time[src].get(dst, time_value),
        )
        self.dst_count[dst] += 1
        self.dst_last_time[dst] = max(time_value, self.dst_last_time.get(dst, time_value))

        if self.min_time is None or time_value < self.min_time:
            self.min_time = time_value
        if self.max_time is None or time_value > self.max_time:
            self.max_time = time_value

    def finalize(self) -> None:
        self.max_dst_log = max((math.log1p(value) for value in self.dst_count.values()), default=1.0)
        if self.min_time is None or self.max_time is None:
            self.time_scale = 1.0
            return

        time_span = max(1, self.max_time - self.min_time)
        self.time_scale = max(1.0, time_span / 20.0)

    def score(self, src: int, dst: int, query_time: int) -> float:
        src_counts = self.src_dst_count.get(src)
        pair_count = src_counts.get(dst, 0) if src_counts is not None else 0
        pair_last_time = self.src_dst_last_time.get(src, {}).get(dst)

        score = 0.0
        if pair_count:
            score += self.pair_weight * math.log1p(pair_count)
            score += self.pair_recency_weight * self._recency(query_time, pair_last_time)

        dst_count = self.dst_count.get(dst, 0)
        if dst_count:
            score += self.dst_pop_weight * math.log1p(dst_count) / self.max_dst_log
            score += self.dst_recency_weight * self._recency(
                query_time,
                self.dst_last_time.get(dst),
            )

        score += self._stable_tie_break(src, dst)
        return score

    def _recency(self, query_time: int, last_time: int | None) -> float:
        if last_time is None:
            return 0.0
        age = max(0, query_time - last_time)
        return 1.0 / (1.0 + age / self.time_scale)

    @staticmethod
    def _stable_tie_break(src: int, dst: int) -> float:
        return ((src * 1_000_003 + dst) % 997) * 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Track 1 result.zip")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/track1/result.zip"))
    parser.add_argument("--pair-weight", type=float, default=4.0)
    parser.add_argument("--pair-recency-weight", type=float, default=3.0)
    parser.add_argument("--dst-pop-weight", type=float, default=0.8)
    parser.add_argument("--dst-recency-weight", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--uniform-mix", type=float, default=0.02)
    parser.add_argument("--limit-test-rows", type=int, default=0)
    return parser.parse_args()


def discover_scenes(data_zip: zipfile.ZipFile) -> list[str]:
    seen: dict[str, set[str]] = defaultdict(set)
    for name in data_zip.namelist():
        match = SCENE_PATTERN.match(name)
        if match:
            scene = match.group(1)
            seen[scene].add(match.group("kind"))

    scenes = sorted(scene for scene, kinds in seen.items() if kinds == {"train", "test"})
    if not scenes:
        raise ValueError("No dataset scenes with train.csv and test.csv were found.")
    return scenes


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def fit_scene_model(
    data_zip: zipfile.ZipFile,
    scene: str,
    args: argparse.Namespace,
) -> HistoryBaseline:
    model = HistoryBaseline(
        pair_weight=args.pair_weight,
        pair_recency_weight=args.pair_recency_weight,
        dst_pop_weight=args.dst_pop_weight,
        dst_recency_weight=args.dst_recency_weight,
    )

    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            model.update(int(row["src"]), int(row["dst"]), int(row["time"]))

    model.finalize()
    return model


def probabilities(scores: list[float], temperature: float, uniform_mix: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("--temperature must be greater than 0.")
    if not 0 <= uniform_mix < 1:
        raise ValueError("--uniform-mix must be in [0, 1).")

    scaled = [score / temperature for score in scores]
    max_score = max(scaled)
    exp_scores = [math.exp(score - max_score) for score in scaled]
    total = sum(exp_scores)
    base = [value / total for value in exp_scores]
    uniform = 1.0 / len(base)
    return [(1.0 - uniform_mix) * value + uniform_mix * uniform for value in base]


def write_scene_predictions(
    data_zip: zipfile.ZipFile,
    output_zip: zipfile.ZipFile,
    scene: str,
    model: HistoryBaseline,
    args: argparse.Namespace,
) -> SceneStats:
    test_rows = 0
    with open_csv(data_zip, f"{scene}/test.csv") as input_file:
        reader = csv.reader(input_file)
        header = next(reader)
        if len(header) != 102:
            raise ValueError(f"{scene}/test.csv should contain src, time, and 100 candidates.")

        with output_zip.open(f"{scene}.csv", "w") as raw_output:
            with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                writer = csv.writer(text_output, lineterminator="\n")
                for row in reader:
                    src = int(row[0])
                    query_time = int(row[1])
                    candidates = [int(value) for value in row[2:]]
                    if len(candidates) != 100:
                        raise ValueError(f"{scene}/test.csv row {test_rows + 2} has {len(candidates)} candidates.")

                    scores = [model.score(src, dst, query_time) for dst in candidates]
                    probs = probabilities(scores, args.temperature, args.uniform_mix)
                    writer.writerow([f"{value:.8f}" for value in probs])

                    test_rows += 1
                    if args.limit_test_rows and test_rows >= args.limit_test_rows:
                        break

    return SceneStats(train_rows=model.train_rows, test_rows=test_rows)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.data_zip) as data_zip:
        scenes = discover_scenes(data_zip)
        print(f"scenes: {', '.join(scenes)}")

        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
            for scene in scenes:
                print(f"[{scene}] fitting history baseline")
                model = fit_scene_model(data_zip, scene, args)
                print(f"[{scene}] train rows: {model.train_rows}")
                stats = write_scene_predictions(data_zip, output_zip, scene, model, args)
                print(f"[{scene}] wrote {stats.test_rows} prediction rows")

    print(f"submission saved to {args.output}")


if __name__ == "__main__":
    main()
