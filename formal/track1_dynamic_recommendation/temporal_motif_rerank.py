"""Temporal motif reranker for Track 1.

This generator keeps the strong history baseline as the main signal and adds
lightweight dynamic-graph features inspired by TGAT/TGN/JODIE/CAW:

- harmonic time decay through recency features;
- source trajectory through the most recent destinations;
- causal motif counts from temporally ordered co-occurrences;
- reciprocal interactions as a directed temporal closure signal.

The method is deterministic and rank-preserving at output time, making it a
safe candidate for leaderboard probing and reproducible code review.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import zipfile
from collections import Counter, defaultdict
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


class TemporalMotifReranker:
    def __init__(
        self,
        reverse_weight: float,
        reverse_recency_weight: float,
        co_motif_weight: float,
        co_motif_recent_weight: float,
        source_pop_weight: float,
        local_repeat_boost: float,
    ) -> None:
        self.base = HistoryBaseline(**HEURISTIC_WEIGHTS)
        self.reverse_weight = reverse_weight
        self.reverse_recency_weight = reverse_recency_weight
        self.co_motif_weight = co_motif_weight
        self.co_motif_recent_weight = co_motif_recent_weight
        self.source_pop_weight = source_pop_weight
        self.local_repeat_boost = local_repeat_boost

        self.rows: list[tuple[int, int, int]] = []
        self.src_count: Counter[int] = Counter()
        self.src_last_time: dict[int, int] = {}
        self.co_motif: dict[int, Counter[int]] = defaultdict(Counter)
        self.max_reverse_log = 1.0
        self.max_source_log = 1.0
        self.max_co_log = 1.0

    def update(self, src: int, dst: int, time_value: int) -> None:
        self.rows.append((src, dst, time_value))
        self.base.update(src, dst, time_value)
        self.src_count[src] += 1
        self.src_last_time[src] = max(time_value, self.src_last_time.get(src, time_value))

    def finalize(self) -> None:
        self.base.finalize()
        self.max_source_log = max((math.log1p(value) for value in self.src_count.values()), default=1.0)
        self.max_reverse_log = max(
            (math.log1p(counts.get(src, 0)) for src, counts in self.base.src_dst_count.items()),
            default=1.0,
        )
        self._build_causal_motifs()

    def _build_causal_motifs(self) -> None:
        max_count = 1
        for history in self.base.src_history.values():
            history.sort()
            sequence = [dst for _, dst in history]
            for index, dst in enumerate(sequence):
                start = max(0, index - 8)
                for prev_dst in sequence[start:index]:
                    if prev_dst == dst:
                        continue
                    self.co_motif[prev_dst][dst] += 1
                    self.co_motif[dst][prev_dst] += 1
                    max_count = max(max_count, self.co_motif[prev_dst][dst], self.co_motif[dst][prev_dst])
        self.max_co_log = math.log1p(max_count)

    def score(self, src: int, dst: int, query_time: int) -> float:
        score = self.base.score(src, dst, query_time)

        reverse_count = self.base.src_dst_count.get(dst, {}).get(src, 0)
        if reverse_count:
            score += self.reverse_weight * math.log1p(reverse_count) / self.max_reverse_log
            reverse_last = self.base.src_dst_last_time.get(dst, {}).get(src)
            score += self.reverse_recency_weight * self.base._recency(query_time, reverse_last)

        src_pop = self.src_count.get(dst, 0)
        if src_pop:
            score += self.source_pop_weight * math.log1p(src_pop) / self.max_source_log

        recent_dsts = self.base.src_recent_dsts.get(src, [])
        if recent_dsts:
            motif_score = 0.0
            motif_recent_score = 0.0
            for offset, prev_dst in enumerate(reversed(recent_dsts[-12:]), start=1):
                count = self.co_motif.get(prev_dst, {}).get(dst, 0)
                if count:
                    value = math.log1p(count) / self.max_co_log
                    motif_score += value / math.sqrt(offset)
                    motif_recent_score += value / offset
            score += self.co_motif_weight * motif_score
            score += self.co_motif_recent_weight * motif_recent_score

            if dst in recent_dsts[-5:]:
                distance = len(recent_dsts) - 1 - max(
                    index for index, value in enumerate(recent_dsts) if value == dst
                )
                score += self.local_repeat_boost / (1.0 + distance)

        return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate temporal motif Track 1 submission")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", default="dataset1,dataset2")
    parser.add_argument("--reverse-weight", type=float, default=0.2)
    parser.add_argument("--reverse-recency-weight", type=float, default=0.2)
    parser.add_argument("--co-motif-weight", type=float, default=0.08)
    parser.add_argument("--co-motif-recent-weight", type=float, default=0.12)
    parser.add_argument("--source-pop-weight", type=float, default=0.05)
    parser.add_argument("--local-repeat-boost", type=float, default=0.05)
    parser.add_argument("--limit-test-rows", type=int, default=0)
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def fit_scene(data_zip: zipfile.ZipFile, scene: str, args: argparse.Namespace) -> TemporalMotifReranker:
    model = TemporalMotifReranker(
        reverse_weight=args.reverse_weight,
        reverse_recency_weight=args.reverse_recency_weight,
        co_motif_weight=args.co_motif_weight,
        co_motif_recent_weight=args.co_motif_recent_weight,
        source_pop_weight=args.source_pop_weight,
        local_repeat_boost=args.local_repeat_boost,
    )
    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            model.update(int(row["src"]), int(row["dst"]), int(row["time"]))
    model.finalize()
    return model


def write_scene(
    data_zip: zipfile.ZipFile,
    output_zip: zipfile.ZipFile,
    scene: str,
    model: TemporalMotifReranker,
    limit_test_rows: int,
) -> int:
    rows = 0
    with open_csv(data_zip, f"{scene}/test.csv") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        with output_zip.open(f"{scene}.csv", "w") as raw_output:
            with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                writer = csv.writer(text_output, lineterminator="\n")
                for row in reader:
                    src = int(row[0])
                    query_time = int(row[1])
                    candidates = [int(value) for value in row[2:]]
                    scores = [model.score(src, dst, query_time) for dst in candidates]
                    probs = rank_probabilities(scores)
                    writer.writerow([f"{value:.8f}" for value in probs])
                    rows += 1
                    if limit_test_rows and rows >= limit_test_rows:
                        break
    return rows


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    with zipfile.ZipFile(args.data_zip) as data_zip:
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
            for scene in scenes:
                print(f"[{scene}] fitting temporal motif reranker", flush=True)
                model = fit_scene(data_zip, scene, args)
                rows = write_scene(data_zip, output_zip, scene, model, args.limit_test_rows)
                print(f"[{scene}] wrote {rows} rows", flush=True)
    print(f"submission saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
