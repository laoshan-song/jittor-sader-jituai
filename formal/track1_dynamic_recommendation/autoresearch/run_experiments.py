"""Small autoresearch runner for Track 1 leaderboard experiments.

The script runs a conservative parameter sweep, validates generated submission
zips, and writes a JSONL record for each candidate. It is intentionally focused
on guardrails rather than pretending the local proxy is the leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


SCENES = ("dataset1", "dataset2")


@dataclass
class ZipStats:
    rows: int
    bad_rows: int
    avg_distinct: float
    avg_max_probability: float
    min_sum: float
    max_sum: float


@dataclass
class AgreementStats:
    top1_agreement: float
    top10_jaccard: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conservative Track 1 experiments")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/track1/autoresearch"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/track1_autoresearch"))
    parser.add_argument("--records", type=Path, default=Path("outputs/track1/autoresearch/records.jsonl"))
    parser.add_argument("--baseline-zip", type=Path, default=Path("outputs/track1/result_sequence.zip"))
    parser.add_argument("--weights", default="0.5,1,1.5,2,3")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def official_row_counts(data_zip_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with zipfile.ZipFile(data_zip_path) as data_zip:
        for scene in SCENES:
            with data_zip.open(f"{scene}/test.csv") as file:
                counts[scene] = sum(1 for _ in file) - 1
    return counts


def read_orders(zip_path: Path, scene: str, top_k: int) -> list[list[int]]:
    orders: list[list[int]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"{scene}.csv") as file:
            reader = csv.reader(line.decode("utf-8") for line in file)
            for row in reader:
                values = [float(value) for value in row if value != ""]
                orders.append(sorted(range(len(values)), key=lambda index: values[index], reverse=True)[:top_k])
    return orders


def validate_zip(zip_path: Path, row_counts: dict[str, int]) -> dict[str, ZipStats]:
    stats: dict[str, ZipStats] = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for scene in SCENES:
            member = f"{scene}.csv"
            if member not in names:
                stats[scene] = ZipStats(0, row_counts[scene], 0.0, 0.0, 0.0, 0.0)
                continue

            rows = 0
            bad_rows = 0
            distinct_total = 0
            max_total = 0.0
            min_sum = float("inf")
            max_sum = float("-inf")
            with archive.open(member) as file:
                reader = csv.reader(line.decode("utf-8") for line in file)
                for row in reader:
                    rows += 1
                    values = [float(value) for value in row if value != ""]
                    row_sum = sum(values)
                    if len(values) != 100 or any(value < 0.0 or value > 1.0 for value in values):
                        bad_rows += 1
                    distinct_total += len(set(values))
                    max_total += max(values) if values else 0.0
                    min_sum = min(min_sum, row_sum)
                    max_sum = max(max_sum, row_sum)

            if rows != row_counts[scene]:
                bad_rows += abs(rows - row_counts[scene])
            stats[scene] = ZipStats(
                rows=rows,
                bad_rows=bad_rows,
                avg_distinct=distinct_total / max(1, rows),
                avg_max_probability=max_total / max(1, rows),
                min_sum=min_sum,
                max_sum=max_sum,
            )
    return stats


def compare_with_baseline(zip_path: Path, baseline_zip: Path) -> dict[str, AgreementStats]:
    agreement: dict[str, AgreementStats] = {}
    if not baseline_zip.exists():
        return agreement

    for scene in SCENES:
        candidate_top10 = read_orders(zip_path, scene, 10)
        baseline_top10 = read_orders(baseline_zip, scene, 10)
        pairs = list(zip(candidate_top10, baseline_top10))
        if not pairs:
            agreement[scene] = AgreementStats(0.0, 0.0)
            continue

        top1 = sum(candidate[0] == baseline[0] for candidate, baseline in pairs) / len(pairs)
        jaccard_total = 0.0
        for candidate, baseline in pairs:
            candidate_set = set(candidate)
            baseline_set = set(baseline)
            jaccard_total += len(candidate_set & baseline_set) / len(candidate_set | baseline_set)
        agreement[scene] = AgreementStats(top1, jaccard_total / len(pairs))
    return agreement


def command_for_weight(args: argparse.Namespace, weight: float, output_zip: Path) -> list[str]:
    return [
        args.python,
        "formal/track1_dynamic_recommendation/train_mf_rerank.py",
        "--data-zip",
        str(args.data_zip),
        "--output",
        str(output_zip),
        "--scenes",
        "dataset1,dataset2",
        "--device",
        args.device,
        "--dim",
        str(args.dim),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--reg",
        "1e-6",
        "--seed",
        str(args.seed),
        "--mf-weight",
        str(weight),
        "--negatives-per-positive",
        "2",
        "--hard-negative-ratio",
        "0",
        "--probability-mode",
        "rank",
        "--skip-validation",
    ]


def run_command(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return process.returncode


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.records.parent.mkdir(parents=True, exist_ok=True)

    row_counts = official_row_counts(args.data_zip)
    weights = [float(value) for value in args.weights.split(",") if value.strip()]

    for weight in weights:
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        safe_weight = str(weight).replace(".", "p")
        output_zip = args.output_dir / f"result_rank_mf_w{safe_weight}.zip"
        log_path = args.logs_dir / f"train_rank_mf_w{safe_weight}.log"
        command = command_for_weight(args, weight, output_zip)
        record: dict[str, object] = {
            "started_at": started_at,
            "weight": weight,
            "output_zip": str(output_zip),
            "log_path": str(log_path),
            "command": " ".join(shlex.quote(part) for part in command),
        }

        if args.dry_run:
            print(record["command"])
            continue

        returncode = run_command(command, log_path)
        record["returncode"] = returncode
        if returncode == 0:
            zip_stats = validate_zip(output_zip, row_counts)
            agreement = compare_with_baseline(output_zip, args.baseline_zip)
            record["zip_stats"] = {scene: asdict(value) for scene, value in zip_stats.items()}
            record["agreement"] = {scene: asdict(value) for scene, value in agreement.items()}

        with args.records.open("a", encoding="utf-8") as records_file:
            records_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
