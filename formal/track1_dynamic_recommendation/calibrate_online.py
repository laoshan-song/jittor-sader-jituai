"""Compare local audit metrics with known online leaderboard scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate local audit metrics against online scores")
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--online-scores", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            scores[Path(row["submission"]).name] = float(row["online_score"])
    return scores


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if var_x == 0 or var_y == 0:
        return float("nan")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for order_index in order[index:end]:
            output[order_index] = rank
        index = end
    return output


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def flatten_metrics(audit: dict[str, object], online_scores: dict[str, float]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path, payload in audit.items():
        name = Path(path).name
        if name not in online_scores:
            continue
        scenes = payload["scenes"]
        d1 = scenes["dataset1"]
        d2 = scenes["dataset2"]
        agreement = payload.get("agreement", {})

        def agree(scene: str, key: str) -> float:
            if not agreement:
                return 1.0
            return float(agreement[scene][key])

        row: dict[str, float | str] = {
            "submission": name,
            "online_score": online_scores[name],
            "d1_distinct": float(d1["avg_distinct"]),
            "d2_distinct": float(d2["avg_distinct"]),
            "avg_distinct": (float(d1["avg_distinct"]) + float(d2["avg_distinct"])) / 2.0,
            "d1_entropy": float(d1["avg_normalized_entropy"]),
            "d2_entropy": float(d2["avg_normalized_entropy"]),
            "avg_entropy": (float(d1["avg_normalized_entropy"]) + float(d2["avg_normalized_entropy"])) / 2.0,
            "d1_max_probability": float(d1["avg_max_probability"]),
            "d2_max_probability": float(d2["avg_max_probability"]),
            "avg_max_probability": (
                float(d1["avg_max_probability"]) + float(d2["avg_max_probability"])
            )
            / 2.0,
            "d1_top1_repeat_pair": float(d1["top1_repeated_pair_ratio"]),
            "d2_top1_repeat_pair": float(d2["top1_repeated_pair_ratio"]),
            "d1_top1_agreement": agree("dataset1", "top1_agreement"),
            "d2_top1_agreement": agree("dataset2", "top1_agreement"),
            "avg_top1_agreement": (
                agree("dataset1", "top1_agreement") + agree("dataset2", "top1_agreement")
            )
            / 2.0,
            "d1_top10_jaccard": agree("dataset1", "top10_jaccard"),
            "d2_top10_jaccard": agree("dataset2", "top10_jaccard"),
            "avg_top10_jaccard": (
                agree("dataset1", "top10_jaccard") + agree("dataset2", "top10_jaccard")
            )
            / 2.0,
        }
        rows.append(row)
    return rows


def correlations(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    metric_names = [key for key in rows[0] if key not in {"submission", "online_score"}]
    online = [float(row["online_score"]) for row in rows]
    output: list[dict[str, float | str]] = []
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        output.append(
            {
                "metric": metric,
                "pearson": pearson(values, online),
                "spearman": spearman(values, online),
            }
        )
    return sorted(output, key=lambda row: abs(float(row["spearman"])), reverse=True)


def markdown_report(rows: list[dict[str, float | str]], corr: list[dict[str, float | str]]) -> str:
    lines = ["# Track 1 Online Calibration", ""]
    lines.append("## Submissions")
    lines.append("")
    lines.append("| submission | online | avg_distinct | avg_entropy | avg_max_probability | d2_top10_jaccard |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda value: float(value["online_score"]), reverse=True):
        lines.append(
            f"| {row['submission']} | {float(row['online_score']):.9f} | "
            f"{float(row['avg_distinct']):.2f} | {float(row['avg_entropy']):.4f} | "
            f"{float(row['avg_max_probability']):.4f} | {float(row['d2_top10_jaccard']):.4f} |"
        )

    lines.extend(["", "## Metric Correlations", ""])
    lines.append("| metric | pearson | spearman |")
    lines.append("|---|---:|---:|")
    for row in corr:
        lines.append(f"| {row['metric']} | {float(row['pearson']):.4f} | {float(row['spearman']):.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    online_scores = load_scores(args.online_scores)
    rows = flatten_metrics(audit, online_scores)
    if len(rows) < 3:
        raise ValueError("Need at least three scored submissions for useful calibration.")
    corr = correlations(rows)
    report = {"submissions": rows, "correlations": corr}

    print(markdown_report(rows, corr))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_report(rows, corr), encoding="utf-8")


if __name__ == "__main__":
    main()
