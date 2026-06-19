"""Jittor student for high-confidence top1 promotion.

The teacher is a known-good gated submission. The student learns which anchor
top-k candidate should be promoted using only reproducible history features and
anchor ranks at inference time. This is a distillation bridge from strong
tabular exploration toward an auditable Jittor path.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import jittor as jt
from jittor import nn
import numpy as np

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


@dataclass
class Report:
    rows: int
    positives: int
    valid_auc_proxy: float
    teacher_changed: int
    student_changed: int
    changed_overlap: int
    top1_agreement_with_teacher: float
    top1_agreement_with_anchor: float


class GateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Relu(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Relu(),
            nn.Linear(hidden_dim, 1),
        )

    def execute(self, x: jt.Var) -> jt.Var:
        return self.net(x).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Jittor top1 gate student")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="dataset2")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-promote-frac", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--report-json", type=Path, default=Path("/tmp/jittor_gate_student_report.json"))
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def read_train_rows(data_zip: zipfile.ZipFile, scene: str) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append((int(row["src"]), int(row["dst"]), int(row["time"])))
    return rows


def read_test_rows(data_zip: zipfile.ZipFile, scene: str) -> list[tuple[int, int, list[int]]]:
    rows: list[tuple[int, int, list[int]]] = []
    with open_csv(data_zip, f"{scene}/test.csv") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            rows.append((int(row[0]), int(row[1]), [int(value) for value in row[2:]]))
    return rows


def read_submission_rows(zip_path: Path, scene: str) -> list[list[float]]:
    rows: list[list[float]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(f"{scene}.csv") as file:
            reader = csv.reader(line.decode("utf-8") for line in file)
            for row in reader:
                rows.append([float(value) for value in row if value != ""])
    return rows


def order_and_rank(values: list[float]) -> tuple[list[int], list[int]]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    rank = [0] * len(values)
    for rank_value, index in enumerate(order, start=1):
        rank[index] = rank_value
    return order, rank


def fit_history(rows: list[tuple[int, int, int]]) -> HistoryBaseline:
    model = HistoryBaseline(**HEURISTIC_WEIGHTS)
    for src, dst, time_value in rows:
        model.update(src, dst, time_value)
    model.finalize()
    return model


def candidate_features(model: HistoryBaseline, src: int, dst: int, time_value: int) -> list[float]:
    src_counts = model.src_dst_count.get(src, {})
    pair_count = src_counts.get(dst, 0)
    pair_last = model.src_dst_last_time.get(src, {}).get(dst)
    dst_count = model.dst_count.get(dst, 0)
    dst_last = model.dst_last_time.get(dst)
    recent = model.src_recent_dsts.get(src, [])
    transition_score = 0.0
    for offset, prev_dst in enumerate(reversed(recent[-10:]), start=1):
        transition_score += model.transition_count.get(prev_dst, {}).get(dst, 0) / offset
    return [
        model.score(src, dst, time_value) / 20.0,
        math.log1p(pair_count),
        model._recency(time_value, pair_last),
        math.log1p(dst_count) / model.max_dst_log if model.max_dst_log else 0.0,
        model._recency(time_value, dst_last),
        1.0 if dst in recent[-1:] else 0.0,
        1.0 if dst in recent[-5:] else 0.0,
        1.0 if dst in recent[-20:] else 0.0,
        math.log1p(transition_score),
        len(recent) / 20.0,
        math.log1p(len(model.src_history.get(src, []))),
        math.log1p(len(src_counts)),
    ]


def pair_features(
    model: HistoryBaseline,
    src: int,
    time_value: int,
    candidate: int,
    anchor_top: int,
    candidate_rank: int,
    candidate_prob: float,
    anchor_top_prob: float,
) -> list[float]:
    cand = candidate_features(model, src, candidate, time_value)
    top = candidate_features(model, src, anchor_top, time_value)
    diff = [a - b for a, b in zip(cand, top)]
    return cand + top + diff + [
        1.0 / candidate_rank,
        candidate_prob,
        anchor_top_prob - candidate_prob,
        1.0 if candidate != anchor_top else 0.0,
    ]


def build_dataset(
    model: HistoryBaseline,
    test_rows: list[tuple[int, int, list[int]]],
    anchor_rows: list[list[float]],
    teacher_rows: list[list[float]],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    xs: list[list[float]] = []
    ys: list[float] = []
    row_ids: list[int] = []
    candidate_keys: list[tuple[int, int]] = []
    for row_index, ((src, time_value, candidates), anchor_values, teacher_values) in enumerate(
        zip(test_rows, anchor_rows, teacher_rows)
    ):
        anchor_order, _ = order_and_rank(anchor_values)
        teacher_order, _ = order_and_rank(teacher_values)
        anchor_top = anchor_order[0]
        teacher_top = teacher_order[0]
        for rank_index, candidate_index in enumerate(anchor_order[:top_k], start=1):
            xs.append(
                pair_features(
                    model,
                    src,
                    time_value,
                    candidates[candidate_index],
                    candidates[anchor_top],
                    rank_index,
                    anchor_values[candidate_index],
                    anchor_values[anchor_top],
                )
            )
            ys.append(1.0 if teacher_top != anchor_top and candidate_index == teacher_top else 0.0)
            row_ids.append(row_index)
            candidate_keys.append((row_index, candidate_index))
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(row_ids, dtype=np.int32),
        candidate_keys,
    )


def train_model(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> GateMLP:
    jt.flags.use_cuda = 0 if args.cpu or not jt.has_cuda else 1
    model = GateMLP(x.shape[1], args.hidden_dim)
    opt = nn.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    positives = max(1.0, float(y.sum()))
    negatives = max(1.0, float(len(y) - y.sum()))
    pos_weight = min(50.0, negatives / positives)
    steps = max(1, math.ceil(len(x) / args.batch_size))
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x))
        total = 0.0
        for step in range(steps):
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            batch_x = jt.array(x[idx])
            batch_y = jt.array(y[idx])
            logits = model(batch_x)
            weights = batch_y * pos_weight + (1.0 - batch_y)
            loss = (nn.binary_cross_entropy_with_logits(logits, batch_y, reduction="none") * weights).mean()
            opt.step(loss)
            total += float(loss.item())
        print(f"epoch={epoch} loss={total / steps:.6f} pos_weight={pos_weight:.3f}", flush=True)
    return model


def predict(model: GateMLP, x: np.ndarray, batch_size: int) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        logits = model(jt.array(x[start:start + batch_size]))
        scores.append(np.asarray(logits.numpy(), dtype=np.float32))
    return np.concatenate(scores)


def auc_proxy(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    rng = np.random.default_rng(2026)
    neg_sample = neg if len(neg) <= 200000 else rng.choice(neg, 200000, replace=False)
    return float((pos[:, None] > neg_sample[None, :]).mean())


def write_submission(
    output: Path,
    anchor: Path,
    scene: str,
    anchor_rows: list[list[float]],
    candidate_keys: list[tuple[int, int]],
    scores: np.ndarray,
    max_promote_frac: float,
) -> tuple[int, list[int]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    best_by_row: dict[int, tuple[float, int]] = {}
    for (row_index, candidate_index), score in zip(candidate_keys, scores):
        anchor_order, _ = order_and_rank(anchor_rows[row_index])
        if candidate_index == anchor_order[0]:
            continue
        current = best_by_row.get(row_index)
        if current is None or score > current[0]:
            best_by_row[row_index] = (float(score), candidate_index)

    limit = int(len(anchor_rows) * max_promote_frac)
    selected = sorted(best_by_row.items(), key=lambda item: item[1][0], reverse=True)[:limit]
    promote_by_row = {row_index: candidate_index for row_index, (_, candidate_index) in selected}

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        with zipfile.ZipFile(anchor) as anchor_zip:
            for current_scene in ("dataset1", "dataset2"):
                if current_scene != scene:
                    output_zip.writestr(f"{current_scene}.csv", anchor_zip.read(f"{current_scene}.csv"))
                    continue
                with output_zip.open(f"{scene}.csv", "w") as raw_output:
                    with io.TextIOWrapper(raw_output, encoding="utf-8", newline="") as text_output:
                        writer = csv.writer(text_output, lineterminator="\n")
                        for row_index, anchor_values in enumerate(anchor_rows):
                            promote_index = promote_by_row.get(row_index)
                            if promote_index is None:
                                scores_row = anchor_values
                            else:
                                anchor_order, _ = order_and_rank(anchor_values)
                                final_order = [promote_index] + [index for index in anchor_order if index != promote_index]
                                scores_row = [0.0] * len(anchor_values)
                                for rank, index in enumerate(final_order, start=1):
                                    scores_row[index] = len(anchor_values) + 1 - rank
                            writer.writerow([f"{value:.8f}" for value in rank_probabilities(scores_row)])
    return len(promote_by_row), sorted(promote_by_row)


def top1_changed_rows(rows_a: list[list[float]], rows_b: list[list[float]]) -> set[int]:
    changed: set[int] = set()
    for index, (a_values, b_values) in enumerate(zip(rows_a, rows_b)):
        if order_and_rank(a_values)[0][0] != order_and_rank(b_values)[0][0]:
            changed.add(index)
    return changed


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.misc.set_global_seed(args.seed)

    with zipfile.ZipFile(args.data_zip) as data_zip:
        print("loading data", flush=True)
        history = fit_history(read_train_rows(data_zip, args.scene))
        test_rows = read_test_rows(data_zip, args.scene)

    anchor_rows = read_submission_rows(args.anchor, args.scene)
    teacher_rows = read_submission_rows(args.teacher, args.scene)
    x, y, row_ids, candidate_keys = build_dataset(history, test_rows, anchor_rows, teacher_rows, args.top_k)
    print(f"student data rows={len(test_rows)} examples={len(x)} positives={int(y.sum())}", flush=True)

    rng = np.random.default_rng(args.seed)
    row_order = rng.permutation(len(test_rows))
    valid_rows = set(row_order[: int(len(test_rows) * args.valid_frac)].tolist())
    train_mask = np.asarray([row_id not in valid_rows for row_id in row_ids])
    valid_mask = ~train_mask
    model = train_model(x[train_mask], y[train_mask], args)
    valid_scores = predict(model, x[valid_mask], args.batch_size)
    full_scores = predict(model, x, args.batch_size)

    changed_count, student_changed_rows = write_submission(
        args.output,
        args.anchor,
        args.scene,
        anchor_rows,
        candidate_keys,
        full_scores,
        args.max_promote_frac,
    )
    teacher_changed = top1_changed_rows(anchor_rows, teacher_rows)
    student_changed = set(student_changed_rows)
    agreement = sum(
        order_and_rank(read_submission_rows(args.output, args.scene)[idx])[0][0] == order_and_rank(teacher_rows[idx])[0][0]
        for idx in range(len(anchor_rows))
    ) / len(anchor_rows)
    anchor_agreement = 1.0 - changed_count / len(anchor_rows)
    report = Report(
        rows=len(test_rows),
        positives=int(y.sum()),
        valid_auc_proxy=auc_proxy(valid_scores, y[valid_mask]),
        teacher_changed=len(teacher_changed),
        student_changed=changed_count,
        changed_overlap=len(teacher_changed & student_changed),
        top1_agreement_with_teacher=agreement,
        top1_agreement_with_anchor=anchor_agreement,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2), flush=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
