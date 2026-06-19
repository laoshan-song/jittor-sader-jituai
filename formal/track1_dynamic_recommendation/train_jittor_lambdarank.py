"""Jittor LambdaRank-style feature ranker for Track 1.

This trains on query groups built from train-only validation splits. Each query
has one positive future edge and sampled candidate negatives. The model scores
tabular temporal features and is optimized with pairwise RankNet loss.
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
from validate_heuristic import add_unique_top, load_test_candidate_pools


HEURISTIC_WEIGHTS = {
    "pair_weight": 6.0,
    "pair_recency_weight": 4.0,
    "dst_pop_weight": 0.4,
    "dst_recency_weight": 0.2,
    "sequence_weight": 2.5,
    "repeat_recent_weight": 2.0,
}


@dataclass
class EvalReport:
    strategy: str
    positives: int
    base_mrr: float
    model_mrr: float
    blend_mrr: float


class MLPScorer(nn.Module):
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
    parser = argparse.ArgumentParser(description="Train Jittor LambdaRank feature ranker")
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="dataset2")
    parser.add_argument("--train-queries", type=int, default=80000)
    parser.add_argument("--valid-queries", type=int, default=10000)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-queries", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--blend-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--strategy", choices=("test_pool", "hard", "mixed"), default="test_pool")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def open_csv(data_zip: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(data_zip.open(member, "r"), encoding="utf-8", newline="")


def read_rows(data_zip: zipfile.ZipFile, scene: str) -> list[tuple[int, int, int, str | None]]:
    rows: list[tuple[int, int, int, str | None]] = []
    with open_csv(data_zip, f"{scene}/train.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append((int(row["src"]), int(row["dst"]), int(row["time"]), row.get("split")))
    return rows


def split_rows(rows: list[tuple[int, int, int, str | None]]) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    if any(split == "1" for *_, split in rows):
        return (
            [(src, dst, time) for src, dst, time, split in rows if split == "0"],
            [(src, dst, time) for src, dst, time, split in rows if split != "0"],
        )
    ordered = sorted(rows, key=lambda value: value[2])
    cut = int(len(ordered) * 0.9)
    return (
        [(src, dst, time) for src, dst, time, _ in ordered[:cut]],
        [(src, dst, time) for src, dst, time, _ in ordered[cut:]],
    )


def fit_history(rows: list[tuple[int, int, int]]) -> HistoryBaseline:
    model = HistoryBaseline(**HEURISTIC_WEIGHTS)
    for src, dst, time_value in rows:
        model.update(src, dst, time_value)
    model.finalize()
    return model


def make_candidates(
    positive_dst: int,
    src: int,
    strategy: str,
    all_dsts: list[int],
    popular_dsts: list[int],
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    size: int,
    rng: random.Random,
) -> list[int]:
    candidates = [positive_dst]
    seen = {positive_dst}

    def add_random(pool: list[int], target: int) -> None:
        available = [dst for dst in pool if dst not in seen]
        need = target - len(candidates)
        if need <= 0 or not available:
            return
        chosen = available if len(available) <= need else rng.sample(available, need)
        seen.update(chosen)
        candidates.extend(chosen)

    if strategy == "test_pool":
        add_random(all_test_candidates, min(size, 1 + size * 4 // 5))
    elif strategy == "hard":
        add_random(src_test_candidates.get(src, []), min(size, 1 + size * 2 // 3))
        add_unique_top(candidates, seen, popular_dsts, min(size, 1 + size * 4 // 5))
        add_random(all_test_candidates, size)
    else:
        add_unique_top(candidates, seen, popular_dsts, min(size, 1 + size // 3))
        add_random(all_test_candidates, min(size, 1 + size * 2 // 3))
    add_random(all_dsts, size)
    if len(candidates) != size:
        raise RuntimeError(f"Only built {len(candidates)} candidates")
    rng.shuffle(candidates)
    return candidates


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
    base = model.score(src, dst, time_value)
    return [
        base / 20.0,
        math.log1p(pair_count),
        model._recency(time_value, pair_last),
        math.log1p(dst_count) / model.max_dst_log if dst_count else 0.0,
        model._recency(time_value, dst_last),
        1.0 if dst in recent[-1:] else 0.0,
        1.0 if dst in recent[-5:] else 0.0,
        1.0 if dst in recent[-20:] else 0.0,
        math.log1p(transition_score),
        len(recent) / 20.0,
    ]


def build_groups(
    model: HistoryBaseline,
    positives: list[tuple[int, int, int]],
    all_dsts: list[int],
    popular_dsts: list[int],
    src_test_candidates: dict[int, list[int]],
    all_test_candidates: list[int],
    args: argparse.Namespace,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(args.seed)
    groups: list[list[list[float]]] = []
    labels: list[list[float]] = []
    base_scores: list[list[float]] = []
    for src, dst, time_value in positives:
        candidates = make_candidates(
            dst,
            src,
            strategy,
            all_dsts,
            popular_dsts,
            src_test_candidates,
            all_test_candidates,
            args.candidates,
            rng,
        )
        feats = [candidate_features(model, src, candidate, time_value) for candidate in candidates]
        groups.append(feats)
        labels.append([1.0 if candidate == dst else 0.0 for candidate in candidates])
        base_scores.append([feat[0] * 20.0 for feat in feats])
    return (
        np.asarray(groups, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(base_scores, dtype=np.float32),
    )


def train_ranker(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> MLPScorer:
    jt.flags.use_cuda = 0 if args.cpu or not jt.has_cuda else 1
    model = MLPScorer(x.shape[-1], args.hidden_dim)
    opt = nn.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    steps = max(1, math.ceil(len(x) / args.batch_queries))
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x))
        total = 0.0
        for step in range(steps):
            idx = order[step * args.batch_queries:(step + 1) * args.batch_queries]
            batch_x = jt.array(x[idx].reshape((-1, x.shape[-1])))
            scores = model(batch_x).reshape((len(idx), x.shape[1]))
            labels = jt.array(y[idx])
            pos_scores = (scores * labels).sum(dim=1).reshape((-1, 1))
            loss = nn.softplus(-(pos_scores - scores)).mean()
            opt.step(loss)
            total += float(loss.item())
        print(f"epoch={epoch} loss={total / steps:.6f}", flush=True)
    return model


def reciprocal_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    positive = int(np.argmax(labels))
    rank = int(np.where(order == positive)[0][0]) + 1
    return 1.0 / rank


def evaluate_ranker(model: MLPScorer, x: np.ndarray, y: np.ndarray, base: np.ndarray, args: argparse.Namespace, strategy: str) -> EvalReport:
    scores = np.asarray(model(jt.array(x.reshape((-1, x.shape[-1])))).numpy()).reshape((len(x), x.shape[1]))
    base_rr = 0.0
    model_rr = 0.0
    blend_rr = 0.0
    for index in range(len(x)):
        base_rr += reciprocal_rank(base[index], y[index])
        model_rr += reciprocal_rank(scores[index], y[index])
        blend_rr += reciprocal_rank(base[index] + args.blend_weight * scores[index], y[index])
    n = len(x)
    return EvalReport(strategy=strategy, positives=n, base_mrr=base_rr / n, model_mrr=model_rr / n, blend_mrr=blend_rr / n)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.misc.set_global_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.data_zip) as data_zip:
        rows = read_rows(data_zip, args.scene)
        train_rows, valid_rows = split_rows(rows)
        history = fit_history(train_rows)
        all_dsts = sorted({dst for _, dst, _ in train_rows})
        popular_dsts = [dst for dst, _ in Counter(dst for _, dst, _ in train_rows).most_common(10000)]
        src_test_candidates, all_test_candidates = load_test_candidate_pools(data_zip, args.scene)
        rng = random.Random(args.seed)
        rng.shuffle(valid_rows)
        train_pos = valid_rows[:args.train_queries]
        valid_pos = valid_rows[args.train_queries:args.train_queries + args.valid_queries]
        print(f"building groups train={len(train_pos)} valid={len(valid_pos)} strategy={args.strategy}", flush=True)
        x_train, y_train, _ = build_groups(history, train_pos, all_dsts, popular_dsts, src_test_candidates, all_test_candidates, args, args.strategy)
        ranker = train_ranker(x_train, y_train, args)
        reports: list[EvalReport] = []
        for strategy in ("test_pool", "hard", "mixed"):
            x_valid, y_valid, base_valid = build_groups(history, valid_pos, all_dsts, popular_dsts, src_test_candidates, all_test_candidates, args, strategy)
            report = evaluate_ranker(ranker, x_valid, y_valid, base_valid, args, strategy)
            reports.append(report)
            print(
                f"{strategy}: base={report.base_mrr:.8f} model={report.model_mrr:.8f} blend={report.blend_mrr:.8f}",
                flush=True,
            )
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps([asdict(row) for row in reports], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
