#!/usr/bin/env python3
"""Jittor gate for D4 session-graph features against frozen hard negatives."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.update({
    "use_cutt": "0",
    "use_cutlass": "0",
    "use_nccl": "0",
    "use_mkl": "0",
})

import jittor as jt
import numpy as np
from jittor import nn

from b_rank import pairnew_transformer_jittor as pairnew, replay_score_cache

import session_graph_gate as graph_gate


GRAPH_NAMES = ("raw_sum", "raw_max", "norm_sum", "norm_max", "hit")
TRAIN_ROWS = 60_000
SELECT_ROWS = 30_000
HARD_BASE = 30
HARD_GRAPH = 20
MAX_SELECTION_NEGATIVE_RATE = 0.10
ALPHAS = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
LOSS_MODES = ("ce", "psl-tanh", "psl-relu", "psl-atan")


class HardNegativeGate(nn.Module):
    def __init__(self, feature_count: int, hidden: int = 64) -> None:
        self.candidate = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def execute(self, feature: jt.Var) -> jt.Var:
        value = self.candidate(feature)
        mean = value.mean(dim=1, keepdims=True)
        maximum = value.max(dim=1, keepdims=True)
        context = jt.concat(
            [value, value * 0.0 + mean, value * 0.0 + maximum], dim=2
        )
        return self.output(context).squeeze(-1)


def qnorm(values: np.ndarray) -> np.ndarray:
    return pairnew._qnorm(np.asarray(values, dtype=np.float32))


def feature_plane(
    graph: dict[str, np.ndarray],
    baseline: np.ndarray,
    static: np.ndarray,
    seen: np.ndarray,
) -> np.ndarray:
    rank = graph_gate.rank_positions(baseline).astype(np.float32) / 99.0
    base_qnorm = qnorm(baseline)
    base_margin = baseline.max(axis=1, keepdims=True) - baseline
    values = [graph[name][:, :, None] for name in GRAPH_NAMES]
    values.extend(
        [
            base_qnorm[:, :, None],
            rank[:, :, None],
            base_margin[:, :, None],
            seen.astype(np.float32)[:, :, None],
            np.asarray(static, dtype=np.float32),
        ]
    )
    return np.concatenate(values, axis=2).astype(np.float32, copy=False)


def prepare(
    scored: dict,
    identity_root: Path,
    baseline_root: Path,
    graph: dict[str, np.ndarray],
    split: str,
    rows: int,
    chunk: int,
    start: int = 0,
) -> dict[str, dict]:
    output = {}
    stop = start + rows
    for strategy in ("history", "test_pool"):
        _, labels, seen, _, static = scored[(strategy, split)]
        source = graph_gate.identity(identity_root, strategy, split, "src")[start:stop]
        timestamp = graph_gate.identity(identity_root, strategy, split, "time")[start:stop]
        candidates = graph_gate.identity(identity_root, strategy, split, "candidates")[start:stop]
        baseline = np.load(
            baseline_root / f"{strategy}__{split}.npy", mmap_mode="r"
        )[start:stop]
        graph_feature, active = graph_gate.graph_features(
            graph, source, timestamp, candidates, chunk
        )
        output[strategy] = {
            "feature": feature_plane(
                graph_feature, baseline, static[start:stop], seen[start:stop]
            ),
            "graph": graph_feature["norm_max"],
            "baseline": baseline,
            "labels": np.asarray(labels[start:stop], dtype=np.int32),
            "seen": np.asarray(seen[start:stop], dtype=bool),
            "active": active,
        }
        print(json.dumps({
            "prepared": [strategy, split], "start": start, "rows": rows
        }), flush=True)
    return output


def merge_blocks(blocks: list[dict[str, dict]]) -> dict[str, dict]:
    return {
        strategy: {
            name: np.concatenate([block[strategy][name] for block in blocks], axis=0)
            for name in blocks[0][strategy]
        }
        for strategy in ("history", "test_pool")
    }


def hard_mask(values: dict, rows: np.ndarray) -> np.ndarray:
    baseline = values["baseline"][rows]
    graph = values["graph"][rows]
    seen = values["seen"][rows]
    labels = values["labels"][rows]
    pair_new = ~seen
    base_order = np.argsort(-np.where(pair_new, baseline, -np.inf), axis=1)[:, :HARD_BASE]
    graph_order = np.argsort(-np.where(pair_new, graph, -np.inf), axis=1)[:, :HARD_GRAPH]
    mask = np.zeros(pair_new.shape, dtype=np.float32)
    batch = np.arange(len(rows))[:, None]
    mask[batch, base_order] = 1.0
    mask[batch, graph_order] = 1.0
    mask[np.arange(len(rows)), labels] = 1.0
    return mask * pair_new.astype(np.float32)


def predict(net: HardNegativeGate, feature: np.ndarray, batch: int) -> np.ndarray:
    output = np.empty(feature.shape[:2], dtype=np.float32)
    net.eval()
    with jt.no_grad():
        for start in range(0, len(feature), batch):
            stop = min(len(feature), start + batch)
            output[start:stop] = np.asarray(
                net(jt.array(feature[start:stop])).data, dtype=np.float32
            )
    return qnorm(output)


def psl_loss(
    score: jt.Var,
    labels: jt.Var,
    activation: str,
    score_divisor: float,
    tau_star: float,
) -> jt.Var:
    rows = jt.arange(score.shape[0])
    positive = score[rows, labels].unsqueeze(1)
    difference = (score - positive) / score_divisor
    if activation == "tanh":
        sigma = jt.tanh(difference) + 1.0
    elif activation == "relu":
        sigma = jt.maximum(difference + 1.0, 0.0)
    elif activation == "atan":
        sigma = jt.atan(difference) + 1.0
    else:
        raise ValueError(f"unsupported PSL activation: {activation}")
    log_weight = jt.log(jt.maximum(sigma, 1e-12)) / tau_star
    maximum = log_weight.max(dim=1, keepdims=True)
    loss = maximum.squeeze(1) + jt.log(
        jt.exp(log_weight - maximum).sum(dim=1)
    )
    return loss.mean()


def train_member(
    values: dict[str, dict],
    epochs: int,
    batch: int,
    seed: int,
    train_rows: int = TRAIN_ROWS,
    loss_mode: str = "ce",
    psl_score_divisor: float = 12.0,
    psl_tau_star: float = 0.2,
) -> HardNegativeGate:
    np.random.seed(seed)
    jt.set_global_seed(seed)
    rng = np.random.default_rng(seed)
    feature_count = values["history"]["feature"].shape[-1]
    net = HardNegativeGate(feature_count)
    optimizer = jt.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5)
    row_strategy = []
    row_index = []
    for strategy_index, strategy in enumerate(("history", "test_pool")):
        limit = min(int(train_rows), len(values[strategy]["labels"]))
        labels = values[strategy]["labels"][:limit]
        seen = values[strategy]["seen"][:limit]
        target_new = ~seen[np.arange(limit), labels]
        ids = np.flatnonzero(target_new)
        row_strategy.append(np.full(len(ids), strategy_index, dtype=np.int8))
        row_index.append(ids.astype(np.int32))
    strategy_ids = np.concatenate(row_strategy)
    row_ids = np.concatenate(row_index)
    names = ("history", "test_pool")
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        order = rng.permutation(len(row_ids))
        for start in range(0, len(order), batch):
            current = order[start:start + batch]
            feature_parts = []
            baseline_parts = []
            label_parts = []
            mask_parts = []
            for strategy_index, strategy in enumerate(names):
                select = current[strategy_ids[current] == strategy_index]
                if not len(select):
                    continue
                rows = row_ids[select]
                value = values[strategy]
                feature_parts.append(value["feature"][rows])
                baseline_parts.append(value["baseline"][rows])
                label_parts.append(value["labels"][rows])
                mask_parts.append(hard_mask(value, rows))
            feature = np.concatenate(feature_parts)
            baseline = np.concatenate(baseline_parts).astype(np.float32)
            labels = np.concatenate(label_parts).astype(np.int32)
            mask = np.concatenate(mask_parts).astype(np.float32)
            raw = net(jt.array(feature))
            residual = pairnew._normalize(raw)
            score = 12.0 * (jt.array(baseline) + 0.02 * residual)
            score += (1.0 - jt.array(mask)) * -1e6
            target = jt.array(labels)
            if loss_mode == "ce":
                loss = nn.cross_entropy_loss(score, target)
            else:
                loss = psl_loss(
                    score,
                    target,
                    loss_mode.removeprefix("psl-"),
                    psl_score_divisor,
                    psl_tau_star,
                )
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.data).item()))
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses))}), flush=True)
    return net


def tune(net: HardNegativeGate, values: dict[str, dict], batch: int) -> tuple[dict, list[dict]]:
    part = slice(TRAIN_ROWS, TRAIN_ROWS + SELECT_ROWS)
    prediction = {
        strategy: predict(net, current["feature"][part], batch)
        for strategy, current in values.items()
    }
    trace = []
    for alpha in ALPHAS:
        metrics = []
        for strategy, current in values.items():
            baseline = current["baseline"][part]
            candidate = pairnew._candidate_score(
                baseline, prediction[strategy], current["seen"][part], alpha
            )
            metrics.append(graph_gate.paired(
                baseline, candidate, current["labels"][part], current["active"][part]
            ))
        trace.append({
            "alpha": alpha,
            "delta": float(np.mean([value["delta"] for value in metrics])),
            "history_delta": metrics[0]["delta"],
            "test_pool_delta": metrics[1]["delta"],
            "max_negative_rate": max(value["negative_row_rate"] for value in metrics),
        })
    safe = [
        value for value in trace
        if value["max_negative_rate"] <= MAX_SELECTION_NEGATIVE_RATE
    ]
    return max(safe, key=lambda value: value["delta"]), trace


def evaluate(
    net: HardNegativeGate,
    values: dict[str, dict],
    alpha: float,
    rows: slice,
    batch: int,
) -> dict:
    output = {}
    for strategy, current in values.items():
        baseline = current["baseline"][rows]
        residual = predict(net, current["feature"][rows], batch)
        candidate = pairnew._candidate_score(
            baseline, residual, current["seen"][rows], alpha
        )
        output[strategy] = graph_gate.paired(
            baseline, candidate, current["labels"][rows], current["active"][rows]
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--replay-cache", type=Path, action="append", required=True)
    parser.add_argument("--identity-cache", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--full-fit", action="store_true")
    parser.add_argument("--fixed-alpha", type=float, default=0.015)
    parser.add_argument("--loss-mode", choices=LOSS_MODES, default="ce")
    parser.add_argument("--psl-score-divisor", type=float, default=12.0)
    parser.add_argument("--psl-tau-star", type=float, default=0.2)
    args = parser.parse_args()
    if args.psl_score_divisor <= 0.0:
        raise ValueError("--psl-score-divisor must be positive")
    if args.psl_tau_star <= 0.0:
        raise ValueError("--psl-tau-star must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    scored, _, manifests = replay_score_cache.load(args.replay_cache, verify=False)
    plan = manifests[0]["metadata"]["group_metadata"]["history"]["plan"]
    source = np.load(args.train_cache / "src.npy", mmap_mode="r")
    item = np.load(args.train_cache / "dst.npy", mmap_mode="r")
    timestamp = np.load(args.train_cache / "time.npy", mmap_mode="r")

    validation_time = graph_gate.identity(
        args.identity_cache, "history", "validation", "time"
    )
    block_rows = 30_000
    blocks = []
    block_cutoffs = []
    for start in range(0, 120_000, block_rows):
        cutoff = int(validation_time[start])
        block_cutoffs.append(cutoff)
        validation_graph = graph_gate.build_graph(source, item, timestamp, cutoff)
        blocks.append(prepare(
            scored, args.identity_cache, args.baseline_cache, validation_graph,
            "validation", block_rows, args.chunk, start=start
        ))
        del validation_graph
    validation = merge_blocks(blocks)
    del blocks
    fit_rows = 120_000 if args.full_fit else TRAIN_ROWS
    net = train_member(
        validation,
        args.epochs,
        args.batch,
        args.seed,
        fit_rows,
        args.loss_mode,
        args.psl_score_divisor,
        args.psl_tau_star,
    )
    if args.full_fit:
        selected = {
            "alpha": float(args.fixed_alpha),
            "delta": None,
            "history_delta": None,
            "test_pool_delta": None,
            "selection_contract": "frozen by prior disjoint 60k/30k selection",
        }
        trace = []
        holdout = {}
    else:
        selected, trace = tune(net, validation, args.batch)
        holdout = evaluate(
            net, validation, selected["alpha"],
            slice(TRAIN_ROWS + SELECT_ROWS, 120_000), args.batch
        )
    confirmation_graph = graph_gate.build_graph(
        source, item, timestamp, int(plan["cutoffs"]["confirm"])
    )
    confirmation_values = prepare(
        scored, args.identity_cache, args.baseline_cache, confirmation_graph,
        "confirmation", 30_000, args.chunk
    )
    confirmation = evaluate(
        net, confirmation_values, selected["alpha"], slice(None), args.batch
    )
    if args.full_fit:
        checks = {
            "both_confirmations_at_least_0_03": all(
                value["delta"] >= 0.03 for value in confirmation.values()
            ),
            "both_confirmations_above_two_se": all(
                value["delta"] > 2.0 * value["delta_se"] for value in confirmation.values()
            ),
        }
    else:
        checks = {
            "selection_at_least_0_01": selected["delta"] >= 0.01,
            "both_holdouts_at_least_0_01": all(value["delta"] >= 0.01 for value in holdout.values()),
            "both_confirmations_at_least_0_01": all(value["delta"] >= 0.01 for value in confirmation.values()),
            "both_confirmations_above_two_se": all(
                value["delta"] > 2.0 * value["delta_se"] for value in confirmation.values()
            ),
        }
    report = {
        "kind": "d4_session_graph_jittor_hard_negative_gate_v1",
        "decision": (
            "HUGE_PASS" if args.full_fit and all(checks.values())
            else "PASS" if all(checks.values()) else "NO_GO"
        ),
        "selected": selected,
        "selection_trace": trace,
        "holdout": holdout,
        "confirmation": confirmation,
        "checks": checks,
        "training": {
            "train_rows_per_replay": fit_rows,
            "full_fit": bool(args.full_fit),
            "hard_base": HARD_BASE,
            "hard_graph": HARD_GRAPH,
            "max_selection_negative_rate": MAX_SELECTION_NEGATIVE_RATE,
            "epochs": args.epochs,
            "seed": args.seed,
            "loss_mode": args.loss_mode,
            "psl_score_divisor": args.psl_score_divisor,
            "psl_tau_star": args.psl_tau_star,
            "validation_block_rows": block_rows,
            "validation_block_cutoffs": block_cutoffs,
        },
    }
    (args.run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    state = {name: np.asarray(value.data).copy() for name, value in net.state_dict().items()}
    names = list(state)
    np.savez_compressed(
        args.run_dir / "model.npz",
        kind=np.asarray("d4_session_graph_jittor_hard_negative_gate_v1"),
        state_names=np.asarray(names),
        **{f"state_{index}": state[name] for index, name in enumerate(names)},
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    # Quality gates are recorded for diagnosis; a trained model remains usable
    # by the from-scratch reproduction path even when a threshold is missed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
