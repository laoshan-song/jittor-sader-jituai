#!/usr/bin/env python3
"""Train a permutation-equivariant Jittor multi-resolution intensity ranker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


STRATEGIES = ("history", "test_pool")
SCALE_GRID = (0.0, 0.01, 0.02, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50)


def qnorm(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    return (centered / np.maximum(scale, 1e-6)).astype(np.float32)


def ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positive = scores[np.arange(len(scores)), labels]
    columns = np.arange(scores.shape[1])
    rank = 1 + np.sum(scores > positive[:, None], axis=1)
    rank += np.sum((scores == positive[:, None]) & (columns[None, :] < labels[:, None]), axis=1)
    return rank.astype(np.int32)


def metrics(control: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> dict:
    before = ranks(control, labels)
    after = ranks(candidate, labels)
    delta = 1.0 / after.astype(np.float64) - 1.0 / before.astype(np.float64)
    return {
        "rows": len(labels),
        "control_mrr": float(np.mean(1.0 / before)),
        "candidate_mrr": float(np.mean(1.0 / after)),
        "delta": float(delta.mean()),
        "delta_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "positive_row_rate": float(np.mean(delta > 0)),
        "negative_row_rate": float(np.mean(delta < 0)),
        "top1_changed_rate": float(np.mean(np.argmax(control, axis=1) != np.argmax(candidate, axis=1))),
    }


def paths(replay: Path, identity: Path, baseline: Path, strategy: str, split: str) -> dict:
    prefix = f"{strategy}__{split}"
    return {
        "features": None,
        "labels": replay / strategy / f"{prefix}__labels.npy",
        "seen": replay / strategy / f"{prefix}__seen.npy",
        "candidates": identity / f"{prefix}__candidates.npy",
        "baseline": baseline / f"{prefix}.npy",
    }


def duplicate_mask(candidates: np.ndarray, chunk: int = 4096) -> np.ndarray:
    output = np.empty(len(candidates), dtype=bool)
    for start in range(0, len(candidates), chunk):
        stop = min(len(candidates), start + chunk)
        output[start:stop] = np.any(
            np.diff(np.sort(candidates[start:stop], axis=1), axis=1) == 0, axis=1
        )
    return output


def hard_mask(control: np.ndarray, seen: np.ndarray, labels: np.ndarray, count: int = 16) -> np.ndarray:
    available = np.where(~seen, control, -np.inf)
    order = np.argsort(-available, axis=1, kind="stable")[:, :count]
    mask = np.zeros(control.shape, dtype=np.float32)
    rows = np.broadcast_to(np.arange(len(control))[:, None], order.shape)
    mask[rows, order] = 1.0
    mask[np.arange(len(labels)), labels] = 1.0
    return mask


def augment(values: np.ndarray, rng: np.random.Generator, probability: float) -> np.ndarray:
    if probability <= 0.0:
        return values
    output = values.copy()
    groups = ((0, 22), (0, 22, 2), (1, 22, 2), (22, values.shape[2]))
    for group in groups:
        active = rng.random(len(output)) < probability
        if not np.any(active):
            continue
        if len(group) == 2:
            output[active, :, group[0]:group[1]] = 0.0
        else:
            output[active, :, group[0]:group[1]:group[2]] = 0.0
    return output


def apply_policy(
    baseline: np.ndarray,
    residual: np.ndarray,
    seen: np.ndarray,
    duplicate: np.ndarray,
    scale: float,
    mode: str,
) -> np.ndarray:
    correction = residual.copy()
    if mode == "pair_new":
        correction[seen] = 0.0
    elif mode == "pair_seen":
        correction[~seen] = 0.0
    elif mode != "all":
        raise ValueError(mode)
    candidate = baseline + np.float32(scale) * correction
    candidate[duplicate] = baseline[duplicate]
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--replay-cache", type=Path, required=True)
    parser.add_argument("--identity-cache", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=50000)
    parser.add_argument("--selection-rows", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--predict-batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--scale-dropout", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--use-cuda", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        help="bounded API smoke only; any report produced with this option is non-submittable",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import jittor as jt
    from jittor import nn
    jt.flags.use_cuda = args.use_cuda
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    class IntensityNet(nn.Module):
        def __init__(self, feature_count: int, hidden: int):
            self.full = nn.Sequential(nn.Linear(22, hidden), nn.Relu(), nn.Linear(hidden, hidden), nn.Relu())
            self.recent = nn.Sequential(nn.Linear(feature_count - 22, hidden), nn.Relu(), nn.Linear(hidden, hidden), nn.Relu())
            self.local = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Relu())
            self.output = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Relu(), nn.Linear(hidden, 1))

        def execute(self, values):
            full = self.full(values[:, :, :22])
            recent = self.recent(values[:, :, 22:])
            local = self.local(jt.concat((full, recent), dim=2))
            context = local.mean(dim=1, keepdims=True)
            context = context.broadcast((local.shape[0], local.shape[1], local.shape[2]))
            return self.output(jt.concat((full, recent, context), dim=2)).squeeze(-1)

    feature_meta = json.loads((args.feature_cache / "metadata.json").read_text())
    feature_count = int(feature_meta["feature_count"])
    model = IntensityNet(feature_count, args.hidden)
    optimizer = jt.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    datasets = {}
    for strategy in STRATEGIES:
        for split in ("validation", "confirmation"):
            files = paths(args.replay_cache, args.identity_cache, args.baseline_cache, strategy, split)
            files["features"] = args.feature_cache / f"{strategy}__{split}.npy"
            values = {name: np.load(path, mmap_mode="r") for name, path in files.items()}
            values["duplicate"] = duplicate_mask(values["candidates"])
            datasets[(strategy, split)] = values

    fit_features = []
    fit_labels = []
    fit_baseline = []
    fit_hard = []
    for strategy in STRATEGIES:
        values = datasets[(strategy, "validation")]
        slc = slice(0, args.train_rows)
        baseline = np.asarray(values["baseline"][slc])
        labels = np.asarray(values["labels"][slc], dtype=np.int32)
        seen = np.asarray(values["seen"][slc], dtype=bool)
        base_rank = ranks(baseline, labels)
        keep = (base_rank > 1) | (rng.random(len(labels)) < 0.25)
        fit_features.append(np.asarray(values["features"][slc])[keep])
        fit_labels.append(labels[keep])
        fit_baseline.append(baseline[keep])
        fit_hard.append(hard_mask(baseline[keep], seen[keep], labels[keep]))
    fit_features = np.concatenate(fit_features)
    fit_labels = np.concatenate(fit_labels)
    fit_baseline = np.concatenate(fit_baseline)
    fit_hard = np.concatenate(fit_hard)

    def hybrid_loss(score, labels, mask):
        listwise = nn.cross_entropy_loss(score, labels)
        hard_score = score + (1.0 - mask) * -1e6
        hard = nn.cross_entropy_loss(hard_score, labels)
        rows = jt.arange(score.shape[0])
        positive = score[rows, labels].unsqueeze(1)
        soft_rank = 0.5 + jt.sigmoid((score - positive) / 0.25).sum(dim=1)
        rank_loss = jt.log(soft_rank + 1e-6).mean()
        return 0.50 * listwise + 0.30 * hard + 0.20 * rank_loss

    def predict(features: np.ndarray, view: str = "full") -> np.ndarray:
        output = np.empty(features.shape[:2], dtype=np.float32)
        model.eval()
        with jt.no_grad():
            for start in range(0, len(features), args.predict_batch):
                stop = min(len(features), start + args.predict_batch)
                block = np.asarray(features[start:stop]).copy()
                if view == "no_recent":
                    block[:, :, 22:] = 0.0
                elif view == "reverse":
                    block = block[:, ::-1]
                value = np.asarray(model(jt.array(block)).data, dtype=np.float32)
                if view == "reverse":
                    value = value[:, ::-1]
                output[start:stop] = value
        return qnorm(output)

    selection_slice = slice(args.train_rows, args.train_rows + args.selection_rows)
    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.lr = args.learning_rate * 0.5 * (1.0 + np.cos(np.pi * (epoch - 1) / max(1, args.epochs)))
        order = rng.permutation(len(fit_labels))
        losses = []
        for start in range(0, len(order), args.batch):
            index = order[start:start + args.batch]
            block = augment(fit_features[index], rng, args.scale_dropout)
            raw = model(jt.array(block))
            residual = (raw - raw.mean(dim=1, keepdims=True)) / jt.sqrt(
                ((raw - raw.mean(dim=1, keepdims=True)) ** 2).mean(dim=1, keepdims=True) + 1e-6
            )
            logits = 2.0 * jt.array(fit_baseline[index]) + residual
            loss = hybrid_loss(logits, jt.array(fit_labels[index]), jt.array(fit_hard[index]))
            loss += 1e-4 * (raw * raw).mean()
            optimizer.step(loss)
            losses.append(float(loss.item()))
        trials = []
        for strategy in STRATEGIES:
            values = datasets[(strategy, "validation")]
            residual = predict(values["features"][selection_slice])
            for mode in ("all", "pair_new", "pair_seen"):
                for scale in SCALE_GRID:
                    candidate = apply_policy(
                        np.asarray(values["baseline"][selection_slice]), residual,
                        np.asarray(values["seen"][selection_slice]), values["duplicate"][selection_slice],
                        scale, mode,
                    )
                    result = metrics(np.asarray(values["baseline"][selection_slice]), candidate,
                                     np.asarray(values["labels"][selection_slice]))
                    trials.append({"strategy": strategy, "epoch": epoch, "mode": mode, "scale": scale, **result})
        aggregate = {}
        for mode in ("all", "pair_new", "pair_seen"):
            for scale in SCALE_GRID:
                selected = [row for row in trials if row["mode"] == mode and row["scale"] == scale]
                aggregate[(mode, scale)] = float(np.mean([row["delta"] for row in selected]))
        mode, scale = max(aggregate, key=lambda key: (aggregate[key], -key[1], key[0]))
        epoch_report = {"epoch": epoch, "loss": float(np.mean(losses)), "mode": mode,
                        "scale": scale, "mean_selection_delta": aggregate[(mode, scale)], "trials": trials}
        history.append(epoch_report)
        print(json.dumps({key: value for key, value in epoch_report.items() if key != "trials"}), flush=True)
        candidate_key = (aggregate[(mode, scale)], -epoch)
        if best is None or candidate_key > best[0]:
            state = {name: np.asarray(value.data).copy() for name, value in model.state_dict().items()}
            best = (candidate_key, epoch, mode, scale, state)

    _, selected_epoch, selected_mode, selected_scale, state = best
    model.load_state_dict({name: jt.array(value) for name, value in state.items()})
    checkpoint = args.output / "model.npz"
    np.savez_compressed(checkpoint, names=np.asarray(list(state)), **{f"state_{i}": value for i, value in enumerate(state.values())})

    evaluations = {}
    holdout_stop = (
        args.train_rows + args.selection_rows + args.max_eval_rows
        if args.max_eval_rows is not None else None
    )
    confirmation_stop = args.max_eval_rows if args.max_eval_rows is not None else None
    for split, row_slice in (("holdout", slice(args.train_rows + args.selection_rows, holdout_stop)),
                             ("confirmation", slice(0, confirmation_stop))):
        source_split = "validation" if split == "holdout" else "confirmation"
        evaluations[split] = {}
        for strategy in STRATEGIES:
            values = datasets[(strategy, source_split)]
            residual_full = predict(values["features"][row_slice], "full")
            residual_no_recent = predict(values["features"][row_slice], "no_recent")
            residual_reverse = predict(values["features"][row_slice], "reverse")
            permutation_error = float(np.max(np.abs(residual_full - residual_reverse)))
            residual = qnorm(0.80 * residual_full + 0.20 * residual_no_recent)
            candidate = apply_policy(
                np.asarray(values["baseline"][row_slice]), residual,
                np.asarray(values["seen"][row_slice]), values["duplicate"][row_slice],
                selected_scale, selected_mode,
            )
            evaluations[split][strategy] = {
                **metrics(np.asarray(values["baseline"][row_slice]), candidate,
                          np.asarray(values["labels"][row_slice])),
                "permutation_tta_max_error": permutation_error,
            }
    gates = {
        "holdout_mean_delta": bool(
            np.mean([v["delta"] for v in evaluations["holdout"].values()]) >= 0.002
        ),
        "confirmation_mean_delta": bool(
            np.mean([v["delta"] for v in evaluations["confirmation"].values()]) >= 0.001
        ),
        "all_confirmation_nonnegative": bool(
            all(v["delta"] >= 0.0 for v in evaluations["confirmation"].values())
        ),
        "permutation_equivariant": bool(
            all(
                v["permutation_tta_max_error"] <= 1e-5
                for split in evaluations.values() for v in split.values()
            )
        ),
    }
    report = {
        "kind": "jittor_multiresolution_conditional_intensity_ranker_v1",
        "decision": (
            "SMOKE_ONLY" if args.max_eval_rows is not None
            else ("PASS" if all(gates.values()) else "NO_GO")
        ),
        "smoke_only": args.max_eval_rows is not None,
        "use_cuda": bool(args.use_cuda),
        "selection_confirmation_blind": True,
        "selected_epoch": selected_epoch,
        "selected_mode": selected_mode,
        "selected_scale": selected_scale,
        "training_rows": len(fit_labels),
        "feature_count": feature_count,
        "history": history,
        "evaluations": evaluations,
        "gates": gates,
        "checkpoint": str(checkpoint),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": report["decision"], "selected_epoch": selected_epoch,
                      "selected_mode": selected_mode, "selected_scale": selected_scale,
                      "evaluations": evaluations, "gates": gates}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
