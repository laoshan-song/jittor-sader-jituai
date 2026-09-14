#!/usr/bin/env python3
"""D4 multi-scale SetRank residual with metric-aligned hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

os.environ.update({
    "use_cutt": "0",
    "use_cutlass": "0",
    "use_nccl": "0",
    "use_mkl": "0",
})

import jittor as jt
import numpy as np
from jittor import nn

from . import (
    data_features,
    pairnew_transformer_jittor as v12,
    replay_score_cache,
    temporal_attention_jittor,
    verify_run,
)


DEFAULT_MEMBERS = (
    (96, 3, 4, 20260821, "hybrid"),
    (96, 3, 4, 20260822, "hybrid"),
    (128, 2, 8, 20260823, "hybrid"),
    (128, 2, 8, 20260824, "hybrid"),
    (96, 2, 8, 20260825, "ce"),
    (128, 3, 8, 20260826, "ce"),
)
TRAIN_RESIDUAL_SCALE = 1.0
HARD_NEGATIVES = 32
SOFT_RANK_TEMPERATURE = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize(values: jt.Var) -> jt.Var:
    centered = values - values.mean(dim=1, keepdims=True)
    return centered / jt.sqrt(
        (centered * centered).mean(dim=1, keepdims=True) + 1e-6
    )


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        self.attention = jt.attention.MultiheadAttention(
            hidden, heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.Relu(),
            nn.Linear(4 * hidden, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def execute(self, values: jt.Var) -> jt.Var:
        context, _ = self.attention(
            values, values, values, need_weights=False
        )
        values = self.norm1(values + context)
        return self.norm2(values + self.feedforward(values))


class FullSetRanker(nn.Module):
    def __init__(
        self, feature_count: int, hidden: int, layers: int, heads: int
    ) -> None:
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.Relu(),
            nn.Linear(hidden, hidden),
            nn.Relu(),
            nn.Linear(hidden, hidden),
            nn.Relu(),
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads) for _ in range(layers)]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Relu(),
            nn.Linear(hidden, 1),
        )

    def execute(self, values: jt.Var) -> jt.Var:
        values = self.encoder(values)
        for block in self.blocks:
            values = block(values)
        return self.output(values).squeeze(-1)


def feature_names(component_names: list[str]) -> list[str]:
    families = _component_families(component_names)
    return [
        *(f"component:{name}" for name in component_names),
        *(f"causal:{name}" for name in data_features.FEATURE_NAMES),
        "frozen_control",
        "candidate_pair_new",
        "control_rank_slot",
        "control_gap_from_top",
        "row_pair_seen_fraction",
        "row_best_new_minus_seen",
        *(f"family:{name}:mean" for name in families),
        *(f"family:{name}:std" for name in families),
    ]


def _component_families(component_names: list[str]) -> dict[str, np.ndarray]:
    prefixes = {
        "history_h32": "temporal_h32_",
        "history_h64": "temporal_h64_",
        "testpool_temporal": "temporal_testpool_",
        "fullhistory_mf": "fullhistory_mf_",
    }
    families = {
        name: np.asarray(
            [index for index, value in enumerate(component_names) if value.startswith(prefix)],
            dtype=np.int64,
        )
        for name, prefix in prefixes.items()
    }
    if any(len(indices) < 2 for indices in families.values()):
        raise ValueError("multi-scale component family is incomplete")
    return families


def _row_new_seen_gap(control: np.ndarray, seen: np.ndarray) -> np.ndarray:
    best_new = np.max(np.where(~seen, control, -np.inf), axis=1)
    best_seen = np.max(np.where(seen, control, -np.inf), axis=1)
    gap = best_new - best_seen
    gap[~np.isfinite(best_seen)] = 4.0
    gap[~np.isfinite(best_new)] = -4.0
    return np.clip(gap, -4.0, 4.0).astype(np.float32)


def build_features(
    scores: np.ndarray,
    static: np.ndarray,
    control: np.ndarray,
    seen: np.ndarray,
    static_mean: np.ndarray,
    static_std: np.ndarray,
    component_names: list[str],
) -> np.ndarray:
    base = v12._features(
        scores, static, control, seen, static_mean, static_std
    )
    top_gap = control - control.max(axis=1, keepdims=True)
    seen_fraction = np.broadcast_to(
        seen.mean(axis=1, dtype=np.float32)[:, None], control.shape
    )
    new_seen_gap = np.broadcast_to(
        _row_new_seen_gap(control, seen)[:, None], control.shape
    )
    extras = [
        v12._strict_control_slots(control),
        np.clip(top_gap, -8.0, 0.0).astype(np.float32),
        seen_fraction,
        new_seen_gap,
    ]
    families = _component_families(component_names)
    extras.extend(scores[indices].mean(axis=0) for indices in families.values())
    extras.extend(scores[indices].std(axis=0) for indices in families.values())
    return np.concatenate(
        (base, *(np.asarray(value, dtype=np.float32)[:, :, None] for value in extras)),
        axis=2,
    )


def predict(net: FullSetRanker, features: np.ndarray, batch: int) -> np.ndarray:
    output = np.empty(features.shape[:2], dtype=np.float32)
    net.eval()
    with jt.no_grad():
        for start in range(0, len(features), batch):
            block = features[start : start + batch]
            output[start : start + len(block)] = np.asarray(
                net(jt.array(block)).data, dtype=np.float32
            )
    return v12._qnorm(output)


def _hard_negative_mask(
    control: np.ndarray,
    seen: np.ndarray,
    labels: np.ndarray,
    count: int,
) -> np.ndarray:
    available = np.where(~seen, control, -np.inf)
    order = np.argsort(-available, axis=1, kind="stable")[:, :count]
    mask = np.zeros(control.shape, dtype=np.float32)
    rows = np.broadcast_to(np.arange(len(control))[:, None], order.shape)
    mask[rows, order] = 1.0
    mask[np.arange(len(labels)), labels] = 1.0
    return mask


def _hybrid_loss(
    score: jt.Var,
    labels: jt.Var,
    hard_mask: jt.Var,
) -> jt.Var:
    listwise = nn.cross_entropy_loss(score, labels)
    hard_score = score + (1.0 - hard_mask) * -1e6
    hard = nn.cross_entropy_loss(hard_score, labels)
    rows = jt.arange(score.shape[0])
    positive = score[rows, labels].unsqueeze(1)
    soft_rank = 0.5 + jt.sigmoid(
        (score - positive) / SOFT_RANK_TEMPERATURE
    ).sum(dim=1)
    rank = jt.log(soft_rank + 1e-6).mean()
    return 0.5 * listwise + 0.25 * hard + 0.25 * rank


def _quick_slot_score(
    control: np.ndarray,
    residual: np.ndarray,
    seen: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    candidates = []
    slots = v12._strict_control_slots(control)
    for alpha in np.arange(0.0, 2.0001, 0.1):
        score = v12._candidate_score(
            control, residual, seen, float(alpha), slots
        )
        candidates.append((v12._mrr(score, labels), -float(alpha), float(alpha)))
    value, _, alpha = max(candidates)
    return value, alpha


def _train_member(
    *,
    train_features: np.ndarray,
    train_control: np.ndarray,
    train_seen: np.ndarray,
    train_labels: np.ndarray,
    train_hard_mask: np.ndarray,
    select_features: np.ndarray,
    select_control: np.ndarray,
    select_seen: np.ndarray,
    select_labels: np.ndarray,
    hidden: int,
    layers: int,
    heads: int,
    seed: int,
    loss_name: str,
    epochs: int,
    batch: int,
) -> tuple[FullSetRanker, dict[str, Any]]:
    np.random.seed(seed)
    jt.set_global_seed(seed)
    rng = np.random.default_rng(seed)
    net = FullSetRanker(train_features.shape[-1], hidden, layers, heads)
    optimizer = jt.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-5)
    best = None
    records = []
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        order = rng.permutation(len(train_labels))
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            raw = net(jt.array(train_features[ids]))
            pair_new = jt.array((~train_seen[ids]).astype(np.float32, copy=False))
            score = jt.array(train_control[ids]) + TRAIN_RESIDUAL_SCALE * _normalize(raw)
            score += (1.0 - pair_new) * -1e6
            labels = jt.array(train_labels[ids])
            if loss_name == "hybrid":
                loss = _hybrid_loss(
                    score, labels, jt.array(train_hard_mask[ids])
                )
            elif loss_name == "ce":
                loss = nn.cross_entropy_loss(score, labels)
            else:
                raise ValueError(f"unsupported loss: {loss_name}")
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.data).item()))
        selection_residual = predict(net, select_features, batch)
        selection_mrr, selection_alpha = _quick_slot_score(
            select_control,
            selection_residual,
            select_seen,
            select_labels,
        )
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "selection_mrr": selection_mrr,
            "selection_slot_alpha": selection_alpha,
        }
        records.append(record)
        if best is None or (selection_mrr, -selection_alpha, -epoch) > (
            best[0], -best[1], -best[2]
        ):
            best = (
                selection_mrr,
                selection_alpha,
                epoch,
                {
                    name: np.asarray(value.data).copy()
                    for name, value in net.state_dict().items()
                },
            )
        print(
            f"fullset h={hidden} l={layers} heads={heads} seed={seed} "
            f"loss={loss_name} epoch={epoch} train={record['loss']:.6f} "
            f"selection={selection_mrr:.9f} alpha={selection_alpha:.2f}",
            flush=True,
        )
    net.load_state_dict({name: jt.array(value) for name, value in best[3].items()})
    return net, {
        "hidden": hidden,
        "layers": layers,
        "heads": heads,
        "seed": seed,
        "loss": loss_name,
        "best_epoch": best[2],
        "best_selection_mrr": best[0],
        "best_selection_slot_alpha": best[1],
        "epochs": records,
    }


def save_checkpoint(
    path: Path,
    net: FullSetRanker,
    record: dict[str, Any],
    names: list[str],
    static_mean: np.ndarray,
    static_std: np.ndarray,
) -> None:
    state = {
        name: np.asarray(value.data).copy()
        for name, value in net.state_dict().items()
    }
    state_names = list(state)
    payload = {
        f"state_{index}": state[name]
        for index, name in enumerate(state_names)
    }
    payload.update(
        state_names=np.asarray(state_names),
        feature_names=np.asarray(names),
        feature_count=np.asarray(len(names)),
        hidden=np.asarray(record["hidden"]),
        layers=np.asarray(record["layers"]),
        heads=np.asarray(record["heads"]),
        seed=np.asarray(record["seed"]),
        loss=np.asarray(record["loss"]),
        epoch=np.asarray(record["best_epoch"]),
        static_mean=np.asarray(static_mean, dtype=np.float32),
        static_std=np.asarray(static_std, dtype=np.float32),
        kind=np.asarray("d4_fullset_multiscale_lambdamrr_v15"),
    )
    np.savez(path, **payload)


def load_checkpoint(path: Path) -> tuple[FullSetRanker, dict[str, Any]]:
    saved = np.load(path, allow_pickle=False)
    if str(saved["kind"]) != "d4_fullset_multiscale_lambdamrr_v15":
        raise ValueError(f"full-set architecture mismatch: {path}")
    net = FullSetRanker(
        int(saved["feature_count"]),
        int(saved["hidden"]),
        int(saved["layers"]),
        int(saved["heads"]),
    )
    net.load_state_dict(
        {
            str(name): jt.array(saved[f"state_{index}"])
            for index, name in enumerate(saved["state_names"])
        }
    )
    return net, {
        "feature_names": [str(value) for value in saved["feature_names"]],
        "feature_count": int(saved["feature_count"]),
        "hidden": int(saved["hidden"]),
        "layers": int(saved["layers"]),
        "heads": int(saved["heads"]),
        "seed": int(saved["seed"]),
        "loss": str(saved["loss"]),
        "epoch": int(saved["epoch"]),
        "static_mean": np.asarray(saved["static_mean"], dtype=np.float32),
        "static_std": np.asarray(saved["static_std"], dtype=np.float32),
    }


def _load_v12_models(
    report_path: Path,
    component_names: list[str],
) -> tuple[list[Any], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("kind")
        != "d4_pairnew_rank_slot_candidate_set_transformer_v12"
        or report.get("decision") != "PASS"
        or report.get("component_names") != component_names
    ):
        raise ValueError("comparison v12 report differs")
    nets = []
    for record in report["training"]["members"]:
        checkpoint = Path(record["checkpoint"])
        if _sha256(checkpoint) != record["sha256"]:
            raise ValueError(f"comparison v12 checkpoint hash differs: {checkpoint}")
        net, _ = v12.load_checkpoint(checkpoint)
        nets.append(net)
    return nets, report


def _v12_residual(
    nets: list[Any],
    values: dict[str, Any],
    report: dict[str, Any],
    batch: int,
) -> np.ndarray:
    features = v12._features(
        values["scores"],
        values["static"],
        values["control"],
        values["seen"],
        np.asarray(report["training"]["static_mean"], dtype=np.float32),
        np.asarray(report["training"]["static_std"], dtype=np.float32),
    )
    return v12._qnorm(
        np.mean([v12._predict(net, features, batch) for net in nets], axis=0)
    )


def _candidate_policy(
    control: np.ndarray,
    residual: np.ndarray,
    seen: np.ndarray,
    policy: dict[str, Any],
    strict_slots: np.ndarray | None = None,
) -> np.ndarray:
    slot = v12._candidate_score(
        control,
        residual,
        seen,
        float(policy["slot_alpha"]),
        strict_slots,
    )
    escape_alpha = float(policy["escape_alpha"])
    if escape_alpha <= 0.0:
        return slot
    active = _row_new_seen_gap(control, seen) >= float(policy["escape_margin"])
    direct = control + escape_alpha * residual * (~seen).astype(np.float32)
    return np.where(active[:, None], direct, slot)


def _selection_value(
    scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> float:
    return float(np.mean([v12._mrr(scores[key], labels[key]) for key in scores]))


def _tune_policy(
    selection: dict[str, dict[str, Any]],
    old_residual: dict[str, np.ndarray],
    new_residual: dict[str, np.ndarray],
    v12_candidates: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = {key: values["labels"] for key, values in selection.items()}
    strict_slots = {
        key: v12._strict_control_slots(values["control"])
        for key, values in selection.items()
    }
    baseline = _selection_value(v12_candidates, labels)
    trace = []
    best = None
    for new_weight in np.arange(0.0, 1.0001, 0.25):
        residual = {
            key: v12._qnorm(
                (1.0 - new_weight) * old_residual[key]
                + new_weight * new_residual[key]
            )
            for key in selection
        }
        slot_best = None
        for alpha in np.arange(0.0, 2.0001, 0.05):
            scores = {
                key: v12._candidate_score(
                    values["control"],
                    residual[key],
                    values["seen"],
                    float(alpha),
                    strict_slots[key],
                )
                for key, values in selection.items()
            }
            value = _selection_value(scores, labels)
            candidate = (value, -float(alpha), float(alpha), scores)
            if slot_best is None or candidate[:2] > slot_best[:2]:
                slot_best = candidate
        value, _, slot_alpha, slot_scores = slot_best
        policy = {
            "new_residual_weight": float(new_weight),
            "slot_alpha": slot_alpha,
            "escape_alpha": 0.0,
            "escape_margin": 4.0,
            "selection_mrr": value,
            "selection_delta_vs_v12": value - baseline,
        }
        candidates = [(value, policy)]
        for margin in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            for escape_alpha in np.arange(0.1, 1.5001, 0.1):
                trial = {
                    **policy,
                    "escape_alpha": float(escape_alpha),
                    "escape_margin": float(margin),
                }
                scores = {
                    key: _candidate_policy(
                        values["control"],
                        residual[key],
                        values["seen"],
                        trial,
                        strict_slots[key],
                    )
                    for key, values in selection.items()
                }
                pair_seen_ok = True
                for key, values in selection.items():
                    target_seen = values["seen"][
                        np.arange(len(values["labels"])), values["labels"]
                    ]
                    if np.any(target_seen):
                        before = v12._mrr(
                            v12_candidates[key][target_seen],
                            values["labels"][target_seen],
                        )
                        after = v12._mrr(
                            scores[key][target_seen], values["labels"][target_seen]
                        )
                        pair_seen_ok &= after >= before - 0.001
                if pair_seen_ok:
                    value = _selection_value(scores, labels)
                    trial["selection_mrr"] = value
                    trial["selection_delta_vs_v12"] = value - baseline
                    candidates.append((value, trial))
        member_best = max(
            candidates,
            key=lambda item: (
                item[0],
                -item[1]["escape_alpha"],
                item[1]["escape_margin"],
            ),
        )
        trace.append(member_best[1])
        if best is None or member_best[0] > best[0]:
            best = member_best
    return best[1], trace


def _paired_evaluation(
    control: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    segments: dict[str, np.ndarray],
) -> dict[str, Any]:
    baseline_rr = v12._reciprocal_ranks(baseline, labels)
    candidate_rr = v12._reciprocal_ranks(candidate, labels)
    paired = candidate_rr - baseline_rr
    return {
        "control": data_features.ranking_metrics(control, labels, segments=segments),
        "v12": data_features.ranking_metrics(baseline, labels, segments=segments),
        "candidate": data_features.ranking_metrics(candidate, labels, segments=segments),
        "delta_vs_v12": float(paired.mean()),
        "delta_vs_v12_se": float(paired.std(ddof=1) / np.sqrt(len(paired))),
        "positive_rows_vs_v12": float(np.mean(paired > 0.0)),
        "negative_rows_vs_v12": float(np.mean(paired < 0.0)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing run directory reuse: {run_dir}")
    run_dir.mkdir(parents=True)
    temporal_attention_jittor.configure_cuda()
    scored, component_names, cache_manifests = replay_score_cache.load(
        args.replay_cache, verify=not args.skip_cache_hash
    )
    cache_group_seeds = {
        int(manifest.get("metadata", {}).get("group_seed", -1))
        for manifest in cache_manifests
    }
    if (
        any(
            manifest.get("metadata", {}).get("data_sha256")
            != verify_run.EXPECTED_DATA_SHA256
            for manifest in cache_manifests
        )
        or len(cache_group_seeds) != 1
        or next(iter(cache_group_seeds)) < 0
    ):
        raise ValueError("replay cache data or group seed differs")
    required = {
        (strategy, split)
        for strategy in ("history", "test_pool")
        for split in ("validation", "confirmation")
    }
    if set(scored) != required:
        raise ValueError(f"replay cache keys differ: {sorted(scored)}")
    control_report, indices, base_index, seen_alpha, new_alpha = v12._control_contract(
        args.control_fit.resolve(), component_names
    )
    prepared = {}
    for key, (scores, labels, seen, segments, static) in scored.items():
        prepared[key] = {
            "scores": scores,
            "labels": np.asarray(labels),
            "seen": np.asarray(seen),
            "segments": {name: np.asarray(values) for name, values in segments.items()},
            "static": static,
            "control": v12._control_score(
                scores,
                seen,
                control_report,
                indices,
                base_index,
                seen_alpha,
                new_alpha,
            ),
        }
    history = prepared[("history", "validation")]
    testpool = prepared[("test_pool", "validation")]
    train_rows = int(args.train_rows)
    if not 0 < train_rows < len(history["labels"]):
        raise ValueError("train rows must leave a validation holdout")
    static_mean, static_std = v12._static_normalization(
        [history["static"], testpool["static"]], train_rows
    )
    names = feature_names(component_names)
    for values in prepared.values():
        values["features"] = build_features(
            values["scores"],
            values["static"],
            values["control"],
            values["seen"],
            static_mean,
            static_std,
            component_names,
        )
        if values["features"].shape[-1] != len(names):
            raise ValueError("full-set feature schema differs")

    training_parts = []
    for values in (history, testpool):
        target_new = ~values["seen"][
            np.arange(train_rows), values["labels"][:train_rows]
        ]
        ids = np.flatnonzero(target_new)
        training_parts.append(
            (
                values["features"][ids],
                values["control"][ids],
                values["seen"][ids],
                values["labels"][ids],
            )
        )
    train_features, train_control, train_seen, train_labels = (
        np.concatenate([part[index] for part in training_parts], axis=0)
        for index in range(4)
    )
    train_hard_mask = _hard_negative_mask(
        train_control, train_seen, train_labels, HARD_NEGATIVES
    )
    holdout = slice(train_rows, None)
    selection_parts = [
        (
            values["features"][holdout],
            values["control"][holdout],
            values["seen"][holdout],
            values["labels"][holdout],
        )
        for values in (history, testpool)
    ]
    select_features, select_control, select_seen, select_labels = (
        np.concatenate([part[index] for part in selection_parts], axis=0)
        for index in range(4)
    )

    members = args.member or DEFAULT_MEMBERS
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    nets = []
    records = []
    for hidden, layers, heads, seed, loss_name in members:
        hidden, layers, heads, seed = map(int, (hidden, layers, heads, seed))
        if hidden % heads or layers < 1 or loss_name not in {"ce", "hybrid"}:
            raise ValueError("invalid full-set member")
        net, record = _train_member(
            train_features=train_features,
            train_control=train_control,
            train_seen=train_seen,
            train_labels=train_labels,
            train_hard_mask=train_hard_mask,
            select_features=select_features,
            select_control=select_control,
            select_seen=select_seen,
            select_labels=select_labels,
            hidden=hidden,
            layers=layers,
            heads=heads,
            seed=seed,
            loss_name=loss_name,
            epochs=int(args.epochs),
            batch=int(args.batch),
        )
        path = checkpoint_dir / (
            f"fullset_h{hidden}_l{layers}_a{heads}_{loss_name}_seed{seed}.npz"
        )
        save_checkpoint(path, net, record, names, static_mean, static_std)
        reloaded, metadata = load_checkpoint(path)
        before = predict(net, train_features[:128], int(args.batch))
        after = predict(reloaded, train_features[:128], int(args.batch))
        if not np.allclose(before, after, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"checkpoint reload mismatch: {path}")
        record.update(
            checkpoint=str(path),
            sha256=_sha256(path),
            checkpoint_metadata={
                key: value
                for key, value in metadata.items()
                if key not in {"static_mean", "static_std", "feature_names"}
            },
        )
        nets.append(net)
        records.append(record)

    v12_nets, v12_report = _load_v12_models(
        args.comparison_pairnew_report.resolve(), component_names
    )
    old_residual = {}
    new_residual = {}
    v12_candidates = {}
    for key, values in prepared.items():
        old_residual[key] = _v12_residual(
            v12_nets, values, v12_report, int(args.batch)
        )
        new_residual[key] = v12._qnorm(
            np.mean(
                [predict(net, values["features"], int(args.batch)) for net in nets],
                axis=0,
            )
        )
        v12_candidates[key] = v12._candidate_score(
            values["control"],
            old_residual[key],
            values["seen"],
            float(v12_report["residual_alpha"]),
        )

    selection = {
        strategy: {
            key: (
                value[holdout]
                if isinstance(value, np.ndarray) and len(value) == len(prepared[(strategy, "validation")]["labels"])
                else value
            )
            for key, value in prepared[(strategy, "validation")].items()
            if key in {"control", "seen", "labels"}
        }
        for strategy in ("history", "test_pool")
    }
    selected_old = {
        strategy: old_residual[(strategy, "validation")][holdout]
        for strategy in selection
    }
    selected_new = {
        strategy: new_residual[(strategy, "validation")][holdout]
        for strategy in selection
    }
    selected_v12 = {
        strategy: v12_candidates[(strategy, "validation")][holdout]
        for strategy in selection
    }
    policy, policy_trace = _tune_policy(
        selection, selected_old, selected_new, selected_v12
    )

    evaluations = {}
    new_weight = float(policy["new_residual_weight"])
    for key, values in prepared.items():
        strategy, split = key
        evaluation_slice = holdout if split == "validation" else slice(None)
        residual = v12._qnorm(
            (1.0 - new_weight) * old_residual[key][evaluation_slice]
            + new_weight * new_residual[key][evaluation_slice]
        )
        candidate = _candidate_policy(
            values["control"][evaluation_slice],
            residual,
            values["seen"][evaluation_slice],
            policy,
        )
        segments = {
            name: mask[evaluation_slice] for name, mask in values["segments"].items()
        }
        output_split = "holdout" if split == "validation" else split
        evaluations.setdefault(strategy, {})[output_split] = _paired_evaluation(
            values["control"][evaluation_slice],
            v12_candidates[key][evaluation_slice],
            candidate,
            values["labels"][evaluation_slice],
            segments,
        )

    checks = {
        "selection_improves_v12": policy["selection_delta_vs_v12"] >= 0.001,
        "history_confirmation_improves_v12": evaluations["history"]["confirmation"]["delta_vs_v12"] >= 0.001,
        "testpool_confirmation_improves_v12": evaluations["test_pool"]["confirmation"]["delta_vs_v12"] >= 0.001,
        "history_confirmation_above_one_se": evaluations["history"]["confirmation"]["delta_vs_v12"] >= evaluations["history"]["confirmation"]["delta_vs_v12_se"],
        "testpool_confirmation_above_one_se": evaluations["test_pool"]["confirmation"]["delta_vs_v12"] >= evaluations["test_pool"]["confirmation"]["delta_vs_v12_se"],
    }
    for strategy in ("history", "test_pool"):
        report = evaluations[strategy]["confirmation"]
        before = report["v12"]["segments"]["pair_seen"]["mrr"]
        after = report["candidate"]["segments"]["pair_seen"]["mrr"]
        checks[f"{strategy}_pair_seen_guard"] = after >= before - 0.001

    report = {
        "kind": "d4_fullset_multiscale_lambdamrr_candidate_v15",
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "data_sha256": verify_run.EXPECTED_DATA_SHA256,
        "component_names": component_names,
        "feature_names": names,
        "feature_count": len(names),
        "architecture": {
            "set_ranker": "permutation-equivariant full-candidate self-attention",
            "encoder_layers": 3,
            "feedforward_expansion": 4,
            "multi_scale_families": list(_component_families(component_names)),
            "hard_negatives": HARD_NEGATIVES,
            "soft_rank_temperature": SOFT_RANK_TEMPERATURE,
            "training_losses": sorted({record["loss"] for record in records}),
            "candidate_policy": "v12 rank slots plus confidence-gated pair-new escape",
        },
        "selection_replays": [
            f"history validation rows {train_rows}:30000",
            f"test_pool validation rows {train_rows}:30000",
        ],
        "training_replays": [
            f"history validation rows 0:{train_rows}",
            f"test_pool validation rows 0:{train_rows}",
        ],
        "confirmation_excluded_from_selection": True,
        "control_fit": {
            "path": str(args.control_fit.resolve()),
            "sha256": _sha256(args.control_fit.resolve()),
        },
        "comparison_v12": {
            "path": str(args.comparison_pairnew_report.resolve()),
            "sha256": _sha256(args.comparison_pairnew_report.resolve()),
            "residual_alpha": float(v12_report["residual_alpha"]),
        },
        "cache_manifests": [
            {
                "kind": manifest["kind"],
                "entries": sorted(manifest["entries"]),
            }
            for manifest in cache_manifests
        ],
        "replay_group_seed": next(iter(cache_group_seeds)),
        "training": {
            "pair_new_rows": int(len(train_labels)),
            "requested_rows_per_replay": train_rows,
            "epochs": int(args.epochs),
            "batch": int(args.batch),
            "members": records,
            "static_mean": static_mean.tolist(),
            "static_std": static_std.tolist(),
        },
        "policy": policy,
        "policy_trace": policy_trace,
        "metrics": evaluations,
        "checks": checks,
        "source_hashes": {
            Path(__file__).name: _sha256(Path(__file__).resolve()),
            Path(replay_score_cache.__file__).name: _sha256(
                Path(replay_score_cache.__file__).resolve()
            ),
        },
    }
    _atomic_json(run_dir / "research_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-cache", type=Path, action="append", required=True)
    parser.add_argument("--control-fit", type=Path, required=True)
    parser.add_argument("--comparison-pairnew-report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--member",
        nargs=5,
        action="append",
        metavar=("HIDDEN", "LAYERS", "HEADS", "SEED", "LOSS"),
    )
    parser.add_argument("--train-rows", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--skip-cache-hash", action="store_true")
    return parser


def main() -> int:
    try:
        print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "kind": "d4_fullset_multiscale_lambdamrr_candidate_v15",
                    "decision": "ERROR",
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
