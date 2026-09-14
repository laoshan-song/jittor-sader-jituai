#!/usr/bin/env python3
"""D4 pair-new rank-slot Transformer anchored to a frozen control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np
from jittor import nn

from . import data_features


DEFAULT_MEMBERS = ((64, 20260813), (96, 20260814))
LAYERS = 2
HEADS = 4
RESIDUAL_SCALE = 1.0
EXPECTED_CONTROL_SHA256 = (
    "86c38a86fafcadaa43deb3b196ad5e60f511ce779a1e6944a95c5d7a02643c4e"
)
EXPECTED_V12_REPORT_SHA256 = (
    "7d566b9793e054a1351653d5bfccd708379069e5b4b87e15f448ddce0fc1ebf5"
)
EXPECTED_DATA_SHA256 = (
    "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
)
V12_TRAIN_ROWS = 20000
V12_REPLAY_ROWS = 30000
V12_BASE_ALPHA = 1.05


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        self.attention = jt.attention.MultiheadAttention(
            hidden, HEADS, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.Relu(),
            nn.Linear(2 * hidden, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def execute(self, values: jt.Var) -> jt.Var:
        context, _ = self.attention(
            values, values, values, need_weights=False
        )
        values = self.norm1(values + context)
        return self.norm2(values + self.feedforward(values))


class PairNewTransformer(nn.Module):
    def __init__(self, feature_count: int, hidden: int) -> None:
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.Relu(),
            nn.Linear(hidden, hidden),
            nn.Relu(),
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden) for _ in range(LAYERS)]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Relu(), nn.Linear(hidden, 1)
        )

    def execute(self, values: jt.Var) -> jt.Var:
        values = self.encoder(values)
        for block in self.blocks:
            values = block(values)
        return self.output(values).squeeze(-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _qnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / (values.std(axis=1, keepdims=True) + 1e-6)


def _normalize(values: jt.Var) -> jt.Var:
    centered = values - values.mean(dim=1, keepdims=True)
    return centered / jt.sqrt(
        (centered * centered).mean(dim=1, keepdims=True) + 1e-6
    )


def _reciprocal_ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positive = scores[np.arange(len(labels)), labels]
    columns = np.arange(scores.shape[1])[None, :]
    rank = 1 + (scores > positive[:, None]).sum(axis=1)
    rank += ((scores == positive[:, None]) & (columns < labels[:, None])).sum(
        axis=1
    )
    return 1.0 / rank


def _mrr(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(_reciprocal_ranks(scores, labels).mean())


def _control_contract(
    path: Path, component_names: list[str]
) -> tuple[dict[str, Any], np.ndarray, int, float, float]:
    if _sha256(path) != EXPECTED_CONTROL_SHA256:
        raise ValueError("control fit is not the online-1.149 frozen fit report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("kind") not in {"d4_multimodel_fit_v1", "d4_poolset_multimodel_fit_v1"}
        or report.get("decision") != "PASS"
        or report.get("selection_replay") != "history validation only"
        or report.get("confirmation_excluded_from_selection") is not True
        or report.get("test_pool_is_diagnostic_only") is not True
    ):
        raise ValueError("control fit report is not the frozen causal PASS contract")
    control_names = list(report["component_names"])
    if not set(control_names) <= set(component_names):
        raise ValueError("control components are absent from candidate scores")
    weights = np.asarray(
        [float(report["weights"][name]) for name in control_names],
        dtype=np.float64,
    )
    if (
        not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or abs(float(weights.sum()) - 1.0) > 1e-8
    ):
        raise ValueError("control weights are invalid")
    seen_alpha = float(report["seen_alpha"])
    new_alpha = float(report["new_alpha"])
    if not (0.0 <= seen_alpha <= 1.0 and 0.0 <= new_alpha <= 1.0):
        raise ValueError("control gate is invalid")
    indices = np.asarray(
        [component_names.index(name) for name in control_names], dtype=np.int64
    )
    base_index = component_names.index(str(report["best_component"]))
    return report, indices, base_index, seen_alpha, new_alpha


def _control_score(
    scores: np.ndarray,
    candidate_seen: np.ndarray,
    control_report: dict[str, Any],
    indices: np.ndarray,
    base_index: int,
    seen_alpha: float,
    new_alpha: float,
) -> np.ndarray:
    names = list(control_report["component_names"])
    weights = np.asarray(
        [float(control_report["weights"][name]) for name in names],
        dtype=np.float64,
    )
    mixed = np.tensordot(weights, scores[indices], axes=(0, 0)).astype(
        np.float32, copy=False
    )
    base = scores[base_index]
    alpha = np.where(candidate_seen, seen_alpha, new_alpha).astype(np.float32)
    return base + alpha * (mixed - base)


def _static_normalization(
    static_parts: list[np.ndarray], rows: int
) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = np.zeros(static_parts[0].shape[-1], dtype=np.float64)
    squared = np.zeros_like(total)
    for values in static_parts:
        block = np.asarray(values[:rows], dtype=np.float64)
        total += block.sum(axis=(0, 1))
        squared += np.square(block).sum(axis=(0, 1))
        count += block.shape[0] * block.shape[1]
    mean = total / count
    variance = np.maximum(squared / count - mean * mean, 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _features(
    scores: np.ndarray,
    static: np.ndarray,
    control: np.ndarray,
    candidate_seen: np.ndarray,
    static_mean: np.ndarray,
    static_std: np.ndarray,
) -> np.ndarray:
    normalized_static = np.clip(
        (static - static_mean[None, None, :]) / static_std[None, None, :],
        -8.0,
        8.0,
    ).astype(np.float32, copy=False)
    return np.concatenate(
        (
            np.moveaxis(scores, 0, 2),
            normalized_static,
            control[:, :, None],
            (~candidate_seen).astype(np.float32)[:, :, None],
        ),
        axis=2,
    )


def _predict(net: PairNewTransformer, feature: np.ndarray, batch: int) -> np.ndarray:
    output = np.empty(feature.shape[:2], dtype=np.float32)
    net.eval()
    with jt.no_grad():
        for start in range(0, len(feature), batch):
            block = feature[start : start + batch]
            output[start : start + len(block)] = np.asarray(
                net(jt.array(block)).data, dtype=np.float32
            )
    return _qnorm(output)


def _strict_control_slots(control: np.ndarray) -> np.ndarray:
    """Encode the exact score/column tie-break order as unique row-wise slots."""
    control = np.asarray(control, dtype=np.float32)
    order = np.argsort(-control, axis=1, kind="stable")
    rows = np.broadcast_to(np.arange(len(control))[:, None], order.shape)
    values = np.linspace(1.0, 0.0, control.shape[1], dtype=np.float32)
    output = np.empty_like(control)
    output[rows, order] = values[None, :]
    return output


def _candidate_score(
    control: np.ndarray,
    residual: np.ndarray,
    candidate_seen: np.ndarray,
    alpha: float,
    strict_slots: np.ndarray | None = None,
) -> np.ndarray:
    """Reorder only pair-new rank slots, preserving every seen-candidate rank."""
    control = np.asarray(control, dtype=np.float32)
    key = control + float(alpha) * np.asarray(residual, dtype=np.float32)
    pair_new = ~np.asarray(candidate_seen, dtype=bool)
    strict_slots = (
        _strict_control_slots(control)
        if strict_slots is None
        else np.asarray(strict_slots, dtype=np.float32)
    )
    order = np.argsort(np.where(pair_new, -key, np.inf), axis=1, kind="stable")
    slots = np.sort(np.where(pair_new, strict_slots, -np.inf), axis=1)[:, ::-1]
    counts = pair_new.sum(axis=1)
    active = np.arange(control.shape[1])[None, :] < counts[:, None]
    rows = np.broadcast_to(np.arange(len(control))[:, None], order.shape)
    output = strict_slots.copy()
    output[rows[active], order[active]] = slots[active]
    return output


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    """Return exact zero-based ranks using the stable column tie-break."""
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores, axis=1, kind="stable")
    rows = np.broadcast_to(np.arange(len(scores))[:, None], order.shape)
    ranks = np.empty(order.shape, dtype=np.int64)
    ranks[rows, order] = np.arange(scores.shape[1], dtype=np.int64)[None, :]
    return ranks


def _tune_gate(
    control: np.ndarray,
    residual: np.ndarray,
    candidate_seen: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float]:
    target_seen = candidate_seen[np.arange(len(labels)), labels]
    candidates = []
    strict_slots = _strict_control_slots(control)
    for alpha in np.arange(0.0, 2.0001, 0.025):
        score = _candidate_score(
            control, residual, candidate_seen, float(alpha), strict_slots
        )
        seen_mrr = (
            _mrr(score[target_seen], labels[target_seen])
            if np.any(target_seen)
            else float("nan")
        )
        candidates.append(
            (_mrr(score, labels), -float(alpha), float(alpha), seen_mrr)
        )
    if not candidates:
        raise RuntimeError("no residual weight candidate was evaluated")
    value, _, alpha, seen_mrr = max(candidates)
    return alpha, value, seen_mrr


def _save_checkpoint(
    path: Path,
    net: PairNewTransformer,
    *,
    feature_count: int,
    hidden: int,
    seed: int,
    epoch: int,
    static_mean: np.ndarray,
    static_std: np.ndarray,
) -> None:
    state = {
        name: np.asarray(value.data).copy()
        for name, value in net.state_dict().items()
    }
    names = list(state)
    payload = {f"state_{index}": state[name] for index, name in enumerate(names)}
    payload.update(
        state_names=np.asarray(names),
        feature_count=np.asarray(feature_count),
        hidden=np.asarray(hidden),
        layers=np.asarray(LAYERS),
        heads=np.asarray(HEADS),
        seed=np.asarray(seed),
        epoch=np.asarray(epoch),
        static_mean=np.asarray(static_mean, dtype=np.float32),
        static_std=np.asarray(static_std, dtype=np.float32),
        kind=np.asarray("d4_pairnew_rank_slot_transformer_v12"),
    )
    np.savez(path, **payload)


def load_checkpoint(path: Path) -> tuple[PairNewTransformer, dict[str, Any]]:
    saved = np.load(path, allow_pickle=False)
    if (
        str(saved["kind"]) != "d4_pairnew_rank_slot_transformer_v12"
        or int(saved["layers"]) != LAYERS
        or int(saved["heads"]) != HEADS
    ):
        raise ValueError(f"residual architecture mismatch: {path}")
    net = PairNewTransformer(int(saved["feature_count"]), int(saved["hidden"]))
    net.load_state_dict(
        {
            str(name): jt.array(saved[f"state_{index}"])
            for index, name in enumerate(saved["state_names"])
        }
    )
    metadata = {
        "feature_count": int(saved["feature_count"]),
        "hidden": int(saved["hidden"]),
        "seed": int(saved["seed"]),
        "epoch": int(saved["epoch"]),
        "static_mean": np.asarray(saved["static_mean"], dtype=np.float32),
        "static_std": np.asarray(saved["static_std"], dtype=np.float32),
    }
    return net, metadata


def _train_member(
    *,
    train_feature: np.ndarray,
    train_control: np.ndarray,
    train_seen: np.ndarray,
    train_labels: np.ndarray,
    select_feature: np.ndarray,
    select_control: np.ndarray,
    select_seen: np.ndarray,
    select_labels: np.ndarray,
    hidden: int,
    seed: int,
    epochs: int,
    batch: int,
) -> tuple[PairNewTransformer, np.ndarray, dict[str, Any]]:
    np.random.seed(seed)
    jt.set_global_seed(seed)
    rng = np.random.default_rng(seed)
    net = PairNewTransformer(train_feature.shape[-1], hidden)
    optimizer = jt.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best = None
    epoch_records = []
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        order = rng.permutation(len(train_labels))
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            raw = net(jt.array(train_feature[ids]))
            pair_new = jt.array((~train_seen[ids]).astype(np.float32, copy=False))
            score = jt.array(train_control[ids]) + RESIDUAL_SCALE * _normalize(raw)
            score += (1.0 - pair_new) * -1e6
            loss = nn.cross_entropy_loss(score, jt.array(train_labels[ids]))
            optimizer.step(loss)
            losses.append(float(np.asarray(loss.data).item()))
        prediction = _predict(net, select_feature, batch)
        alpha, value, pair_seen_mrr = _tune_gate(
            select_control, prediction, select_seen, select_labels
        )
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "selection_alpha": alpha,
            "selection_margin": 0.0,
            "selection_mrr": value,
            "selection_pair_seen_mrr": pair_seen_mrr,
        }
        epoch_records.append(record)
        if best is None or (value, -alpha, -epoch) > (
            best[0], -best[1], -best[2]
        ):
            best = (
                value,
                alpha,
                epoch,
                {name: np.asarray(value.data).copy() for name, value in net.state_dict().items()},
            )
        print(
            f"pairnew set{hidden} seed={seed} epoch={epoch} "
            f"loss={record['loss']:.6f} selection_mrr={value:.9f} "
            f"alpha={alpha:.3f} pair_new_rank_slots",
            flush=True,
        )
    net.load_state_dict({name: jt.array(value) for name, value in best[3].items()})
    prediction = _predict(net, select_feature, batch)
    return net, prediction, {
        "hidden": hidden,
        "seed": seed,
        "best_epoch": best[2],
        "best_selection_mrr": best[0],
        "best_selection_alpha": best[1],
        "best_selection_margin": 0.0,
        "epochs": epoch_records,
    }


def _evaluation(
    control: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    segments: dict[str, np.ndarray],
) -> dict[str, Any]:
    before = _reciprocal_ranks(control, labels)
    after = _reciprocal_ranks(candidate, labels)
    paired = after - before
    return {
        "control": data_features.ranking_metrics(control, labels, segments=segments),
        "candidate": data_features.ranking_metrics(candidate, labels, segments=segments),
        "delta": float(paired.mean()),
        "delta_se": float(paired.std(ddof=1) / np.sqrt(len(paired))),
        "positive_rows": float(np.mean(paired > 0.0)),
        "negative_rows": float(np.mean(paired < 0.0)),
    }


def run(
    *,
    scored: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]],
    component_names: list[str],
    control_fit_path: Path,
    run_dir: Path,
    members: list[tuple[int, int]],
    train_rows: int,
    epochs: int,
    batch: int,
    baseline_report_path: Path | None = None,
) -> dict[str, Any]:
    if (
        len(members) < 2
        or {hidden for hidden, _ in members} < {64, 96}
        or len({seed for _, seed in members}) != len(members)
        or any(hidden < HEADS or hidden % HEADS for hidden, _ in members)
        or epochs < 1
        or batch < 1
    ):
        raise ValueError("v12 audit requires independent hidden-64 and hidden-96 members")
    control_report, indices, base_index, seen_alpha, new_alpha = _control_contract(
        control_fit_path, component_names
    )
    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    for key, (scores, labels, seen, segments, static) in scored.items():
        if static is None:
            raise ValueError("static candidate features were not collected")
        prepared[key] = {
            "scores": scores,
            "labels": labels,
            "seen": seen,
            "segments": segments,
            "static": static,
            "control": _control_score(
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
    if not (0 < train_rows < len(history["labels"])):
        raise ValueError("train_rows must leave a non-empty validation holdout")
    static_mean, static_std = _static_normalization(
        [history["static"], testpool["static"]], train_rows
    )
    for values in prepared.values():
        values["feature"] = _features(
            values["scores"],
            values["static"],
            values["control"],
            values["seen"],
            static_mean,
            static_std,
        )

    frozen_v12 = None
    if baseline_report_path is not None:
        if _sha256(baseline_report_path) != EXPECTED_V12_REPORT_SHA256:
            raise ValueError("scaled v21 baseline is not the frozen submitted v12 report")
        frozen_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        frozen_mean = np.asarray(frozen_report["training"]["static_mean"], dtype=np.float32)
        frozen_std = np.asarray(frozen_report["training"]["static_std"], dtype=np.float32)
        frozen_nets = []
        for member in frozen_report["training"]["members"]:
            checkpoint = Path(member["checkpoint"]).resolve()
            if _sha256(checkpoint) != member["sha256"]:
                raise ValueError(f"frozen v12 checkpoint hash differs: {checkpoint}")
            net, metadata = load_checkpoint(checkpoint)
            if int(metadata["feature_count"]) != int(frozen_report["feature_count"]):
                raise ValueError("frozen v12 feature count differs")
            frozen_nets.append(net)
        frozen_v12 = {
            "path": str(baseline_report_path),
            "sha256": _sha256(baseline_report_path),
            "metrics": {},
        }
        for key, values in prepared.items():
            strategy, split = key
            evaluation_slice = slice(train_rows, None) if split == "validation" else slice(None)
            frozen_feature = _features(
                values["scores"], values["static"], values["control"], values["seen"],
                frozen_mean, frozen_std,
            )[evaluation_slice]
            prediction = _qnorm(np.mean([
                _predict(net, frozen_feature, batch) for net in frozen_nets
            ], axis=0))
            control = values["control"][evaluation_slice]
            candidate = _candidate_score(
                control, prediction, values["seen"][evaluation_slice],
                float(frozen_report["residual_alpha"]),
            )
            segments = {
                name: mask[evaluation_slice] for name, mask in values["segments"].items()
            }
            output_split = "holdout" if split == "validation" else split
            frozen_v12["metrics"].setdefault(strategy, {})[output_split] = _evaluation(
                control, candidate, values["labels"][evaluation_slice], segments
            )
            del frozen_feature

    training_parts = []
    for values in (history, testpool):
        target_new = ~values["seen"][
            np.arange(train_rows), values["labels"][:train_rows]
        ]
        ids = np.flatnonzero(target_new)
        training_parts.append(
            (
                values["feature"][ids],
                values["control"][ids],
                values["seen"][ids],
                values["labels"][ids],
            )
        )
    train_feature, train_control, train_seen, train_labels = (
        np.concatenate([part[index] for part in training_parts], axis=0)
        for index in range(4)
    )
    del training_parts
    holdout = slice(train_rows, None)
    select_feature = history["feature"][holdout]
    select_control = history["control"][holdout]
    select_seen = history["seen"][holdout]
    select_labels = history["labels"][holdout]

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    records = []
    nets = []
    selection_predictions = []
    for hidden, seed in members:
        net, prediction, record = _train_member(
            train_feature=train_feature,
            train_control=train_control,
            train_seen=train_seen,
            train_labels=train_labels,
            select_feature=select_feature,
            select_control=select_control,
            select_seen=select_seen,
            select_labels=select_labels,
            hidden=hidden,
            seed=seed,
            epochs=epochs,
            batch=batch,
        )
        path = checkpoint_dir / f"pairnew_set{hidden}_seed{seed}.npz"
        _save_checkpoint(
            path,
            net,
            feature_count=train_feature.shape[-1],
            hidden=hidden,
            seed=seed,
            epoch=record["best_epoch"],
            static_mean=static_mean,
            static_std=static_std,
        )
        reloaded, metadata = load_checkpoint(path)
        before = _predict(net, select_feature[:256], batch)
        after = _predict(reloaded, select_feature[:256], batch)
        if not np.allclose(before, after, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"checkpoint reload mismatch: {path}")
        record.update(
            {
                "checkpoint": str(path),
                "sha256": _sha256(path),
                "checkpoint_metadata": {
                    key: value
                    for key, value in metadata.items()
                    if key not in {"static_mean", "static_std"}
                },
            }
        )
        records.append(record)
        nets.append(net)
        selection_predictions.append(prediction)

    selection_residual = _qnorm(np.mean(selection_predictions, axis=0))
    alpha, selection_mrr, selection_pair_seen_mrr = _tune_gate(
        select_control, selection_residual, select_seen, select_labels
    )
    evaluations = {}
    for key, values in prepared.items():
        strategy, split = key
        if key == ("history", "validation"):
            residual = selection_residual
            evaluation_slice = holdout
        else:
            evaluation_slice = holdout if split == "validation" else slice(None)
            predictions = [
                _predict(net, values["feature"][evaluation_slice], batch)
                for net in nets
            ]
            residual = _qnorm(np.mean(predictions, axis=0))
        control = values["control"][evaluation_slice]
        candidate = _candidate_score(
            control,
            residual,
            values["seen"][evaluation_slice],
            alpha,
        )
        segments = {
            name: mask[evaluation_slice] for name, mask in values["segments"].items()
        }
        output_split = "holdout" if split == "validation" else split
        evaluations.setdefault(strategy, {})[output_split] = _evaluation(
            control, candidate, values["labels"][evaluation_slice], segments
        )

    checks = {
        "residual_weight_active": alpha >= 0.025,
        "history_holdout_delta": evaluations["history"]["holdout"]["delta"] >= 0.0005,
        "history_confirmation_delta": evaluations["history"]["confirmation"]["delta"]
        >= 0.0002,
        "history_confirmation_positive_after_one_se": evaluations["history"]["confirmation"]["delta"]
        >= evaluations["history"]["confirmation"]["delta_se"],
        "test_pool_holdout_nonnegative": evaluations["test_pool"]["holdout"]["delta"]
        >= 0.0,
        "test_pool_confirmation_nonnegative": evaluations["test_pool"]["confirmation"]["delta"]
        >= 0.0,
    }
    for strategy in ("history", "test_pool"):
        report = evaluations[strategy]["confirmation"]
        for segment in ("pair_new", "pair_seen", "source_hot"):
            before = report["control"]["segments"][segment]["mrr"]
            after = report["candidate"]["segments"][segment]["mrr"]
            checks[f"{strategy}_confirmation_{segment}"] = (
                before is not None
                and after is not None
                and (
                    abs(after - before) <= 1e-12
                    if segment == "pair_seen"
                    else after >= before - 0.001
                )
            )

    feature_names = [
        *(f"component:{name}" for name in component_names),
        *(f"causal:{name}" for name in data_features.FEATURE_NAMES),
        "frozen_control",
        "candidate_pair_new",
    ]
    return {
        "kind": "d4_pairnew_rank_slot_scaled_replay_transformer_v21",
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "selection_replay": f"history validation rows {train_rows}:{len(history['labels'])} only",
        "training_replays": [
            f"history validation rows 0:{train_rows}",
            f"test_pool validation rows 0:{train_rows}",
        ],
        "confirmation_excluded_from_selection": True,
        "test_pool_holdout_is_diagnostic_only": True,
        "control_fit": {
            "path": str(control_fit_path),
            "sha256": _sha256(control_fit_path),
            "kind": control_report["kind"],
            "best_component": control_report["best_component"],
            "seen_alpha": seen_alpha,
            "new_alpha": new_alpha,
        },
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "architecture": {
            "layers": LAYERS,
            "heads": HEADS,
            "residual_scale": RESIDUAL_SCALE,
            "alpha_search": "0.000:0.025:2.000 on history validation holdout",
            "candidate_gate": "the residual ranks pair-new candidates into their original control-score slots",
            "pair_seen_invariant": "seen control-rank slots and the pair-new slot multiset are fixed, so every seen-candidate rank is unchanged even under ties",
            "member_blend": "equal mean then row normalization",
        },
        "training": {
            "requested_rows_per_replay": train_rows,
            "pair_new_rows": int(len(train_labels)),
            "epochs": epochs,
            "batch_rows": batch,
            "members": records,
            "static_mean": static_mean.tolist(),
            "static_std": static_std.tolist(),
        },
        "residual_alpha": alpha,
        "residual_margin": 0.0,
        "selection_mrr": selection_mrr,
        "selection_pair_seen_mrr": selection_pair_seen_mrr,
        "metrics": evaluations,
        "frozen_v12_same_rows": frozen_v12,
        "checks": checks,
    }


def _weighted_residual(
    predictions: list[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    if (
        not predictions
        or len(predictions) != len(weights)
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-7)
    ):
        raise ValueError("member weights must be a nonnegative simplex vector")
    stacked = np.stack(predictions, axis=0)
    if stacked.ndim != 3 or not np.isfinite(stacked).all():
        raise ValueError("member predictions must be finite row-by-candidate matrices")
    return _qnorm(np.sum(stacked * weights[:, None, None], axis=0))


def _weight_candidates(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = len(members)
    if count != 6:
        raise ValueError("weighted v13 requires the frozen six-member v12 ensemble")
    hidden = np.asarray([int(member["hidden"]) for member in members])
    if sorted(hidden.tolist()) != [64, 64, 64, 96, 96, 96]:
        raise ValueError("weighted v13 requires three hidden-64 and three hidden-96 members")
    identities = [(int(member["hidden"]), int(member["seed"])) for member in members]
    if len(set(identities)) != count:
        raise ValueError("weighted v13 member identities must be unique")

    candidates: list[dict[str, Any]] = []

    def add(name: str, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        if any(np.allclose(weights, item["weights"], rtol=0.0, atol=1e-12) for item in candidates):
            return
        candidates.append({"name": name, "weights": weights})

    add("equal6", np.ones(count))
    for index, member in enumerate(members):
        weights = np.ones(count)
        weights[index] = 0.0
        add(f"leave_out_h{int(member['hidden'])}_seed{int(member['seed'])}", weights)

    group64 = hidden == 64
    group96 = hidden == 96
    add("hidden64_only", group64.astype(np.float64))
    add("hidden96_only", group96.astype(np.float64))
    for hidden64_share in (0.25, 0.375, 0.625, 0.75):
        weights = np.zeros(count, dtype=np.float64)
        weights[group64] = hidden64_share / int(group64.sum())
        weights[group96] = (1.0 - hidden64_share) / int(group96.sum())
        add(f"family_h64_{hidden64_share:.3f}", weights)
    if len(candidates) != 13:
        raise AssertionError("weighted v13 candidate family must contain exactly 13 blends")
    return candidates


def _paired_comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    segments: dict[str, np.ndarray],
) -> dict[str, Any]:
    before = _reciprocal_ranks(reference, labels)
    after = _reciprocal_ranks(candidate, labels)
    paired = after - before
    return {
        "reference": data_features.ranking_metrics(reference, labels, segments=segments),
        "candidate": data_features.ranking_metrics(candidate, labels, segments=segments),
        "delta": float(paired.mean()),
        "delta_se": float(paired.std(ddof=1) / np.sqrt(len(paired))),
        "positive_rows": float(np.mean(paired > 0.0)),
        "negative_rows": float(np.mean(paired < 0.0)),
    }


def run_weighted_audit(
    *,
    scored: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]],
    component_names: list[str],
    control_fit_path: Path,
    base_report_path: Path,
    batch: int,
) -> dict[str, Any]:
    """Select a constrained v12 member blend on history holdout only."""
    if batch < 1:
        raise ValueError("batch must be positive")
    if _sha256(base_report_path) != EXPECTED_V12_REPORT_SHA256:
        raise ValueError("base pair-new report is not the frozen submitted v12 report")
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    if (
        base_report.get("kind") != "d4_pairnew_rank_slot_candidate_set_transformer_v12"
        or base_report.get("decision") != "PASS"
        or base_report.get("data_sha256") != EXPECTED_DATA_SHA256
        or base_report.get("component_names") != component_names
        or base_report.get("selection_replay")
        != "history validation rows 20000:30000 only"
        or base_report.get("confirmation_excluded_from_selection") is not True
        or base_report.get("test_pool_holdout_is_diagnostic_only") is not True
        or base_report.get("architecture", {}).get("member_blend")
        != "equal mean then row normalization"
    ):
        raise ValueError("base pair-new report does not satisfy the frozen v12 contract")

    control_report, indices, base_index, seen_alpha, new_alpha = _control_contract(
        control_fit_path, component_names
    )
    if base_report["control_fit"].get("sha256") != _sha256(control_fit_path):
        raise ValueError("weighted audit control report differs from v12")
    train_rows = int(base_report["training"]["requested_rows_per_replay"])
    base_alpha = float(base_report["residual_alpha"])
    if train_rows != V12_TRAIN_ROWS or abs(base_alpha - V12_BASE_ALPHA) > 1e-12:
        raise ValueError("frozen v12 selection rows or alpha differ")
    static_mean = np.asarray(base_report["training"]["static_mean"], dtype=np.float32)
    static_std = np.asarray(base_report["training"]["static_std"], dtype=np.float32)

    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    expected_keys = {
        ("history", "validation"),
        ("history", "confirmation"),
        ("test_pool", "validation"),
        ("test_pool", "confirmation"),
    }
    if set(scored) != expected_keys:
        raise ValueError("weighted v13 requires exactly four frozen replay groups")
    for key, (scores, labels, seen, segments, static) in scored.items():
        if len(labels) != V12_REPLAY_ROWS:
            raise ValueError(f"weighted v13 replay row count differs: {key}")
        if static is None:
            raise ValueError("static candidate features were not collected")
        control = _control_score(
            scores,
            seen,
            control_report,
            indices,
            base_index,
            seen_alpha,
            new_alpha,
        )
        prepared[key] = {
            "labels": labels,
            "seen": seen,
            "segments": segments,
            "control": control,
            "feature": _features(
                scores, static, control, seen, static_mean, static_std
            ),
        }

    history = prepared[("history", "validation")]
    if not (0 < train_rows < len(history["labels"])):
        raise ValueError("frozen v12 train rows do not leave a history holdout")
    members = list(base_report["training"]["members"])
    nets = []
    member_records = []
    for member in members:
        checkpoint = Path(member["checkpoint"]).resolve()
        if _sha256(checkpoint) != member["sha256"]:
            raise ValueError(f"pair-new checkpoint hash differs: {checkpoint}")
        net, metadata = load_checkpoint(checkpoint)
        if (
            int(metadata["feature_count"]) != int(base_report["feature_count"])
            or int(metadata["hidden"]) != int(member["hidden"])
            or int(metadata["seed"]) != int(member["seed"])
            or not np.array_equal(metadata["static_mean"], static_mean)
            or not np.array_equal(metadata["static_std"], static_std)
        ):
            raise ValueError(f"pair-new checkpoint metadata differs: {checkpoint}")
        nets.append(net)
        member_records.append(
            {
                "hidden": int(member["hidden"]),
                "seed": int(member["seed"]),
                "checkpoint": str(checkpoint),
                "sha256": member["sha256"],
            }
        )

    holdout = slice(train_rows, None)
    selection_predictions = [
        _predict(net, history["feature"][holdout], batch) for net in nets
    ]
    selection_control = history["control"][holdout]
    selection_seen = history["seen"][holdout]
    selection_labels = history["labels"][holdout]
    selection_segments = {
        name: mask[holdout] for name, mask in history["segments"].items()
    }

    candidates = _weight_candidates(member_records)
    selection_records = []
    selected = None
    equal_score = None
    equal_weights = None
    for index, item in enumerate(candidates):
        residual = _weighted_residual(selection_predictions, item["weights"])
        if item["name"] == "equal6":
            alpha = base_alpha
            score = _candidate_score(
                selection_control, residual, selection_seen, alpha
            )
            value = _mrr(score, selection_labels)
            pair_seen = selection_seen[np.arange(len(selection_labels)), selection_labels]
            pair_seen_mrr = _mrr(score[pair_seen], selection_labels[pair_seen])
            equal_score = score
            equal_weights = item["weights"]
        else:
            alpha, value, pair_seen_mrr = _tune_gate(
                selection_control, residual, selection_seen, selection_labels
            )
            score = _candidate_score(
                selection_control, residual, selection_seen, alpha
            )
        record = {
            "name": item["name"],
            "weights": item["weights"].tolist(),
            "alpha": float(alpha),
            "selection_mrr": float(value),
            "selection_pair_seen_mrr": float(pair_seen_mrr),
            "nonzero_members": int(np.count_nonzero(item["weights"])),
            "selection_order": index,
        }
        selection_records.append(record)
        key = (float(value), -float(alpha), -index)
        if selected is None or key > selected[0]:
            selected = (key, record, score, item["weights"])

    if equal_score is None or equal_weights is None or selected is None:
        raise RuntimeError("weighted candidate search did not evaluate the equal baseline")
    selected_record = selected[1]
    selected_score = selected[2]
    selected_weights = np.asarray(selected[3], dtype=np.float64)
    selection_comparison = _paired_comparison(
        equal_score,
        selected_score,
        selection_labels,
        selection_segments,
    )

    evaluations: dict[str, dict[str, Any]] = {}
    baseline_errors = []
    for key, values in prepared.items():
        strategy, split = key
        evaluation_slice = holdout if split == "validation" else slice(None)
        if key == ("history", "validation"):
            predictions = selection_predictions
            baseline = equal_score
            weighted = selected_score
        else:
            predictions = [
                _predict(net, values["feature"][evaluation_slice], batch)
                for net in nets
            ]
            baseline = _candidate_score(
                values["control"][evaluation_slice],
                _weighted_residual(predictions, equal_weights),
                values["seen"][evaluation_slice],
                base_alpha,
            )
            weighted = _candidate_score(
                values["control"][evaluation_slice],
                _weighted_residual(predictions, selected_weights),
                values["seen"][evaluation_slice],
                float(selected_record["alpha"]),
            )
        control = values["control"][evaluation_slice]
        labels = values["labels"][evaluation_slice]
        segments = {
            name: mask[evaluation_slice] for name, mask in values["segments"].items()
        }
        output_split = "holdout" if split == "validation" else split
        baseline_evaluation = _evaluation(control, baseline, labels, segments)
        weighted_evaluation = _evaluation(control, weighted, labels, segments)
        comparison = _paired_comparison(baseline, weighted, labels, segments)
        seen_mask = values["seen"][evaluation_slice]
        comparison["pair_seen_rank_exact"] = bool(
            np.array_equal(
                _rank_positions(baseline)[seen_mask],
                _rank_positions(weighted)[seen_mask],
            )
        )
        evaluations.setdefault(strategy, {})[output_split] = {
            "baseline_equal6": baseline_evaluation,
            "candidate_weighted": weighted_evaluation,
            "weighted_vs_equal6": comparison,
        }
        expected = float(base_report["metrics"][strategy][output_split]["candidate"]["mrr"])
        baseline_errors.append(abs(expected - baseline_evaluation["candidate"]["mrr"]))

    comparisons = {
        f"{strategy}_{split}": evaluations[strategy][split]["weighted_vs_equal6"]
        for strategy in ("history", "test_pool")
        for split in ("holdout", "confirmation")
    }

    def clears(record: dict[str, Any], floor: float, se_fraction: float) -> bool:
        return float(record["delta"]) >= max(
            floor, se_fraction * float(record["delta_se"])
        )

    checks = {
        "selected_non_equal_blend": selected_record["name"] != "equal6",
        "selection_history_holdout_delta": clears(
            comparisons["history_holdout"], 0.0003, 0.5
        ),
        "history_confirmation_delta": clears(
            comparisons["history_confirmation"], 0.0001, 0.2
        ),
        "test_pool_holdout_delta": clears(
            comparisons["test_pool_holdout"], 0.0001, 0.2
        ),
        "test_pool_confirmation_delta": clears(
            comparisons["test_pool_confirmation"], 0.0001, 0.2
        ),
        "baseline_replay_reproduced": max(baseline_errors) <= 0.0002,
        "base_alpha_unchanged": abs(base_alpha - V12_BASE_ALPHA) <= 1e-12,
    }
    for name, comparison in comparisons.items():
        before = comparison["reference"]["segments"]["pair_seen"]["mrr"]
        after = comparison["candidate"]["segments"]["pair_seen"]["mrr"]
        checks[f"{name}_pair_seen_invariant"] = (
            before is not None
            and after is not None
            and abs(after - before) <= 1e-12
            and comparison["pair_seen_rank_exact"]
        )

    return {
        "kind": "d4_pairnew_rank_slot_constrained_member_weight_v13",
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "base_report": {
            "path": str(base_report_path),
            "sha256": _sha256(base_report_path),
            "kind": base_report["kind"],
            "residual_alpha": base_alpha,
        },
        "control_fit": {
            "path": str(control_fit_path),
            "sha256": _sha256(control_fit_path),
            "kind": control_report["kind"],
            "best_component": control_report["best_component"],
            "seen_alpha": seen_alpha,
            "new_alpha": new_alpha,
        },
        "selection_policy": {
            "data": f"history validation rows {train_rows}:30000 only",
            "candidate_count": len(selection_records),
            "candidate_family": "equal6, leave-one-out, hidden-family-only, and four coarse hidden-family mixtures",
            "tie_break": "smaller alpha, then earlier simpler candidate",
            "confirmation_excluded": True,
            "test_pool_excluded": True,
        },
        "selection_candidates": selection_records,
        "selected": selected_record,
        "member_weights": selected_record["weights"],
        "residual_alpha": selected_record["alpha"],
        "selection_weighted_vs_equal6": selection_comparison,
        "feature_names": base_report["feature_names"],
        "feature_count": base_report["feature_count"],
        "architecture": {
            **base_report["architecture"],
            "member_blend": "frozen constrained nonnegative weights then row normalization",
        },
        "training": {
            **base_report["training"],
            "members": members,
            "reused_frozen_checkpoints": True,
        },
        "metrics": evaluations,
        "baseline_replay_max_abs_mrr_error": max(baseline_errors),
        "checks": checks,
        "confirmation_excluded_from_selection": True,
        "test_pool_holdout_is_diagnostic_only": True,
        "members": member_records,
    }


def _self_check() -> None:
    rng = np.random.default_rng(20260812)
    control = rng.integers(-2, 3, size=(32, 100)).astype(np.float32)
    residual = rng.normal(size=control.shape).astype(np.float32)
    seen = rng.random(control.shape) < 0.2
    seen[:, 0] = True
    labels = np.asarray(
        [rng.choice(np.flatnonzero(row)) for row in seen], dtype=np.int64
    )
    unchanged = _candidate_score(control, residual, seen, 0.0)
    changed = _candidate_score(control, residual, seen, 0.8)
    if not np.array_equal(
        _reciprocal_ranks(control, labels), _reciprocal_ranks(unchanged, labels)
    ):
        raise AssertionError("zero-weight rank slots changed the control order")
    if not np.array_equal(
        _reciprocal_ranks(control, labels), _reciprocal_ranks(changed, labels)
    ):
        raise AssertionError("pair-new slot projection changed a seen-target rank")
    slots = _strict_control_slots(control)
    for row in range(len(control)):
        mask = ~seen[row]
        if not np.array_equal(np.sort(changed[row, mask]), np.sort(slots[row, mask])):
            raise AssertionError("pair-new rank-slot multiset changed")


if __name__ == "__main__":
    _self_check()
