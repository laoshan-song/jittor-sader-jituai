#!/usr/bin/env python3
"""Jittor-only candidate-conditioned temporal attention ranker for Dataset3/4.

The data layer supplies compact source/item indices and a strictly causal
recent-item window for every query.  Index zero is reserved for unknown or
padding, so a model trained on warm entities has a defined cold fallback.
This module deliberately does not read CSV or construct histories: keeping
that boundary separate makes temporal leakage auditable.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

os.environ.update({
    "use_cutt": "0",
    "use_cutlass": "0",
    "use_nccl": "0",
    "use_mkl": "0",
})

import jittor as jt
from jittor import nn


CANDIDATE_COUNT = 100
# Kept here rather than importing the data layer so the scorer remains usable
# with precomputed tensors. It matches data_features.FEATURE_NAMES.
PAIR_SEEN_FEATURE_INDEX = 2


def configure_cuda() -> None:
    """Require CUDA explicitly; Dataset4 training is not a CPU fallback job."""
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is unavailable")
    jt.flags.use_cuda = 1


def configure_seed(seed: int) -> np.random.Generator:
    seed = int(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)
    return np.random.default_rng(seed)


def _integer(values: np.ndarray, name: str, *, ndim: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype")
    if len(values) and int(values.min()) < 0:
        raise ValueError(f"{name} cannot contain negative indices")
    return values


def _source(values: np.ndarray, rows: int, source_count: int) -> np.ndarray:
    values = _integer(values, "source", ndim=1)
    if len(values) != rows:
        raise ValueError("source length differs from candidate rows")
    if len(values) and int(values.max()) >= source_count:
        raise ValueError("source index exceeds source_count")
    return values


def _candidates(values: np.ndarray, rows: int, item_count: int) -> np.ndarray:
    values = _integer(values, "candidates", ndim=2)
    if values.shape != (rows, CANDIDATE_COUNT):
        raise ValueError(
            f"candidates must have shape ({rows}, {CANDIDATE_COUNT}), got {values.shape}"
        )
    if len(values) and int(values.max()) >= item_count:
        raise ValueError("candidate index exceeds item_count")
    return values


def _history(
    values: np.ndarray, gap: np.ndarray, rows: int, item_count: int
) -> tuple[np.ndarray, np.ndarray]:
    values = _integer(values, "history", ndim=2)
    gap = np.asarray(gap)
    if gap.shape != values.shape or values.shape[0] != rows or values.shape[1] < 1:
        raise ValueError("history and history_gap must have matching shape (rows, history_length)")
    if len(values) and int(values.max()) >= item_count:
        raise ValueError("history index exceeds item_count")
    if not np.issubdtype(gap.dtype, np.number) or not np.isfinite(gap).all():
        raise ValueError("history_gap must be finite numeric values")
    if np.any(gap < 0.0):
        raise ValueError("history_gap cannot be negative")
    return values, gap


def _labels(values: np.ndarray, rows: int) -> np.ndarray:
    values = _integer(values, "labels", ndim=1)
    if len(values) != rows or (len(values) and int(values.max()) >= CANDIDATE_COUNT):
        raise ValueError("labels must be candidate columns in [0, 99]")
    return values


def _sample_weight(values: np.ndarray | None, rows: int) -> np.ndarray | None:
    if values is None:
        return None
    values = np.asarray(values)
    if values.shape != (rows,) or not np.issubdtype(values.dtype, np.number):
        raise ValueError("sample_weight must be a numeric vector matching candidate rows")
    values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("sample_weight must be finite and strictly positive")
    return values


def _features(values: np.ndarray | None, rows: int) -> np.ndarray | None:
    if values is None:
        return None
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[:2] != (rows, CANDIDATE_COUNT):
        raise ValueError("features must have shape (rows, 100, feature_dim)")
    if values.shape[2] < 1 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("features must have a positive numeric feature dimension")
    return values


def _batch(values: np.ndarray, dtype: np.dtype[Any]) -> jt.Var:
    values = np.ascontiguousarray(values, dtype=dtype)
    if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
        raise ValueError("floating-point model input contains NaN or infinity")
    return jt.array(values)


class TemporalAttentionRanker(nn.Module):
    """MF, item bias, and candidate-conditioned attention over recent history."""

    def __init__(
        self,
        source_count: int,
        item_count: int,
        *,
        embedding_dim: int = 32,
        feature_dim: int = 0,
        static_context: bool = False,
        static_context_pair_seen_only: bool = False,
        dropout: float = 0.0,
        time_scale: float = 0.25,
    ):
        super().__init__()
        self.source_count = int(source_count)
        self.item_count = int(item_count)
        self.embedding_dim = int(embedding_dim)
        self.feature_dim = int(feature_dim)
        self.static_context = bool(static_context)
        self.static_context_pair_seen_only = bool(static_context_pair_seen_only)
        self.dropout_rate = float(dropout)
        self.time_scale = float(time_scale)
        if self.source_count < 1 or self.item_count < 1:
            raise ValueError("source_count and item_count must be positive")
        if self.embedding_dim < 1 or self.feature_dim < 0:
            raise ValueError("embedding_dim must be positive and feature_dim non-negative")
        if self.static_context and not self.feature_dim:
            raise ValueError("static_context requires a positive feature_dim")
        if self.static_context_pair_seen_only and not self.static_context:
            raise ValueError("static_context_pair_seen_only requires static_context")
        if self.static_context_pair_seen_only and self.feature_dim <= PAIR_SEEN_FEATURE_INDEX:
            raise ValueError("static_context_pair_seen_only requires the pair_seen feature")
        if not 0.0 <= self.dropout_rate < 1.0 or self.time_scale < 0.0:
            raise ValueError("dropout must be in [0, 1), time_scale must be non-negative")

        self.source = nn.Embedding(self.source_count, self.embedding_dim)
        self.item = nn.Embedding(self.item_count, self.embedding_dim)
        self.history_item = nn.Embedding(self.item_count, self.embedding_dim)
        self.item_bias = nn.Embedding(self.item_count, 1)
        for embedding in (self.source, self.item, self.history_item):
            embedding.weight.assign(jt.randn(embedding.weight.shape) * 0.01)
        self.item_bias.weight.assign(jt.zeros(self.item_bias.weight.shape))
        self.query = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.key = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value = nn.Linear(self.embedding_dim, self.embedding_dim)
        # Explicit identity matches complement embedding attention for the
        # repeat-edge regime, while the data layer guarantees past-only items.
        self.history_match = nn.Linear(2, 1)
        # A finite learned warm/cold offset is trained only when a causal
        # replay deliberately injects history-domain unseen negative ids.
        self.candidate_known = nn.Linear(1, 1)
        self.candidate_known.weight.assign(jt.zeros(self.candidate_known.weight.shape))
        self.candidate_known.bias.assign(jt.zeros(self.candidate_known.bias.shape))
        self.dropout = nn.Dropout(self.dropout_rate)
        if self.feature_dim:
            self.feature_mlp = nn.Sequential(
                nn.Linear(self.feature_dim, 32),
                nn.Relu(),
                nn.Linear(32, 1),
            )
        else:
            self.feature_mlp = None
        if self.static_context:
            self.static_context_encoder = nn.Sequential(
                nn.Linear(self.feature_dim, 32),
                nn.Relu(),
            )
            self.static_context_mlp = nn.Sequential(
                nn.Linear(64, 32),
                nn.Relu(),
                nn.Linear(32, 1),
            )
        else:
            self.static_context_encoder = None
            self.static_context_mlp = None

    def execute(
        self,
        source: jt.Var,
        candidates: jt.Var,
        history: jt.Var,
        history_gap: jt.Var,
        features: jt.Var | None = None,
    ) -> jt.Var:
        batch = source.shape[0]
        if candidates.shape[:2] != (batch, CANDIDATE_COUNT):
            raise ValueError("candidate tensor shape differs from source batch")
        if history.shape[0] != batch or history_gap.shape != history.shape:
            raise ValueError("history tensors differ from source batch")
        # Row zero represents OOV/padding.  Mask it explicitly instead of
        # allowing an untrained random embedding to affect a cold query.
        candidate_known = (candidates > 0).unsqueeze(-1)
        source_known = (source > 0).unsqueeze(1).unsqueeze(2)
        history_known = (history > 0).unsqueeze(-1)
        candidate_vec = self.item(candidates) * candidate_known
        source_vec = self.source(source).unsqueeze(1) * source_known
        mf = (source_vec * candidate_vec).sum(dim=2)
        bias = self.item_bias(candidates).squeeze(-1) * (candidates > 0)

        history_vec = self.dropout(self.history_item(history)) * history_known
        length = history.shape[1]
        query = self.query(candidate_vec.reshape((-1, self.embedding_dim)))
        query = query.reshape((batch, CANDIDATE_COUNT, self.embedding_dim))
        key = self.key(history_vec.reshape((-1, self.embedding_dim)))
        key = key.reshape((batch, length, self.embedding_dim))
        value = self.value(history_vec.reshape((-1, self.embedding_dim)))
        value = value.reshape((batch, length, self.embedding_dim))
        attention_score = (query.unsqueeze(2) * key.unsqueeze(1)).sum(dim=3)
        attention_score /= float(self.embedding_dim) ** 0.5
        attention_score -= history_gap.unsqueeze(1) * self.time_scale
        valid = history > 0
        any_valid = valid.sum(dim=1, keepdims=True) > 0
        attention_score = jt.where(
            valid.unsqueeze(1), attention_score, jt.full_like(attention_score, -1e9)
        )
        attention = nn.softmax(attention_score, dim=2)
        attention = jt.where(any_valid.unsqueeze(1), attention, jt.zeros_like(attention))
        context = (attention.unsqueeze(3) * value.unsqueeze(1)).sum(dim=2)
        sequence = (context * candidate_vec).sum(dim=2)
        matches = (candidates.unsqueeze(2) == history.unsqueeze(1)) * valid.unsqueeze(1)
        match_count = matches.sum(dim=2)
        match_recency = (
            matches / (1.0 + history_gap.unsqueeze(1))
        ).sum(dim=2)
        match_input = jt.concat(
            [match_count.unsqueeze(2), match_recency.unsqueeze(2)], dim=2
        )
        exact = self.history_match(match_input.reshape((-1, 2)))
        exact = exact.reshape((batch, CANDIDATE_COUNT))
        known_feature = (candidates > 0).reshape((-1, 1)).float32()
        known_score = self.candidate_known(known_feature)
        known_score = known_score.reshape((batch, CANDIDATE_COUNT))
        score = mf + bias + sequence + exact + known_score
        if self.feature_mlp is not None:
            if features is None or features.shape != (batch, CANDIDATE_COUNT, self.feature_dim):
                raise ValueError("configured feature_mlp needs matching feature tensor")
            static = self.feature_mlp(features.reshape((-1, self.feature_dim)))
            score += static.reshape((batch, CANDIDATE_COUNT))
            if self.static_context:
                encoded = self.static_context_encoder(
                    features.reshape((-1, self.feature_dim))
                )
                encoded = encoded.reshape((batch, CANDIDATE_COUNT, 32))
                context = encoded.mean(dim=1, keepdims=True)
                combined = jt.concat([encoded, encoded * 0.0 + context], dim=2)
                residual = self.static_context_mlp(combined.reshape((-1, 64)))
                residual = residual.reshape((batch, CANDIDATE_COUNT))
                if self.static_context_pair_seen_only:
                    residual *= (features[:, :, PAIR_SEEN_FEATURE_INDEX] > 0.0).float32()
                score += residual
        elif features is not None:
            raise ValueError("ranker was built without static features")
        return score


def model_config(model: TemporalAttentionRanker) -> dict[str, Any]:
    return {
        "kind": "temporal_attention",
        "source_count": model.source_count,
        "item_count": model.item_count,
        "embedding_dim": model.embedding_dim,
        "feature_dim": model.feature_dim,
        "static_context": model.static_context,
        "static_context_pair_seen_only": model.static_context_pair_seen_only,
        "dropout": model.dropout_rate,
        "time_scale": model.time_scale,
    }


def build_ranker(config: dict[str, Any]) -> TemporalAttentionRanker:
    if config.get("kind") != "temporal_attention":
        raise ValueError("checkpoint kind must be temporal_attention")
    return TemporalAttentionRanker(
        int(config["source_count"]),
        int(config["item_count"]),
        embedding_dim=int(config["embedding_dim"]),
        feature_dim=int(config["feature_dim"]),
        static_context=bool(config.get("static_context", False)),
        static_context_pair_seen_only=bool(config.get("static_context_pair_seen_only", False)),
        dropout=float(config["dropout"]),
        time_scale=float(config["time_scale"]),
    )


def _validate_inputs(
    source: np.ndarray,
    candidates: np.ndarray,
    history: np.ndarray,
    history_gap: np.ndarray,
    *,
    source_count: int,
    item_count: int,
    labels: np.ndarray | None = None,
    features: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    rows = len(source)
    source = _source(source, rows, source_count)
    candidates = _candidates(candidates, rows, item_count)
    history, history_gap = _history(history, history_gap, rows, item_count)
    labels = _labels(labels, rows) if labels is not None else None
    features = _features(features, rows)
    sample_weight = _sample_weight(sample_weight, rows)
    return source, candidates, history, history_gap, labels, features, sample_weight


def _model_score(
    model: TemporalAttentionRanker,
    source: np.ndarray,
    candidates: np.ndarray,
    history: np.ndarray,
    history_gap: np.ndarray,
    features: np.ndarray | None,
) -> jt.Var:
    return model(
        _batch(source, np.int32),
        _batch(candidates, np.int32),
        _batch(history, np.int32),
        _batch(history_gap, np.float32),
        _batch(features, np.float32) if features is not None else None,
    )


def train_ranker(
    source: np.ndarray,
    candidates: np.ndarray,
    history: np.ndarray,
    history_gap: np.ndarray,
    labels: np.ndarray,
    *,
    source_count: int,
    item_count: int,
    features: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    embedding_dim: int = 32,
    static_context: bool = False,
    static_context_pair_seen_only: bool = False,
    dropout: float = 0.0,
    time_scale: float = 0.25,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
    seed: int = 20260810,
    verbose: bool = False,
) -> tuple[TemporalAttentionRanker, list[float]]:
    """Fit group-wise CE with optional positive per-row loss weights."""
    configure_cuda()
    source, candidates, history, history_gap, labels, features, sample_weight = _validate_inputs(
        source,
        candidates,
        history,
        history_gap,
        source_count=source_count,
        item_count=item_count,
        labels=labels,
        features=features,
        sample_weight=sample_weight,
    )
    if labels is None or len(source) < 1:
        raise ValueError("at least one labelled row is required")
    if static_context and features is None:
        raise ValueError("static_context requires static features")
    if static_context_pair_seen_only and not static_context:
        raise ValueError("static_context_pair_seen_only requires static_context")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("invalid training hyperparameters")
    rng = configure_seed(seed)
    model = TemporalAttentionRanker(
        source_count,
        item_count,
        embedding_dim=embedding_dim,
        feature_dim=0 if features is None else int(features.shape[2]),
        static_context=static_context,
        static_context_pair_seen_only=static_context_pair_seen_only,
        dropout=dropout,
        time_scale=time_scale,
    )
    optimizer = jt.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    losses: list[float] = []
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(len(source))
        total = 0.0
        for start in range(0, len(order), int(batch_size)):
            ids = order[start : start + int(batch_size)]
            logits = _model_score(
                model,
                source[ids],
                candidates[ids],
                history[ids],
                history_gap[ids],
                None if features is None else features[ids],
            )
            targets = _batch(labels[ids], np.int32)
            if sample_weight is None:
                loss = nn.cross_entropy_loss(logits, targets)
            else:
                row_loss = nn.cross_entropy_loss(logits, targets, reduction="none")
                weights = _batch(sample_weight[ids], np.float32)
                loss = (row_loss * weights).sum() / (weights.sum() + 1e-6)
            optimizer.step(loss)
            total += float(np.asarray(loss.data).item()) * len(ids)
        losses.append(total / len(source))
        if verbose:
            print(f"temporal_attention epoch={epoch + 1} loss={losses[-1]:.6f}", flush=True)
    return model, losses


def predict_scores(
    model: TemporalAttentionRanker,
    source: np.ndarray,
    candidates: np.ndarray,
    history: np.ndarray,
    history_gap: np.ndarray,
    *,
    features: np.ndarray | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Score a bounded candidate plane without materializing Dataset4 test data."""
    configure_cuda()
    source, candidates, history, history_gap, _, features, _ = _validate_inputs(
        source,
        candidates,
        history,
        history_gap,
        source_count=model.source_count,
        item_count=model.item_count,
        features=features,
    )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    scores = np.empty((len(source), CANDIDATE_COUNT), dtype=np.float32)
    with jt.no_grad():
        for start in range(0, len(source), int(batch_size)):
            stop = min(len(source), start + int(batch_size))
            value = _model_score(
                model,
                source[start:stop],
                candidates[start:stop],
                history[start:stop],
                history_gap[start:stop],
                None if features is None else features[start:stop],
            )
            scores[start:stop] = np.asarray(value.data, dtype=np.float32)
    return scores


def save_checkpoint(path: str | Path, model: TemporalAttentionRanker) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        name: np.asarray(value) for name, value in model_config(model).items()
    }
    payload.update(
        {
            f"param__{name}": np.asarray(value.data, dtype=np.float32).copy()
            for name, value in model.state_dict().items()
        }
    )
    np.savez_compressed(path, **payload)
    return path


def load_checkpoint(path: str | Path) -> tuple[TemporalAttentionRanker, dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        config = {
            "kind": str(archive["kind"].item()),
            "source_count": int(archive["source_count"].item()),
            "item_count": int(archive["item_count"].item()),
            "embedding_dim": int(archive["embedding_dim"].item()),
            "feature_dim": int(archive["feature_dim"].item()),
            "static_context": bool(archive["static_context"].item())
            if "static_context" in archive.files
            else False,
            "static_context_pair_seen_only": bool(
                archive["static_context_pair_seen_only"].item()
            )
            if "static_context_pair_seen_only" in archive.files
            else False,
            "dropout": float(archive["dropout"].item()),
            "time_scale": float(archive["time_scale"].item()),
        }
        model = build_ranker(config)
        state = {
            name.removeprefix("param__"): jt.array(archive[name])
            for name in archive.files
            if name.startswith("param__")
        }
    if set(state) != set(model.state_dict()):
        raise ValueError("checkpoint parameter names do not match model architecture")
    model.load_state_dict(state)
    return model, config


def _self_test() -> None:
    configure_cuda()
    rng = np.random.default_rng(7)
    rows, source_count, item_count, length = 24, 8, 16, 5
    source = rng.integers(0, source_count, size=rows, dtype=np.int32)
    candidates = rng.integers(0, item_count, size=(rows, CANDIDATE_COUNT), dtype=np.int32)
    history = rng.integers(0, item_count, size=(rows, length), dtype=np.int32)
    history[:, -1] = 0
    gap = rng.random((rows, length), dtype=np.float32)
    labels = rng.integers(0, CANDIDATE_COUNT, size=rows, dtype=np.int32)
    features = rng.normal(size=(rows, CANDIDATE_COUNT, 3)).astype(np.float32)
    features[:, :, PAIR_SEEN_FEATURE_INDEX] = rng.integers(
        0, 2, size=(rows, CANDIDATE_COUNT), dtype=np.int32
    )
    sample_weight = np.where(source > 0, 1.5, 1.0).astype(np.float32)
    model, losses = train_ranker(
        source,
        candidates,
        history,
        gap,
        labels,
        source_count=source_count,
        item_count=item_count,
        features=features,
        embedding_dim=8,
        static_context=True,
        epochs=2,
        batch_size=6,
        seed=11,
    )
    score = predict_scores(model, source, candidates, history, gap, features=features, batch_size=7)
    assert score.shape == (rows, CANDIDATE_COUNT) and np.isfinite(score).all()
    assert len(losses) == 2 and np.isfinite(losses).all()
    duplicate_candidates = candidates.copy()
    duplicate_features = features.copy()
    duplicate_candidates[:, 1] = duplicate_candidates[:, 0]
    duplicate_features[:, 1] = duplicate_features[:, 0]
    duplicate_score = predict_scores(
        model,
        source,
        duplicate_candidates,
        history,
        gap,
        features=duplicate_features,
        batch_size=7,
    )
    assert np.allclose(duplicate_score[:, 0], duplicate_score[:, 1], atol=3e-6, rtol=1e-6)
    permutation = rng.permutation(CANDIDATE_COUNT)
    permuted_score = predict_scores(
        model,
        source,
        duplicate_candidates[:, permutation],
        history,
        gap,
        features=duplicate_features[:, permutation],
        batch_size=7,
    )
    assert np.allclose(
        permuted_score,
        duplicate_score[:, permutation],
        atol=3e-6,
        rtol=1e-6,
    )
    try:
        TemporalAttentionRanker(source_count, item_count, static_context=True)
    except ValueError:
        pass
    else:
        raise AssertionError("static_context accepted an absent feature dimension")
    try:
        train_ranker(
            source,
            candidates,
            history,
            gap,
            labels,
            source_count=source_count,
            item_count=item_count,
            static_context=True,
            embedding_dim=8,
            epochs=1,
            batch_size=6,
            seed=11,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("static_context accepted absent static features")
    try:
        train_ranker(
            source,
            candidates,
            history,
            gap,
            labels,
            source_count=source_count,
            item_count=item_count,
            features=features,
            static_context_pair_seen_only=True,
            embedding_dim=8,
            epochs=1,
            batch_size=6,
            seed=11,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("pair-seen context gate accepted absent static_context")
    gated_model, gated_losses = train_ranker(
        source,
        candidates,
        history,
        gap,
        labels,
        source_count=source_count,
        item_count=item_count,
        features=features,
        static_context=True,
        static_context_pair_seen_only=True,
        embedding_dim=8,
        epochs=1,
        batch_size=6,
        seed=13,
    )
    gated_score = predict_scores(
        gated_model, source, candidates, history, gap, features=features, batch_size=7
    )
    assert gated_score.shape == (rows, CANDIDATE_COUNT) and np.isfinite(gated_score).all()
    assert len(gated_losses) == 1 and np.isfinite(gated_losses).all()
    weighted_model, weighted_losses = train_ranker(
        source,
        candidates,
        history,
        gap,
        labels,
        source_count=source_count,
        item_count=item_count,
        features=features,
        sample_weight=sample_weight,
        embedding_dim=8,
        epochs=2,
        batch_size=6,
        seed=11,
    )
    weighted_score = predict_scores(
        weighted_model, source, candidates, history, gap, features=features, batch_size=7
    )
    assert weighted_score.shape == (rows, CANDIDATE_COUNT) and np.isfinite(weighted_score).all()
    assert len(weighted_losses) == 2 and np.isfinite(weighted_losses).all()
    with TemporaryDirectory() as directory:
        checkpoint = save_checkpoint(Path(directory) / "model.npz", gated_model)
        restored, config = load_checkpoint(checkpoint)
        assert config["kind"] == "temporal_attention"
        assert config["static_context"] is True
        assert config["static_context_pair_seen_only"] is True
        assert np.allclose(
            gated_score,
            predict_scores(restored, source, candidates, history, gap, features=features),
            atol=1e-6,
        )
        baseline = TemporalAttentionRanker(source_count, item_count, feature_dim=3)
        legacy_checkpoint = save_checkpoint(Path(directory) / "legacy.npz", baseline)
        with np.load(legacy_checkpoint, allow_pickle=False) as archive:
            legacy_payload = {
                name: archive[name]
                for name in archive.files
                if name not in {"static_context", "static_context_pair_seen_only"}
            }
        np.savez_compressed(legacy_checkpoint, **legacy_payload)
        _, legacy_config = load_checkpoint(legacy_checkpoint)
        assert legacy_config["static_context"] is False
        assert legacy_config["static_context_pair_seen_only"] is False
    print("temporal_attention_jittor self-test passed", flush=True)


if __name__ == "__main__":
    _self_test()
