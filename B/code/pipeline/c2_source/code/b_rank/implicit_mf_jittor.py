#!/usr/bin/env python3
"""Jittor-only implicit matrix factorization for full-history D4 edges."""

from __future__ import annotations

import os
from pathlib import Path
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


def configure_cuda() -> None:
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is unavailable")
    jt.flags.use_cuda = 1


def configure_seed(seed: int) -> np.random.Generator:
    seed = int(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)
    return np.random.default_rng(seed)


def dense_indices(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Map raw uint32 ids to stable 1-based int32 indices; unknown is zero."""
    values = np.asarray(values)
    ids = np.asarray(ids)
    flat = values.reshape(-1)
    output = np.zeros(flat.shape, dtype=np.int32)
    if len(flat) and len(ids):
        positions = np.searchsorted(ids, flat)
        inside = positions < len(ids)
        matched = np.zeros(flat.shape, dtype=bool)
        matched[inside] = ids[positions[inside]] == flat[inside]
        output[matched] = positions[matched].astype(np.int32, copy=False) + 1
    return output.reshape(values.shape)


def _batch(values: np.ndarray, dtype: np.dtype[Any]) -> jt.Var:
    return jt.array(np.ascontiguousarray(values, dtype=dtype))


class ImplicitMF(nn.Module):
    def __init__(self, source_count: int, item_count: int, embedding_dim: int):
        super().__init__()
        self.source_count = int(source_count)
        self.item_count = int(item_count)
        self.embedding_dim = int(embedding_dim)
        if min(self.source_count, self.item_count, self.embedding_dim) < 1:
            raise ValueError("source_count, item_count, and embedding_dim must be positive")
        self.source = nn.Embedding(self.source_count, self.embedding_dim)
        self.item = nn.Embedding(self.item_count, self.embedding_dim)
        self.item_bias = nn.Embedding(self.item_count, 1)
        self.source.weight.assign(jt.randn(self.source.weight.shape) * 0.01)
        self.item.weight.assign(jt.randn(self.item.weight.shape) * 0.01)
        self.item_bias.weight.assign(jt.zeros(self.item_bias.weight.shape))

    def execute(self, source: jt.Var, candidates: jt.Var) -> jt.Var:
        if source.shape[0] != candidates.shape[0] or len(candidates.shape) != 2:
            raise ValueError("source and candidate batch shapes differ")
        source_known = (source > 0).unsqueeze(1).unsqueeze(2)
        candidate_known = (candidates > 0).unsqueeze(2)
        source_vector = self.source(source).unsqueeze(1) * source_known
        item_vector = self.item(candidates) * candidate_known
        score = (source_vector * item_vector).sum(dim=2)
        score += self.item_bias(candidates).squeeze(2) * (candidates > 0)
        return score


def train_full_history(
    raw_source: np.ndarray,
    raw_item: np.ndarray,
    source_ids: np.ndarray,
    item_ids: np.ndarray,
    *,
    embedding_dim: int = 64,
    negative_count: int = 32,
    epochs: int = 3,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
    seed: int = 20260810,
    verbose: bool = False,
) -> tuple[ImplicitMF, list[float]]:
    configure_cuda()
    raw_source = np.asarray(raw_source)
    raw_item = np.asarray(raw_item)
    source_ids = np.asarray(source_ids)
    item_ids = np.asarray(item_ids)
    if raw_source.ndim != 1 or raw_item.shape != raw_source.shape or not len(raw_source):
        raise ValueError("training source/item arrays must be non-empty matching vectors")
    if negative_count < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("negative_count, epochs, and batch_size must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("invalid optimizer hyperparameters")

    source = dense_indices(raw_source, source_ids)
    positive = dense_indices(raw_item, item_ids)
    if np.any(source == 0) or np.any(positive == 0):
        raise ValueError("training vocabulary does not cover the allowed history")
    source_count = len(source_ids) + 1
    item_count = len(item_ids) + 1
    rng = configure_seed(seed)
    model = ImplicitMF(source_count, item_count, embedding_dim)
    optimizer = jt.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    losses: list[float] = []
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(len(source))
        total = 0.0
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            current_source = source[rows]
            current_positive = positive[rows]
            negative = rng.integers(
                1,
                item_count,
                size=(len(rows), int(negative_count)),
                dtype=np.int32,
            )
            collision = negative == current_positive[:, None]
            while collision.any():
                negative[collision] = rng.integers(
                    1, item_count, size=int(collision.sum()), dtype=np.int32
                )
                collision = negative == current_positive[:, None]
            candidates = np.concatenate([current_positive[:, None], negative], axis=1)
            logits = model(
                _batch(current_source, np.int32), _batch(candidates, np.int32)
            )
            target = _batch(np.zeros(len(rows), dtype=np.int32), np.int32)
            loss = nn.cross_entropy_loss(logits, target)
            optimizer.step(loss)
            total += float(np.asarray(loss.data).item()) * len(rows)
        losses.append(total / len(source))
        if verbose:
            print(
                f"implicit_mf epoch={epoch + 1}/{epochs} loss={losses[-1]:.6f}",
                flush=True,
            )
    return model, losses


def predict_scores(
    model: ImplicitMF,
    raw_source: np.ndarray,
    raw_candidates: np.ndarray,
    source_ids: np.ndarray,
    item_ids: np.ndarray,
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    configure_cuda()
    source = dense_indices(np.asarray(raw_source), np.asarray(source_ids))
    candidates = dense_indices(np.asarray(raw_candidates), np.asarray(item_ids))
    if source.ndim != 1 or candidates.ndim != 2 or len(source) != len(candidates):
        raise ValueError("prediction source/candidates have invalid shapes")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    output = np.empty(candidates.shape, dtype=np.float32)
    with jt.no_grad():
        for start in range(0, len(source), int(batch_size)):
            stop = min(len(source), start + int(batch_size))
            score = model(
                _batch(source[start:stop], np.int32),
                _batch(candidates[start:stop], np.int32),
            )
            output[start:stop] = np.asarray(score.data, dtype=np.float32)
    return output


def save_checkpoint(
    path: str | Path,
    model: ImplicitMF,
    source_ids: np.ndarray,
    item_ids: np.ndarray,
) -> Path:
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "kind": np.asarray("d4_implicit_mf"),
        "source_count": np.asarray(model.source_count),
        "item_count": np.asarray(model.item_count),
        "embedding_dim": np.asarray(model.embedding_dim),
        "source_ids": np.asarray(source_ids, dtype=np.uint32),
        "item_ids": np.asarray(item_ids, dtype=np.uint32),
    }
    payload.update(
        {
            f"param__{name}": np.asarray(value.data, dtype=np.float32).copy()
            for name, value in model.state_dict().items()
        }
    )
    np.savez_compressed(path, **payload)
    return path


def load_checkpoint(
    path: str | Path,
) -> tuple[ImplicitMF, np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        if str(archive["kind"].item()) != "d4_implicit_mf":
            raise ValueError("checkpoint kind differs")
        model = ImplicitMF(
            int(archive["source_count"].item()),
            int(archive["item_count"].item()),
            int(archive["embedding_dim"].item()),
        )
        state = {
            name.removeprefix("param__"): jt.array(np.asarray(archive[name], dtype=np.float32))
            for name in archive.files
            if name.startswith("param__")
        }
        if set(state) != set(model.state_dict()):
            raise ValueError("checkpoint parameter names differ")
        model.load_state_dict(state)
        source_ids = np.asarray(archive["source_ids"], dtype=np.uint32).copy()
        item_ids = np.asarray(archive["item_ids"], dtype=np.uint32).copy()
    if model.source_count != len(source_ids) + 1 or model.item_count != len(item_ids) + 1:
        raise ValueError("checkpoint vocabulary size differs")
    return model, source_ids, item_ids
