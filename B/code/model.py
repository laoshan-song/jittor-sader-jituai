#!/usr/bin/env python3
"""Jittor implicit-MF model shared by inference and training."""

from __future__ import annotations

from pathlib import Path

import jittor as jt
import numpy as np
from jittor import nn


def configure_cuda() -> None:
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is unavailable")
    jt.flags.use_cuda = 1


def dense_indices(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    ids = np.asarray(ids)
    output = np.zeros(flat.shape, dtype=np.int32)
    positions = np.searchsorted(ids, flat)
    inside = positions < len(ids)
    matched = np.zeros(flat.shape, dtype=bool)
    matched[inside] = ids[positions[inside]] == flat[inside]
    output[matched] = positions[matched].astype(np.int32) + 1
    return output.reshape(np.asarray(values).shape)


class ImplicitMF(nn.Module):
    def __init__(self, source_count: int, item_count: int, embedding_dim: int):
        super().__init__()
        self.source_count = int(source_count)
        self.item_count = int(item_count)
        self.embedding_dim = int(embedding_dim)
        self.source = nn.Embedding(self.source_count, self.embedding_dim)
        self.item = nn.Embedding(self.item_count, self.embedding_dim)
        self.item_bias = nn.Embedding(self.item_count, 1)

    def execute(self, source: jt.Var, candidates: jt.Var) -> jt.Var:
        source_known = (source > 0).unsqueeze(1).unsqueeze(2)
        candidate_known = (candidates > 0).unsqueeze(2)
        source_vector = self.source(source).unsqueeze(1) * source_known
        item_vector = self.item(candidates) * candidate_known
        score = (source_vector * item_vector).sum(dim=2)
        return score + self.item_bias(candidates).squeeze(2) * (candidates > 0)


def load_checkpoint(path: Path) -> tuple[ImplicitMF, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        kind = str(archive["kind"].item())
        quantized = kind in {f"d4_implicit_mf_q{bits}row_v1" for bits in range(3, 9)}
        if kind != "d4_implicit_mf" and not quantized:
            raise ValueError("checkpoint kind differs")
        if not quantized:
            source_ids = np.asarray(archive["source_ids"], dtype=np.uint32).copy()
            item_ids = np.asarray(archive["item_ids"], dtype=np.uint32).copy()
        else:
            source_ids = np.cumsum(archive["source_ids_delta"], dtype=np.uint64).astype(np.uint32)
            item_ids = np.cumsum(archive["item_ids_delta"], dtype=np.uint64).astype(np.uint32)
        model = ImplicitMF(
            int(archive["source_count"].item()),
            int(archive["item_count"].item()),
            int(archive["embedding_dim"].item()),
        )
        if not quantized:
            state = {
                name.removeprefix("param__"): jt.array(np.asarray(archive[name], dtype=np.float32))
                for name in archive.files
                if name.startswith("param__")
            }
        else:
            names = ("source.weight", "item.weight", "item_bias.weight")
            state = {
                name: jt.array(
                    np.asarray(archive[f"param__{name}_q"], dtype=np.float32)
                    * np.asarray(archive[f"param__{name}_scale"], dtype=np.float32)
                )
                for name in names
            }
    if set(state) != set(model.state_dict()):
        raise ValueError("checkpoint parameter names differ")
    if model.source_count != len(source_ids) + 1 or model.item_count != len(item_ids) + 1:
        raise ValueError("checkpoint vocabulary sizes differ")
    model.load_state_dict(state)
    model.eval()
    return model, source_ids, item_ids


def predict_scores(
    model: ImplicitMF,
    source_ids: np.ndarray,
    item_ids: np.ndarray,
    raw_source: np.ndarray,
    raw_candidates: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    source = dense_indices(raw_source, source_ids)
    candidates = dense_indices(raw_candidates, item_ids)
    output = np.empty(candidates.shape, dtype=np.float32)
    with jt.no_grad():
        for start in range(0, len(source), batch_size):
            stop = min(len(source), start + batch_size)
            score = model(
                jt.array(np.ascontiguousarray(source[start:stop], dtype=np.int32)),
                jt.array(np.ascontiguousarray(candidates[start:stop], dtype=np.int32)),
            )
            output[start:stop] = np.asarray(score.data, dtype=np.float32)
    return output


def bounded_residual(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    normalized = centered / np.maximum(scale, np.float32(1e-6))
    return np.tanh(normalized * np.float32(0.5)).astype(np.float32)
