#!/usr/bin/env python3
"""Pure NumPy deployment helpers for the hierarchy residual ranker."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np


WIDTH = 100


def qnorm(values: np.ndarray) -> np.ndarray:
    """Normalize every candidate row without changing candidate semantics."""
    values = np.asarray(values, dtype=np.float32)
    centered = values - values.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    return (centered / np.maximum(scale, np.float32(1e-6))).astype(np.float32)


def strict_rank_scores(values: np.ndarray) -> np.ndarray:
    """Map each row to the exact 0..1 rank grid with stable column tie breaks."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("values must have shape (rows, candidates), candidates >= 2")
    if not np.isfinite(values).all():
        raise ValueError("rank input contains non-finite values")
    order = np.argsort(values, axis=1, kind="stable")
    result = np.empty(values.shape, dtype=np.float32)
    grid = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float32)
    rows = np.arange(len(values))[:, None]
    result[rows, order] = grid[None, :]
    return result


def duplicate_mask(candidates: np.ndarray) -> np.ndarray:
    """Return rows containing at least one repeated candidate ID."""
    candidates = np.asarray(candidates)
    if candidates.ndim != 2:
        raise ValueError("candidates must be two-dimensional")
    return np.any(
        np.diff(np.sort(candidates, axis=1, kind="stable"), axis=1) == 0,
        axis=1,
    )


def remap_probability_slots(
    base_probability: np.ndarray,
    candidate_score: np.ndarray,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Assign the baseline probability multiset according to a new ordering.

    This changes ranking only. It preserves every row's exact probability slots,
    calibration, and sum. Rows selected by ``fallback`` remain byte-value
    equivalent to the baseline after formatting.
    """
    base_probability = np.asarray(base_probability)
    candidate_score = np.asarray(candidate_score)
    if base_probability.shape != candidate_score.shape or base_probability.ndim != 2:
        raise ValueError("base_probability and candidate_score must have equal 2-D shapes")
    if not np.isfinite(base_probability).all() or not np.isfinite(candidate_score).all():
        raise ValueError("probabilities and scores must be finite")
    if np.any(base_probability < 0.0) or np.any(base_probability > 1.0):
        raise ValueError("baseline probabilities are outside [0, 1]")
    probability_order = np.argsort(base_probability, axis=1, kind="stable")
    score_order = np.argsort(candidate_score, axis=1, kind="stable")
    sorted_probability = np.take_along_axis(base_probability, probability_order, axis=1)
    output = np.empty_like(base_probability)
    rows = np.arange(len(base_probability))[:, None]
    output[rows, score_order] = sorted_probability
    if fallback is not None:
        fallback = np.asarray(fallback, dtype=bool)
        if fallback.shape != (len(output),):
            raise ValueError("fallback must have one boolean per row")
        output[fallback] = base_probability[fallback]
    return output


def pair_keys(source: np.ndarray, destination: np.ndarray, base: int) -> np.ndarray:
    """Encode source/destination pairs as sorted-searchable signed int64 keys."""
    source = np.asarray(source, dtype=np.int64)
    destination = np.asarray(destination, dtype=np.int64)
    if base < 1 or np.any(destination < 0) or np.any(destination >= base):
        raise ValueError("destination IDs must lie in [0, base)")
    if np.any(source < 0):
        raise ValueError("source IDs must be non-negative")
    maximum = int(source.max(initial=0)) * int(base) + int(destination.max(initial=0))
    if maximum > np.iinfo(np.int64).max:
        raise OverflowError("pair key exceeds signed int64")
    return source * np.int64(base) + destination


class PairSeenIndex:
    """Compact exact membership index over historical source/destination pairs."""

    def __init__(self, source: np.ndarray, destination: np.ndarray, base: int) -> None:
        self.base = int(base)
        self.keys = np.unique(pair_keys(source, destination, self.base))

    def contains(self, source: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        source = np.asarray(source)
        candidates = np.asarray(candidates)
        if source.shape != (len(candidates),) or candidates.ndim != 2:
            raise ValueError("source/candidate shapes differ")
        query = pair_keys(source[:, None], candidates, self.base)
        flat = query.reshape(-1)
        positions = np.searchsorted(self.keys, flat)
        valid = positions < len(self.keys)
        output = np.zeros(len(flat), dtype=bool)
        selected = np.flatnonzero(valid)
        output[selected] = self.keys[positions[selected]] == flat[selected]
        return output.reshape(candidates.shape)


def parse_probability_lines(
    lines: list[str], width: int = WIDTH, member: str = "probability.csv"
) -> np.ndarray:
    """Parse a bounded block of headerless floating-point CSV rows."""
    if not lines:
        return np.empty((0, width), dtype=np.float64)
    values = np.fromstring(",".join(lines), dtype=np.float64, sep=",")
    expected = len(lines) * width
    if values.size != expected:
        raise ValueError(f"malformed width or value in {member}")
    values = values.reshape(len(lines), width)
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"invalid probability in {member}")
    return values


def iter_probability_chunks(
    archive_path: str | Path,
    member: str,
    *,
    chunk_rows: int,
    width: int = WIDTH,
) -> Iterator[np.ndarray]:
    """Stream one headerless probability member from a ZIP archive."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    with zipfile.ZipFile(archive_path) as archive:
        if member not in archive.namelist():
            raise ValueError(f"missing {member} in {archive_path}")
        with archive.open(member, "r") as raw:
            with io.TextIOWrapper(raw, encoding="ascii", newline="") as text:
                lines: list[str] = []
                for line in text:
                    line = line.strip()
                    if not line:
                        continue
                    lines.append(line)
                    if len(lines) == chunk_rows:
                        yield parse_probability_lines(lines, width, member)
                        lines.clear()
                if lines:
                    yield parse_probability_lines(lines, width, member)


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def member_sha256(archive: zipfile.ZipFile, member: str, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()
