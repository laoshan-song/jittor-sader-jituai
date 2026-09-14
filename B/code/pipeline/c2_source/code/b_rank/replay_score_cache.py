#!/usr/bin/env python3
"""Persistent, hash-checked D4 replay scores for fast reranker experiments."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np


CACHE_KIND = "d4_multimodel_replay_score_cache_v1"


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


def _save_array(root: Path, name: str, values: np.ndarray) -> dict[str, Any]:
    path = root / f"{name}.npy"
    np.save(path, np.asarray(values), allow_pickle=False)
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def save(
    path: Path,
    scored: dict[
        tuple[str, str],
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[str, np.ndarray],
            np.ndarray,
        ],
    ],
    component_names: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write a new immutable cache directory."""
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing replay cache reuse: {path}")
    path.mkdir(parents=True)
    entries: dict[str, Any] = {}
    for (strategy, split), (scores, labels, seen, segments, static) in scored.items():
        if static is None:
            raise ValueError("replay cache requires static candidate features")
        prefix = f"{strategy}__{split}"
        arrays = {
            "scores": _save_array(path, f"{prefix}__scores", scores),
            "labels": _save_array(path, f"{prefix}__labels", labels),
            "seen": _save_array(path, f"{prefix}__seen", seen),
            "static": _save_array(path, f"{prefix}__static", static),
        }
        segment_records = {
            name: _save_array(path, f"{prefix}__segment__{name}", values)
            for name, values in sorted(segments.items())
        }
        entries[prefix] = {"arrays": arrays, "segments": segment_records}
    manifest = {
        "kind": CACHE_KIND,
        "component_names": list(component_names),
        "entries": entries,
        "metadata": metadata,
    }
    _atomic_json(path / "manifest.json", manifest)
    return manifest


def _load_array(
    root: Path, record: dict[str, Any], *, verify: bool
) -> np.ndarray:
    path = root / str(record["file"])
    if verify and _sha256(path) != record["sha256"]:
        raise ValueError(f"replay cache hash differs: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(values.shape) != record["shape"] or str(values.dtype) != record["dtype"]:
        raise ValueError(f"replay cache array contract differs: {path}")
    return values


def load(
    paths: list[Path], *, verify: bool = True
) -> tuple[
    dict[
        tuple[str, str],
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[str, np.ndarray],
            np.ndarray,
        ],
    ],
    list[str],
    list[dict[str, Any]],
]:
    """Load one or more disjoint cache shards as read-only memmaps."""
    scored = {}
    component_names = None
    manifests = []
    for root in map(Path.resolve, paths):
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != CACHE_KIND:
            raise ValueError(f"replay cache kind differs: {manifest_path}")
        names = list(manifest["component_names"])
        if component_names is None:
            component_names = names
        elif names != component_names:
            raise ValueError("replay cache component order differs")
        for prefix, entry in manifest["entries"].items():
            strategy, split = prefix.split("__", 1)
            key = (strategy, split)
            if key in scored:
                raise ValueError(f"duplicate replay cache entry: {key}")
            arrays = entry["arrays"]
            scores = _load_array(root, arrays["scores"], verify=verify)
            labels = _load_array(root, arrays["labels"], verify=verify)
            seen = _load_array(root, arrays["seen"], verify=verify)
            static = _load_array(root, arrays["static"], verify=verify)
            segments = {
                name: _load_array(root, record, verify=verify)
                for name, record in entry["segments"].items()
            }
            rows, candidates = labels.shape[0], seen.shape[1]
            if (
                scores.shape != (len(names), rows, candidates)
                or seen.shape != (rows, candidates)
                or static.shape[:2] != (rows, candidates)
                or any(values.shape != (rows,) for values in segments.values())
            ):
                raise ValueError(f"replay cache shapes disagree: {key}")
            scored[key] = (scores, labels, seen, segments, static)
        manifests.append(manifest)
    if component_names is None:
        raise ValueError("no replay cache paths were provided")
    return scored, component_names, manifests


def _self_check() -> None:
    import tempfile

    rng = np.random.default_rng(20260812)
    values = (
        rng.normal(size=(2, 5, 3)).astype(np.float32),
        np.arange(5, dtype=np.int64) % 3,
        rng.random((5, 3)) > 0.5,
        {"overall": np.ones(5, dtype=bool)},
        rng.normal(size=(5, 3, 4)).astype(np.float32),
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "cache"
        save(root, {("history", "validation"): values}, ["a", "b"], {})
        loaded, names, _ = load([root])
        if names != ["a", "b"]:
            raise AssertionError("component names changed")
        for before, after in zip(values[:3], loaded[("history", "validation")][:3]):
            if not np.array_equal(before, after):
                raise AssertionError("cache round trip changed an array")


if __name__ == "__main__":
    _self_check()
