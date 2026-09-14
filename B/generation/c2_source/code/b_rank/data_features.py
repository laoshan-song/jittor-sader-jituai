#!/usr/bin/env python3
"""Streaming NumPy data layer for the B-rank Dataset3/Dataset4 archives.

The module deliberately contains no learned code and no Jittor import.  It
keeps raw train columns and derived statistics in ``.npy`` memmaps, while test
rows are always read from the ZIP in bounded chunks.  A :class:`FeatureStore`
is frozen at one strict temporal boundary: every edge used by its node and
pair statistics satisfies ``edge_time < cutoff``.

Typical use::

    cache = BDataCache.build_or_open("data_B.zip", "dataset4", cache_dir)
    groups = build_split1_groups(cache, seed=20260810)
    train_store = FeatureStore.build_or_open(cache, groups["train"].cutoff)
    for batch in groups["train"].iter_feature_batches(train_store):
        features, labels = batch.features, batch.labels

The generated validation groups are a causal replay proxy.  They contain
split1 positives mixed with historical negatives; ``history_cold`` can also
draw unseen IDs from the historical destination-ID range.  They are not a
claim about hidden B-rank labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


CACHE_VERSION = 1
UINT32_MAX = np.iinfo(np.uint32).max
SECONDS_PER_DAY = 86_400.0
DEFAULT_SEED = 20_260_810
DEFAULT_TRAIN_GROUPS = 100_000
DEFAULT_VALID_GROUPS = 30_000
DEFAULT_CONFIRM_GROUPS = 30_000
GROUP_NAMES = ("train", "valid", "confirm")
NEGATIVE_STRATEGIES = frozenset(
    ("history", "history_cold", "test_pool", "popularity_hard")
)
MAX_COLD_POOL_IDS = 10_000_000

TRAIN_COLUMNS = ("src", "dst", "time", "split")
TEST_COLUMNS = ("src", "time") + tuple(f"c{i}" for i in range(1, 101))
SCENES = frozenset(("dataset3", "dataset4"))

# Candidate-level values only.  Values that are constant within a query are
# intentionally omitted because a candidate ranker cannot use them to change a
# row ordering.
FEATURE_NAMES = (
    "pair_log_count",
    "pair_recency",
    "pair_seen",
    "source_log_count",
    "source_recency",
    "source_seen",
    "destination_log_count",
    "destination_recency",
    "destination_seen",
    "pair_source_share",
    "pair_destination_share",
)


class DataContractError(ValueError):
    """Raised when an archive or cache violates the B-rank data contract."""


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    """Return the SHA-256 of a file without loading it into memory."""
    path = Path(path)
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


file_sha256 = sha256_file


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data using a stable, whitespace-free encoding."""
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


json_sha256 = sha256_json


def sha256_array(values: np.ndarray, chunk_bytes: int = 8 << 20) -> str:
    """Hash dtype, shape and values of an array without a large contiguous copy."""
    values = np.asarray(values)
    if values.dtype.hasobject:
        raise TypeError("object arrays do not have a stable binary hash")
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": values.dtype.str, "shape": list(values.shape)},
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(header)
    digest.update(b"\n")
    if values.flags.c_contiguous:
        raw = values.view(np.uint8).reshape(-1)
        for start in range(0, raw.size, chunk_bytes):
            digest.update(memoryview(raw[start : start + chunk_bytes]))
    else:
        # ``external_loop`` bounds temporary memory for non-contiguous views.
        iterator = np.nditer(
            values,
            flags=("external_loop", "buffered"),
            order="C",
            buffersize=max(1, chunk_bytes // max(1, values.dtype.itemsize)),
        )
        for block in iterator:
            digest.update(np.ascontiguousarray(block).view(np.uint8).tobytes())
    return digest.hexdigest()


array_sha256 = sha256_array


def _scene_name(scene: str) -> str:
    scene = str(scene)
    if scene not in SCENES:
        raise ValueError(f"scene must be one of {sorted(SCENES)}, got {scene!r}")
    return scene


def _member(scene: str, kind: str) -> str:
    scene = _scene_name(scene)
    if kind not in {"train", "test"}:
        raise ValueError("kind must be 'train' or 'test'")
    return f"{scene}/{kind}.csv"


def _header_for(kind: str) -> tuple[str, ...]:
    if kind == "train":
        return TRAIN_COLUMNS
    if kind == "test":
        return TEST_COLUMNS
    raise ValueError("kind must be 'train' or 'test'")


def _read_header(archive: zipfile.ZipFile, member: str) -> tuple[str, ...]:
    with archive.open(member, "r") as raw:
        line = raw.readline().decode("utf-8-sig").strip("\r\n")
    if not line:
        raise DataContractError(f"{member} has no CSV header")
    return tuple(next(csv.reader((line,))))


def inspect_archive(
    data_zip: str | Path,
    scene: str,
    *,
    count_rows: bool = False,
) -> dict[str, Any]:
    """Validate B archive members and optionally count decompressed CSV rows.

    ``count_rows=False`` only reads both headers.  Counting is intentionally
    opt-in because Dataset4 is large and its test set is never cached.
    """
    data_zip = Path(data_zip)
    scene = _scene_name(scene)
    if not data_zip.is_file():
        raise FileNotFoundError(data_zip)
    result: dict[str, Any] = {
        "archive": str(data_zip.resolve()),
        "scene": scene,
        "members": {},
    }
    with zipfile.ZipFile(data_zip) as archive:
        names = set(archive.namelist())
        for kind in ("train", "test"):
            member = _member(scene, kind)
            if member not in names:
                raise DataContractError(f"archive is missing {member}")
            header = _read_header(archive, member)
            expected = _header_for(kind)
            if header != expected:
                raise DataContractError(
                    f"{member} header is {header!r}; expected {expected!r}"
                )
            payload: dict[str, Any] = {"columns": list(header)}
            if count_rows:
                with archive.open(member, "r") as raw:
                    # The first line is the header.  Official rows are unquoted
                    # integer records, so a newline count is exact.
                    next(raw, None)
                    payload["rows"] = sum(1 for line in raw if line.strip())
            result["members"][kind] = payload
    return result


def _parse_integer_lines(
    lines: list[str], width: int, member: str
) -> np.ndarray:
    """Parse a bounded set of simple official integer CSV lines."""
    if not lines:
        return np.empty((0, width), dtype=np.uint32)
    payload = ",".join(lines)
    values = np.fromstring(payload, dtype=np.uint64, sep=",")
    expected = len(lines) * width
    if values.size != expected:
        # This fallback gives an actionable error for malformed or quoted CSV,
        # rather than silently accepting a prefix as ``fromstring`` can do.
        try:
            parsed = [
                [int(value) for value in row]
                for row in csv.reader(lines)
                if row
            ]
        except ValueError as exc:
            raise DataContractError(f"non-integer row in {member}") from exc
        if len(parsed) != len(lines) or any(len(row) != width for row in parsed):
            raise DataContractError(f"malformed CSV width in {member}")
        values = np.asarray(parsed, dtype=np.uint64).reshape(-1)
    if values.size != expected:
        raise DataContractError(f"malformed CSV row in {member}")
    if values.size and int(values.max()) > UINT32_MAX:
        raise DataContractError(f"{member} contains a value outside uint32")
    return values.astype(np.uint32, copy=False).reshape((-1, width))


def _iter_member_chunks(
    data_zip: str | Path,
    member: str,
    header: tuple[str, ...],
    *,
    chunk_rows: int,
) -> Iterator[np.ndarray]:
    """Yield one integer CSV member in bounded NumPy arrays."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    with zipfile.ZipFile(data_zip) as archive:
        with archive.open(member, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            actual = tuple(next(csv.reader((text.readline(),))))
            if actual != header:
                raise DataContractError(
                    f"{member} header is {actual!r}; expected {header!r}"
                )
            lines: list[str] = []
            for line in text:
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
                if len(lines) == chunk_rows:
                    yield _parse_integer_lines(lines, len(header), member)
                    lines.clear()
            if lines:
                yield _parse_integer_lines(lines, len(header), member)


def iter_test_chunks(
    data_zip: str | Path,
    scene: str,
    *,
    chunk_rows: int = 8192,
) -> Iterator["TestChunk"]:
    """Stream test candidates; no full candidate or score matrix is retained."""
    scene = _scene_name(scene)
    row_start = 0
    for values in _iter_member_chunks(
        data_zip,
        _member(scene, "test"),
        TEST_COLUMNS,
        chunk_rows=chunk_rows,
    ):
        yield TestChunk(
            row_start=row_start,
            src=values[:, 0],
            time=values[:, 1],
            candidates=values[:, 2:],
        )
        row_start += len(values)


@dataclass(frozen=True)
class TestChunk:
    """A bounded test slice in original CSV order."""

    row_start: int
    src: np.ndarray
    time: np.ndarray
    candidates: np.ndarray

    @property
    def row_stop(self) -> int:
        return self.row_start + len(self.src)


@dataclass(frozen=True)
class TrainChunk:
    """A bounded slice of cached train columns in original CSV order."""

    row_start: int
    src: np.ndarray
    dst: np.ndarray
    time: np.ndarray
    split: np.ndarray

    @property
    def row_stop(self) -> int:
        return self.row_start + len(self.src)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, values: np.ndarray, *, copy_rows: int = 1_000_000) -> None:
    """Write an array through a memmap, then publish it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    values = np.asarray(values)
    mapped = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=values.dtype, shape=values.shape
    )
    try:
        if values.ndim == 0:
            mapped[...] = values
        elif len(values) == 0:
            pass
        else:
            for start in range(0, len(values), copy_rows):
                mapped[start : start + copy_rows] = values[start : start + copy_rows]
        mapped.flush()
        del mapped
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _open_memmap(path: Path) -> np.memmap:
    if not path.is_file():
        raise DataContractError(f"cache is missing {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _cache_root(cache_dir: str | Path, scene: str, archive_hash: str) -> Path:
    return Path(cache_dir) / "b_rank_data" / f"{scene}-{archive_hash[:20]}"


class BDataCache:
    """Immutable memmap cache of one official B train CSV.

    The raw CSV is parsed twice only when a cache does not already exist: once
    to size the memmaps and once to populate them.  Test rows are deliberately
    excluded from this cache to keep Dataset4 bounded.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise DataContractError(f"not a B data cache: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("cache_version") != CACHE_VERSION:
            raise DataContractError(
                f"unsupported cache version in {metadata_path}: "
                f"{self.metadata.get('cache_version')!r}"
            )
        self.scene = _scene_name(self.metadata.get("scene", ""))
        self._arrays: dict[str, np.memmap] = {}

    @classmethod
    def build_or_open(
        cls,
        data_zip: str | Path,
        scene: str,
        cache_dir: str | Path,
        *,
        chunk_rows: int = 250_000,
        verify_hash: bool = True,
    ) -> "BDataCache":
        """Open an immutable matching cache or build it from the official ZIP."""
        data_zip = Path(data_zip)
        scene = _scene_name(scene)
        if not data_zip.is_file():
            raise FileNotFoundError(data_zip)
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        # A content-addressed directory makes stale-cache reuse impossible even
        # when a differently named official archive is supplied.
        archive_hash = sha256_file(data_zip) if verify_hash else _quick_file_key(data_zip)
        root = _cache_root(cache_dir, scene, archive_hash)
        if root.is_dir():
            cache = cls(root)
            expected = archive_hash
            if cache.metadata.get("archive_hash") != expected:
                raise DataContractError(f"cache hash mismatch in {root}")
            if cache.metadata.get("scene") != scene:
                raise DataContractError(f"cache scene mismatch in {root}")
            cache._validate_files()
            return cache

        inspect_archive(data_zip, scene, count_rows=False)
        root.parent.mkdir(parents=True, exist_ok=True)
        stage = root.parent / f".{root.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
        stage.mkdir(parents=False, exist_ok=False)
        try:
            rows = _count_member_rows(data_zip, _member(scene, "train"))
            metadata = _populate_train_cache(
                data_zip=data_zip,
                scene=scene,
                destination=stage,
                rows=rows,
                chunk_rows=chunk_rows,
                archive_hash=archive_hash,
            )
            _atomic_json(stage / "metadata.json", metadata)
            try:
                os.replace(stage, root)
            except FileExistsError:
                # Another process may have finished the same content-addressed
                # cache while this process was parsing.  Never overwrite it.
                existing = cls(root)
                if existing.metadata.get("archive_hash") != archive_hash:
                    raise DataContractError(f"conflicting cache already exists: {root}")
                existing._validate_files()
                return existing
            return cls(root)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    @property
    def archive_hash(self) -> str:
        return str(self.metadata["archive_hash"])

    @property
    def train_rows(self) -> int:
        return int(self.metadata["train_rows"])

    @property
    def source(self) -> np.memmap:
        return self._array("src")

    @property
    def destination(self) -> np.memmap:
        return self._array("dst")

    @property
    def time(self) -> np.memmap:
        return self._array("time")

    @property
    def split(self) -> np.memmap:
        return self._array("split")

    # Compact aliases make the public data API read like the raw CSV.
    @property
    def src(self) -> np.memmap:
        return self.source

    @property
    def dst(self) -> np.memmap:
        return self.destination

    def _array(self, name: str) -> np.memmap:
        if name not in self._arrays:
            values = _open_memmap(self.root / f"{name}.npy")
            if values.shape != (self.train_rows,) or values.dtype != np.dtype(np.uint32):
                raise DataContractError(
                    f"invalid {name}.npy shape/dtype: {values.shape}, {values.dtype}"
                )
            self._arrays[name] = values
        return self._arrays[name]

    def _validate_files(self) -> None:
        for name in ("src", "dst", "time", "split"):
            self._array(name)

    def iter_train_chunks(
        self,
        *,
        chunk_rows: int = 250_000,
        start: int = 0,
        stop: int | None = None,
    ) -> Iterator[TrainChunk]:
        """Yield cached train columns in bounded slices without copying all rows."""
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        start, stop = _slice_bounds(start, stop, self.train_rows)
        for offset in range(start, stop, chunk_rows):
            end = min(stop, offset + chunk_rows)
            yield TrainChunk(
                row_start=offset,
                src=self.src[offset:end],
                dst=self.dst[offset:end],
                time=self.time[offset:end],
                split=self.split[offset:end],
            )

    def history_end(self, cutoff: int) -> int:
        """Return rows satisfying ``time < cutoff``; reject unsafe unsorted data."""
        cutoff = _as_time(cutoff)
        if not self.metadata.get("time_non_decreasing", False):
            raise DataContractError(
                "train timestamps are not monotone; refusing a binary-search history cut"
            )
        end = int(np.searchsorted(self.time, cutoff, side="left"))
        # These O(1) boundary checks make a corrupt cache fail closed.
        if end and int(self.time[end - 1]) >= cutoff:
            raise DataContractError("history cache violates strict < cutoff")
        if end < self.train_rows and int(self.time[end]) < cutoff:
            raise DataContractError("history cache violates strict < cutoff")
        return end

    def history_dst_counts(self, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
        """Return sorted historical destinations and counts from ``time < cutoff``.

        This is the only source for the causal/default, optional history-cold,
        and popularity-hard replay samplers.  It deliberately excludes split1
        and every test row.
        """
        cutoff = _as_time(cutoff)
        pool_dir = self.root / "pools"
        ids_path = pool_dir / f"history_dst_ids_lt_{cutoff}.npy"
        counts_path = pool_dir / f"history_dst_counts_lt_{cutoff}.npy"
        if ids_path.is_file() and counts_path.is_file():
            ids, counts = _open_memmap(ids_path), _open_memmap(counts_path)
            if (
                ids.dtype != np.dtype(np.uint32)
                or counts.dtype != np.dtype(np.uint32)
                or ids.ndim != 1
                or counts.shape != ids.shape
            ):
                raise DataContractError(f"invalid historical negative-pool cache: {pool_dir}")
            return ids, counts
        end = self.history_end(cutoff)
        ids, counts = np.unique(self.dst[:end], return_counts=True)
        ids = ids.astype(np.uint32, copy=False)
        counts = counts.astype(np.uint32, copy=False)
        if len(ids) < 2:
            raise DataContractError("historical destination pool has fewer than two ids")
        pool_dir.mkdir(parents=True, exist_ok=True)
        _atomic_npy(ids_path, ids)
        _atomic_npy(counts_path, counts)
        return _open_memmap(ids_path), _open_memmap(counts_path)

    def history_dst_pool(self, cutoff: int) -> np.ndarray:
        """Return sorted unique historical destinations from edges ``time < cutoff``."""
        return self.history_dst_counts(cutoff)[0]

    def test_candidate_counts(
        self,
        *,
        chunk_rows: int = 8192,
        max_dense_ids: int = 10_000_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stream the unlabeled official test pool into compact id/count memmaps.

        The helper never stores the test matrix.  It makes two streaming passes:
        one to find the largest candidate id and one to aggregate a bounded
        uint32 frequency table.  These counts are for distribution diagnostics
        only, never labels or a replacement for causal validation.
        """
        if chunk_rows < 1 or max_dense_ids < 1:
            raise ValueError("chunk_rows and max_dense_ids must be positive")
        pool_dir = self.root / "pools"
        ids_path = pool_dir / "test_candidate_ids.npy"
        counts_path = pool_dir / "test_candidate_counts.npy"
        metadata_path = pool_dir / "test_candidate_metadata.json"
        if ids_path.is_file() and counts_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            ids, counts = _open_memmap(ids_path), _open_memmap(counts_path)
            if (
                metadata.get("archive_hash") != self.archive_hash
                or ids.dtype != np.dtype(np.uint32)
                or counts.dtype != np.dtype(np.uint32)
                or ids.ndim != 1
                or ids.shape != counts.shape
            ):
                raise DataContractError(f"invalid test-pool cache: {pool_dir}")
            return ids, counts

        maximum = 0
        rows = 0
        for chunk in self.iter_test_chunks(chunk_rows=chunk_rows):
            if len(chunk.candidates):
                maximum = max(maximum, int(chunk.candidates.max()))
            rows += len(chunk.src)
        if maximum > max_dense_ids:
            raise DataContractError(
                f"test candidate id {maximum} exceeds max_dense_ids={max_dense_ids}; "
                "refusing an unbounded dense frequency table"
            )
        frequencies = np.zeros(maximum + 1, dtype=np.uint32)
        for chunk in self.iter_test_chunks(chunk_rows=chunk_rows):
            ids, count = np.unique(chunk.candidates.reshape(-1), return_counts=True)
            frequencies[ids] += count.astype(np.uint32, copy=False)
        ids = np.flatnonzero(frequencies).astype(np.uint32, copy=False)
        counts = frequencies[ids]
        if len(ids) < 2:
            raise DataContractError("unlabeled test candidate pool has fewer than two ids")
        pool_dir.mkdir(parents=True, exist_ok=True)
        _atomic_npy(ids_path, ids)
        _atomic_npy(counts_path, counts)
        _atomic_json(
            metadata_path,
            {
                "archive_hash": self.archive_hash,
                "scene": self.scene,
                "rows": rows,
                "candidate_width": len(TEST_COLUMNS) - 2,
                "max_candidate_id": maximum,
                "purpose": "unlabeled diagnostic candidate distribution only",
            },
        )
        return _open_memmap(ids_path), _open_memmap(counts_path)

    def test_candidate_pool(
        self, *, chunk_rows: int = 8192, max_dense_ids: int = 10_000_000
    ) -> np.ndarray:
        """Return the compact, unlabeled official test candidate-id universe."""
        return self.test_candidate_counts(
            chunk_rows=chunk_rows, max_dense_ids=max_dense_ids
        )[0]

    def split1_bounds(self) -> tuple[int, int]:
        """Return the contiguous split1 row range guaranteed by cache validation."""
        if not self.metadata.get("split_non_decreasing", False):
            raise DataContractError("split values are not monotone")
        bounds = self.metadata.get("split_bounds", {}).get("1")
        if not bounds:
            raise DataContractError("cache has no split1 rows")
        start, stop = int(bounds[0]), int(bounds[1])
        if not (0 <= start < stop <= self.train_rows):
            raise DataContractError("invalid split1 bounds in cache metadata")
        return start, stop

    def split1_plan(
        self, ratios: Sequence[float] = (0.60, 0.20, 0.20)
    ) -> "Split1Plan":
        """Partition split1 at timestamp boundaries, never in the middle of a tie."""
        return Split1Plan.from_cache(self, ratios=ratios)

    def feature_store(self, cutoff: int) -> "FeatureStore":
        return FeatureStore.build_or_open(self, cutoff)

    def iter_test_chunks(self, *, chunk_rows: int = 8192) -> Iterator[TestChunk]:
        """Stream official test rows using the archive path recorded in metadata."""
        return iter_test_chunks(self.metadata["archive_path"], self.scene, chunk_rows=chunk_rows)


def _quick_file_key(path: Path) -> str:
    """Non-cryptographic opt-out key for development-only cache reuse."""
    stat = path.stat()
    return sha256_json(
        {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    )


def _count_member_rows(data_zip: Path, member: str) -> int:
    with zipfile.ZipFile(data_zip) as archive:
        with archive.open(member, "r") as raw:
            header = raw.readline()
            if not header:
                raise DataContractError(f"{member} has no CSV header")
            return sum(1 for line in raw if line.strip())


def _populate_train_cache(
    *,
    data_zip: Path,
    scene: str,
    destination: Path,
    rows: int,
    chunk_rows: int,
    archive_hash: str,
) -> dict[str, Any]:
    mapped = {
        name: np.lib.format.open_memmap(
            destination / f"{name}.npy", mode="w+", dtype=np.uint32, shape=(rows,)
        )
        for name in ("src", "dst", "time", "split")
    }
    offset = 0
    previous_time: int | None = None
    previous_split: int | None = None
    time_non_decreasing = True
    split_non_decreasing = True
    split_counts = np.zeros(2, dtype=np.int64)
    split_min = np.full(2, UINT32_MAX, dtype=np.uint64)
    split_max = np.zeros(2, dtype=np.uint64)
    time_min = UINT32_MAX
    time_max = 0
    try:
        for values in _iter_member_chunks(
            data_zip,
            _member(scene, "train"),
            TRAIN_COLUMNS,
            chunk_rows=chunk_rows,
        ):
            if offset + len(values) > rows:
                raise DataContractError("train row count changed while caching")
            src, dst, time, split = (values[:, column] for column in range(4))
            if np.any((split != 0) & (split != 1)):
                raise DataContractError("train split values must be exactly 0 or 1")
            if len(time):
                if previous_time is not None and int(time[0]) < previous_time:
                    time_non_decreasing = False
                if len(time) > 1 and np.any(time[1:] < time[:-1]):
                    time_non_decreasing = False
                if previous_split is not None and int(split[0]) < previous_split:
                    split_non_decreasing = False
                if len(split) > 1 and np.any(split[1:] < split[:-1]):
                    split_non_decreasing = False
                previous_time = int(time[-1])
                previous_split = int(split[-1])
                time_min = min(time_min, int(time.min()))
                time_max = max(time_max, int(time.max()))
                for value in (0, 1):
                    mask = split == value
                    if np.any(mask):
                        split_counts[value] += int(mask.sum())
                        split_min[value] = min(split_min[value], int(time[mask].min()))
                        split_max[value] = max(split_max[value], int(time[mask].max()))
            mapped["src"][offset : offset + len(values)] = src
            mapped["dst"][offset : offset + len(values)] = dst
            mapped["time"][offset : offset + len(values)] = time
            mapped["split"][offset : offset + len(values)] = split
            offset += len(values)
        if offset != rows:
            raise DataContractError("train row count changed while caching")
        if not split_counts[0] or not split_counts[1]:
            raise DataContractError("both split0 and split1 must have at least one row")
        if split_max[0] >= split_min[1]:
            raise DataContractError(
                "split0 must strictly precede split1 for the B temporal protocol"
            )
        split_bounds = _find_split_bounds(mapped["split"], rows)
        if split_bounds.get("0") != [0, int(split_counts[0])] or split_bounds.get("1") != [
            int(split_counts[0]), rows
        ]:
            split_non_decreasing = False
        return {
            "cache_version": CACHE_VERSION,
            "archive_path": str(data_zip.resolve()),
            "archive_hash": archive_hash,
            "archive_size": int(data_zip.stat().st_size),
            "scene": scene,
            "train_rows": int(rows),
            "columns": list(TRAIN_COLUMNS),
            "time_non_decreasing": bool(time_non_decreasing),
            "split_non_decreasing": bool(split_non_decreasing),
            "time_range": [int(time_min), int(time_max)],
            "split_counts": {str(index): int(value) for index, value in enumerate(split_counts)},
            "split_time_ranges": {
                str(index): [int(split_min[index]), int(split_max[index])]
                for index in (0, 1)
            },
            "split_bounds": split_bounds,
        }
    finally:
        for values in mapped.values():
            values.flush()
        mapped.clear()


def _find_split_bounds(split: np.ndarray, rows: int) -> dict[str, list[int]]:
    """Return contiguous bounds only; a non-contiguous split has no safe bound."""
    values = np.asarray(split)
    first_one = int(np.searchsorted(values, 1, side="left"))
    if first_one == rows or np.any(values[:first_one] != 0) or np.any(values[first_one:] != 1):
        return {}
    return {"0": [0, first_one], "1": [first_one, rows]}


def _slice_bounds(start: int, stop: int | None, length: int) -> tuple[int, int]:
    start = int(start)
    stop = length if stop is None else int(stop)
    if not (0 <= start <= stop <= length):
        raise ValueError(f"invalid slice [{start}:{stop}] for length {length}")
    return start, stop


def _as_time(value: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("cutoff must be an integer timestamp") from exc
    if not 0 <= value <= UINT32_MAX:
        raise ValueError("timestamp is outside uint32")
    return value


def _aggregate_history(keys: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate sorted-key counts and last timestamps with bounded extra state."""
    keys = np.asarray(keys)
    times = np.asarray(times)
    if keys.ndim != 1 or times.ndim != 1 or len(keys) != len(times):
        raise ValueError("keys and times must be matching one-dimensional arrays")
    if not len(keys):
        return (
            np.empty(0, dtype=keys.dtype),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
        )
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.empty(len(sorted_keys), dtype=bool)
    starts[0] = True
    starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
    positions = np.flatnonzero(starts)
    counts = np.diff(np.append(positions, len(sorted_keys))).astype(np.uint32, copy=False)
    last = np.maximum.reduceat(times[order], positions).astype(np.uint32, copy=False)
    return sorted_keys[positions].copy(), counts, last


class FeatureStore:
    """Compact source, destination and pair statistics frozen at ``< cutoff``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise DataContractError(f"not a feature store: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("cache_version") != CACHE_VERSION:
            raise DataContractError(f"unsupported feature-store version in {metadata_path}")
        self.cutoff = _as_time(self.metadata["cutoff"])
        self._arrays: dict[str, np.memmap] = {}

    @classmethod
    def build_or_open(cls, cache: BDataCache, cutoff: int) -> "FeatureStore":
        """Build/read compact stats using exactly the rows with ``time < cutoff``."""
        cutoff = _as_time(cutoff)
        end = cache.history_end(cutoff)
        root = cache.root / "stats" / f"history_lt_{cutoff}"
        if root.is_dir():
            store = cls(root)
            if (
                store.metadata.get("archive_hash") != cache.archive_hash
                or store.metadata.get("scene") != cache.scene
                or int(store.metadata.get("history_rows", -1)) != end
            ):
                raise DataContractError(f"feature store metadata mismatch: {root}")
            store._validate_files()
            return store

        root.parent.mkdir(parents=True, exist_ok=True)
        stage = root.parent / f".{root.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
        stage.mkdir(parents=False, exist_ok=False)
        try:
            # These are views of train memmaps.  The only large new vector is
            # pair_keys; Dataset4 never creates an ID-sized dense table.
            src = cache.src[:end]
            dst = cache.dst[:end]
            time = cache.time[:end]
            src_ids, src_count, src_last = _aggregate_history(src, time)
            dst_ids, dst_count, dst_last = _aggregate_history(dst, time)
            pair_keys = src.astype(np.uint64)
            pair_keys <<= np.uint64(32)
            pair_keys |= dst
            pair_ids, pair_count, pair_last = _aggregate_history(pair_keys, time)
            del pair_keys
            arrays = {
                "src_ids": src_ids,
                "src_count": src_count,
                "src_last": src_last,
                "dst_ids": dst_ids,
                "dst_count": dst_count,
                "dst_last": dst_last,
                "pair_ids": pair_ids,
                "pair_count": pair_count,
                "pair_last": pair_last,
            }
            for name, values in arrays.items():
                _atomic_npy(stage / f"{name}.npy", values)
            source_hot = _hot_threshold(src_count)
            destination_hot = _hot_threshold(dst_count)
            _atomic_json(
                stage / "metadata.json",
                {
                    "cache_version": CACHE_VERSION,
                    "archive_hash": cache.archive_hash,
                    "scene": cache.scene,
                    "cutoff": cutoff,
                    "history_rows": end,
                    "feature_names": list(FEATURE_NAMES),
                    "source_hot_count": source_hot,
                    "destination_hot_count": destination_hot,
                },
            )
            try:
                os.replace(stage, root)
            except FileExistsError:
                existing = cls(root)
                if existing.metadata.get("archive_hash") != cache.archive_hash:
                    raise DataContractError(f"conflicting feature store exists: {root}")
                existing._validate_files()
                return existing
            return cls(root)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    @property
    def feature_dim(self) -> int:
        return len(FEATURE_NAMES)

    @property
    def source_vocab_size(self) -> int:
        return len(self._array("src_ids")) + 1

    @property
    def destination_vocab_size(self) -> int:
        return len(self._array("dst_ids")) + 1

    def _array(self, name: str) -> np.memmap:
        if name not in self._arrays:
            values = _open_memmap(self.root / f"{name}.npy")
            expected = np.uint64 if name == "pair_ids" else np.uint32
            if values.dtype != np.dtype(expected) or values.ndim != 1:
                raise DataContractError(f"invalid feature-store array {name}.npy")
            self._arrays[name] = values
        return self._arrays[name]

    def _validate_files(self) -> None:
        for name in (
            "src_ids",
            "src_count",
            "src_last",
            "dst_ids",
            "dst_count",
            "dst_last",
            "pair_ids",
            "pair_count",
            "pair_last",
        ):
            self._array(name)
        if len(self._array("src_ids")) != len(self._array("src_count")) or len(
            self._array("src_ids")
        ) != len(self._array("src_last")):
            raise DataContractError("source stats have mismatched lengths")
        if len(self._array("dst_ids")) != len(self._array("dst_count")) or len(
            self._array("dst_ids")
        ) != len(self._array("dst_last")):
            raise DataContractError("destination stats have mismatched lengths")
        if len(self._array("pair_ids")) != len(self._array("pair_count")) or len(
            self._array("pair_ids")
        ) != len(self._array("pair_last")):
            raise DataContractError("pair stats have mismatched lengths")

    def source_indices(self, values: np.ndarray) -> np.ndarray:
        """Map historical source ids to compact 1-based indices; 0 means unknown."""
        return _compact_indices(values, self._array("src_ids"))

    def destination_indices(self, values: np.ndarray) -> np.ndarray:
        """Map historical destination ids to compact 1-based indices; 0 means unknown."""
        return _compact_indices(values, self._array("dst_ids"))

    def features(
        self, src: np.ndarray, time: np.ndarray, candidates: np.ndarray
    ) -> np.ndarray:
        """Return float32 candidate features for a bounded query batch.

        The store is static at ``cutoff``.  Query timestamps before it are
        rejected so callers cannot accidentally use future history.
        """
        src, time, candidates = _query_arrays(src, time, candidates)
        if len(time) and int(time.min()) < self.cutoff:
            raise DataContractError(
                f"query time precedes feature-store cutoff {self.cutoff}"
            )
        rows, width = candidates.shape
        src_count, src_last = self._lookup_source(src)
        dst_count, dst_last = self._lookup_destination(candidates)
        pair_count, pair_last = self._lookup_pair(src, candidates)

        output = np.empty((rows, width, self.feature_dim), dtype=np.float32)
        src_count_f = src_count.astype(np.float32, copy=False)
        dst_count_f = dst_count.astype(np.float32, copy=False)
        pair_count_f = pair_count.astype(np.float32, copy=False)
        output[:, :, 0] = np.log1p(pair_count_f)
        output[:, :, 1] = _recency(time[:, None], pair_last, pair_count > 0)
        output[:, :, 2] = (pair_count > 0).astype(np.float32)
        output[:, :, 3] = np.log1p(src_count_f)[:, None]
        output[:, :, 4] = _recency(time, src_last, src_count > 0)[:, None]
        output[:, :, 5] = (src_count > 0).astype(np.float32)[:, None]
        output[:, :, 6] = np.log1p(dst_count_f)
        output[:, :, 7] = _recency(time[:, None], dst_last, dst_count > 0)
        output[:, :, 8] = (dst_count > 0).astype(np.float32)
        output[:, :, 9] = pair_count_f / np.maximum(src_count_f[:, None], 1.0)
        output[:, :, 10] = pair_count_f / np.maximum(dst_count_f, 1.0)
        return output

    def iter_test_features(
        self,
        data_zip: str | Path,
        scene: str,
        *,
        chunk_rows: int = 8192,
    ) -> Iterator[tuple[TestChunk, np.ndarray]]:
        """Stream bounded Dataset3/4 test features in original row order."""
        for chunk in iter_test_chunks(data_zip, scene, chunk_rows=chunk_rows):
            yield chunk, self.features(chunk.src, chunk.time, chunk.candidates)

    def source_hot_mask(self, src: np.ndarray) -> np.ndarray:
        """Return the frozen-history hot-source mask for one source vector."""
        src = _as_uint32_array(src, "src")
        if src.ndim != 1:
            raise ValueError("src must be one-dimensional")
        src_count, _ = self._lookup_source(src)
        return src_count >= int(self.metadata["source_hot_count"])

    def evaluation_segments(
        self,
        src: np.ndarray,
        time: np.ndarray,
        candidates: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return mutually informative warm/hot/pair masks for metrics."""
        src, time, candidates = _query_arrays(src, time, candidates)
        labels = _labels(labels, len(src), candidates.shape[1])
        positive = candidates[np.arange(len(src)), labels]
        src_count, _ = self._lookup_source(src)
        dst_count, _ = self._lookup_destination(positive)
        pair_count, _ = self._lookup_pair(src, positive[:, None])
        pair_seen = pair_count[:, 0] > 0
        source_warm = src_count > 0
        destination_warm = dst_count > 0
        destination_hot = destination_warm & (
            dst_count >= int(self.metadata["destination_hot_count"])
        )
        source_hot = source_warm & (src_count >= int(self.metadata["source_hot_count"]))
        return {
            "source_warm": source_warm,
            "source_cold": ~source_warm,
            "source_hot": source_hot,
            "destination_warm": destination_warm,
            "destination_cold": ~destination_warm,
            "destination_hot": destination_hot,
            "destination_tail": destination_warm & ~destination_hot,
            "pair_seen": pair_seen,
            "pair_new": ~pair_seen,
        }

    def _lookup_source(self, src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _lookup_values(
            src,
            self._array("src_ids"),
            self._array("src_count"),
            self._array("src_last"),
        )

    def _lookup_destination(self, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _lookup_values(
            dst,
            self._array("dst_ids"),
            self._array("dst_count"),
            self._array("dst_last"),
        )

    def _lookup_pair(
        self, src: np.ndarray, dst: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        src = _as_uint32_array(src, "src")
        dst = _as_uint32_array(dst, "dst")
        if dst.ndim == 1:
            if src.ndim != 1 or len(src) != len(dst):
                raise ValueError("one-dimensional src and dst must have matching length")
            keys = src.astype(np.uint64)
            keys <<= np.uint64(32)
            keys |= dst
        elif dst.ndim == 2:
            if src.ndim != 1 or len(src) != len(dst):
                raise ValueError("src length must match candidate rows")
            keys = (src.astype(np.uint64)[:, None] << np.uint64(32)) | dst.astype(
                np.uint64, copy=False
            )
        else:
            raise ValueError("dst must be one- or two-dimensional")
        return _lookup_values(
            keys,
            self._array("pair_ids"),
            self._array("pair_count"),
            self._array("pair_last"),
        )


def _hot_threshold(counts: np.ndarray) -> int:
    counts = np.asarray(counts)
    if not len(counts):
        return 1
    return max(1, int(np.ceil(np.quantile(counts, 0.80))))


def _compact_indices(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    values = _as_uint32_array(values, "ids")
    flat = values.reshape(-1)
    output = np.zeros(flat.shape, dtype=np.int32)
    if len(ids) and len(flat):
        positions = np.searchsorted(ids, flat)
        match = positions < len(ids)
        match[match] &= ids[positions[match]] == flat[match]
        output[match] = positions[match].astype(np.int32) + 1
    return output.reshape(values.shape)


def _lookup_values(
    values: np.ndarray,
    ids: np.ndarray,
    counts: np.ndarray,
    last: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    flat = values.reshape(-1)
    output_count = np.zeros(flat.shape, dtype=np.uint32)
    output_last = np.zeros(flat.shape, dtype=np.uint32)
    if len(ids) and len(flat):
        positions = np.searchsorted(ids, flat)
        match = positions < len(ids)
        match[match] &= ids[positions[match]] == flat[match]
        output_count[match] = counts[positions[match]]
        output_last[match] = last[positions[match]]
    return output_count.reshape(values.shape), output_last.reshape(values.shape)


def _as_uint32_array(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype")
    if values.size:
        if np.issubdtype(values.dtype, np.signedinteger) and int(values.min()) < 0:
            raise ValueError(f"{name} cannot contain negative ids")
        if int(values.max()) > UINT32_MAX:
            raise ValueError(f"{name} exceeds uint32")
    return values.astype(np.uint32, copy=False)


def _query_arrays(
    src: np.ndarray, time: np.ndarray, candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = _as_uint32_array(src, "src")
    time = _as_uint32_array(time, "time")
    candidates = _as_uint32_array(candidates, "candidates")
    if src.ndim != 1 or time.ndim != 1:
        raise ValueError("src and time must be one-dimensional")
    if candidates.ndim != 2 or candidates.shape[1] < 1:
        raise ValueError("candidates must have shape (rows, candidate_count)")
    if len(src) != len(time) or len(src) != len(candidates):
        raise ValueError("src, time and candidates must have matching row counts")
    return src, time, candidates


def _recency(query_time: np.ndarray, last_time: np.ndarray, known: np.ndarray) -> np.ndarray:
    query_time = np.asarray(query_time, dtype=np.int64)
    last_time = np.asarray(last_time, dtype=np.int64)
    known = np.asarray(known, dtype=bool)
    gap = np.maximum(query_time - last_time, 0)
    output = 1.0 / (1.0 + np.log1p(gap.astype(np.float64) / SECONDS_PER_DAY))
    return np.where(known, output, 0.0).astype(np.float32)


@dataclass(frozen=True)
class Split1Plan:
    """Timestamp-safe split1 ranges and the earlier history cutoff for each."""

    scene: str
    split1_start: int
    split1_stop: int
    split1_first_time: int
    ranges: Mapping[str, tuple[int, int]]
    cutoffs: Mapping[str, int]
    time_ranges: Mapping[str, tuple[int, int]]
    ratios: tuple[float, float, float]

    @classmethod
    def from_cache(
        cls, cache: BDataCache, *, ratios: Sequence[float] = (0.60, 0.20, 0.20)
    ) -> "Split1Plan":
        ratios = _normalise_ratios(ratios)
        start, stop = cache.split1_bounds()
        times = cache.time[start:stop]
        if not len(times):
            raise DataContractError("split1 is empty")
        if np.any(times[1:] < times[:-1]):
            raise DataContractError("split1 timestamps are not monotone")
        first_time = int(times[0])
        cut1, cut2 = _timestamp_cuts(times, ratios)
        local_ranges = {
            "train": (0, cut1),
            "valid": (cut1, cut2),
            "confirm": (cut2, len(times)),
        }
        ranges = {
            name: (start + lower, start + upper)
            for name, (lower, upper) in local_ranges.items()
        }
        cutoffs = {
            "train": first_time,
            "valid": int(times[cut1]),
            "confirm": int(times[cut2]),
        }
        time_ranges = {
            name: (int(times[lower]), int(times[upper - 1]))
            for name, (lower, upper) in local_ranges.items()
        }
        return cls(
            scene=cache.scene,
            split1_start=start,
            split1_stop=stop,
            split1_first_time=first_time,
            ranges=ranges,
            cutoffs=cutoffs,
            time_ranges=time_ranges,
            ratios=ratios,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "split1_rows": [self.split1_start, self.split1_stop],
            "split1_first_time": self.split1_first_time,
            "ranges": {name: list(bounds) for name, bounds in self.ranges.items()},
            "cutoffs": dict(self.cutoffs),
            "time_ranges": {
                name: list(bounds) for name, bounds in self.time_ranges.items()
            },
            "ratios": list(self.ratios),
        }


def _normalise_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("ratios must contain train, valid and confirm values")
    values = tuple(float(value) for value in ratios)
    if any(value <= 0.0 for value in values) or not np.isclose(sum(values), 1.0):
        raise ValueError("ratios must be positive and sum to 1")
    return values


def _timestamp_cuts(times: np.ndarray, ratios: Sequence[float]) -> tuple[int, int]:
    """Choose two non-empty cuts without splitting equal timestamp buckets."""
    length = len(times)
    if length < 3:
        raise DataContractError("split1 needs at least three rows for train/valid/confirm")
    desired = (int(round(length * ratios[0])), int(round(length * (ratios[0] + ratios[1]))))
    cuts: list[int] = []
    lower = 0
    for index, target in enumerate(desired):
        target = min(max(target, lower + 1), length - (2 - index))
        timestamp = times[target]
        left = int(np.searchsorted(times, timestamp, side="left"))
        right = int(np.searchsorted(times, timestamp, side="right"))
        # Prefer the left edge: all ties belong to the later split.  If it would
        # empty the previous split, place the whole tie bucket before the cut.
        cut = left if left > lower else right
        if cut <= lower or cut >= length:
            raise DataContractError("cannot form three strict timestamp buckets in split1")
        cuts.append(cut)
        lower = cut
    if not (0 < cuts[0] < cuts[1] < length):
        raise DataContractError("invalid timestamp-safe split1 cuts")
    return cuts[0], cuts[1]


@dataclass(frozen=True)
class GroupBatch:
    """One bounded replay candidate batch."""

    src: np.ndarray
    time: np.ndarray
    candidates: np.ndarray
    labels: np.ndarray
    row_index: np.ndarray


@dataclass(frozen=True)
class FeatureBatch:
    """Features and labels aligned for a Jittor candidate ranker."""

    features: np.ndarray
    labels: np.ndarray
    source: np.ndarray
    time: np.ndarray
    candidates: np.ndarray
    row_index: np.ndarray


class CandidateGroup:
    """Disk-backed replay group with one true destination per candidate row."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise DataContractError(f"not a candidate group: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.name = str(self.metadata["name"])
        if self.name not in GROUP_NAMES:
            raise DataContractError(f"invalid candidate-group name: {self.name}")
        self.cutoff = _as_time(self.metadata["cutoff"])
        self._arrays: dict[str, np.memmap] = {}
        self._validate_files()

    @property
    def rows(self) -> int:
        return int(self.metadata["rows"])

    @property
    def candidate_count(self) -> int:
        return int(self.metadata["candidate_count"])

    @property
    def src(self) -> np.memmap:
        return self._array("src")

    @property
    def time(self) -> np.memmap:
        return self._array("time")

    @property
    def candidates(self) -> np.memmap:
        return self._array("candidates")

    @property
    def labels(self) -> np.memmap:
        return self._array("labels")

    @property
    def row_index(self) -> np.memmap:
        return self._array("row_index")

    def _array(self, name: str) -> np.memmap:
        if name not in self._arrays:
            values = _open_memmap(self.root / f"{name}.npy")
            if name == "candidates":
                expected_shape = (self.rows, self.candidate_count)
                expected_dtype = np.uint32
            elif name == "labels":
                expected_shape = (self.rows,)
                expected_dtype = np.uint16
            elif name == "row_index":
                expected_shape = (self.rows,)
                expected_dtype = np.uint64
            else:
                expected_shape = (self.rows,)
                expected_dtype = np.uint32
            if values.shape != expected_shape or values.dtype != np.dtype(expected_dtype):
                raise DataContractError(f"invalid group array {self.root / (name + '.npy')}")
            self._arrays[name] = values
        return self._arrays[name]

    def _validate_files(self) -> None:
        for name in ("src", "time", "candidates", "labels", "row_index"):
            self._array(name)
        if self.rows and (np.any(self.labels >= self.candidate_count) or np.any(self.time < self.cutoff)):
            raise DataContractError(f"invalid labels or causal times in {self.root}")

    def iter_batches(self, *, batch_rows: int = 8192) -> Iterator[GroupBatch]:
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        for start in range(0, self.rows, batch_rows):
            stop = min(self.rows, start + batch_rows)
            yield GroupBatch(
                src=self.src[start:stop],
                time=self.time[start:stop],
                candidates=self.candidates[start:stop],
                labels=self.labels[start:stop].astype(np.int64, copy=False),
                row_index=self.row_index[start:stop],
            )

    def iter_feature_batches(
        self, store: FeatureStore, *, batch_rows: int = 8192
    ) -> Iterator[FeatureBatch]:
        if store.cutoff != self.cutoff:
            raise DataContractError(
                f"group {self.name} requires cutoff {self.cutoff}, got {store.cutoff}"
            )
        for batch in self.iter_batches(batch_rows=batch_rows):
            yield FeatureBatch(
                features=store.features(batch.src, batch.time, batch.candidates),
                labels=batch.labels,
                source=batch.src,
                time=batch.time,
                candidates=batch.candidates,
                row_index=batch.row_index,
            )

    def materialize_features(
        self,
        store: FeatureStore,
        path: str | Path,
        *,
        batch_rows: int = 8192,
        max_bytes: int | None = 1 << 30,
    ) -> np.memmap:
        """Write disk-backed float32 features, never a full in-memory score plane.

        ``max_bytes`` is a guard against accidentally creating a multi-gigabyte
        Dataset4 feature plane.  Pass ``None`` only after sizing the requested
        group deliberately.
        """
        if store.cutoff != self.cutoff:
            raise DataContractError("feature store cutoff does not match candidate group")
        path = Path(path)
        required = self.rows * self.candidate_count * store.feature_dim * np.dtype(np.float32).itemsize
        if max_bytes is not None and required > int(max_bytes):
            raise ValueError(
                f"feature memmap would be {required} bytes; max_bytes is {max_bytes}"
            )
        if path.exists():
            existing = _open_memmap(path)
            if existing.shape != (self.rows, self.candidate_count, store.feature_dim):
                raise DataContractError(f"existing feature memmap has wrong shape: {path}")
            return existing
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        mapped = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(self.rows, self.candidate_count, store.feature_dim),
        )
        try:
            for start in range(0, self.rows, batch_rows):
                stop = min(self.rows, start + batch_rows)
                mapped[start:stop] = store.features(
                    self.src[start:stop], self.time[start:stop], self.candidates[start:stop]
                )
            mapped.flush()
            del mapped
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return _open_memmap(path)

    def validate_positive_uniqueness(self, *, batch_rows: int = 8192) -> None:
        """Check that the positive id appears exactly once in each candidate row."""
        for batch in self.iter_batches(batch_rows=batch_rows):
            positive = batch.candidates[np.arange(len(batch.labels)), batch.labels]
            appearances = (batch.candidates == positive[:, None]).sum(axis=1)
            if np.any(appearances != 1):
                raise DataContractError(
                    f"candidate group {self.name} contains a duplicated positive"
                )


class Split1Groups(Mapping[str, CandidateGroup]):
    """Three immutable split1 replay groups plus their timestamp-safe plan."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise DataContractError(f"not a split1 group set: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.plan = _plan_from_dict(self.metadata["plan"])
        self._groups = {
            name: CandidateGroup(self.root / name)
            for name in GROUP_NAMES
        }

    def __getitem__(self, name: str) -> CandidateGroup:
        if name == "validation":
            name = "valid"
        return self._groups[name]

    def __iter__(self) -> Iterator[str]:
        return iter(GROUP_NAMES)

    def __len__(self) -> int:
        return len(GROUP_NAMES)

    @property
    def train(self) -> CandidateGroup:
        return self._groups["train"]

    @property
    def valid(self) -> CandidateGroup:
        return self._groups["valid"]

    @property
    def confirm(self) -> CandidateGroup:
        return self._groups["confirm"]


def _plan_from_dict(value: Mapping[str, Any]) -> Split1Plan:
    return Split1Plan(
        scene=str(value["scene"]),
        split1_start=int(value["split1_rows"][0]),
        split1_stop=int(value["split1_rows"][1]),
        split1_first_time=int(value["split1_first_time"]),
        ranges={name: tuple(map(int, bounds)) for name, bounds in value["ranges"].items()},
        cutoffs={name: int(cutoff) for name, cutoff in value["cutoffs"].items()},
        time_ranges={
            name: tuple(map(int, bounds)) for name, bounds in value["time_ranges"].items()
        },
        ratios=tuple(map(float, value["ratios"])),
    )


def build_split1_groups(
    cache: BDataCache,
    *,
    seed: int = DEFAULT_SEED,
    candidate_count: int = 100,
    ratios: Sequence[float] = (0.60, 0.20, 0.20),
    sizes: Mapping[str, int | None] | None = None,
    batch_rows: int = 4096,
    negative_strategy: str = "history",
    cold_fraction: float = 0.15,
    popularity_quantile: float = 0.80,
    popularity_power: float = 0.75,
    test_pool_chunk_rows: int = 8192,
    test_pool_max_dense_ids: int = 10_000_000,
) -> Split1Groups:
    """Build/read strict split1 train, validation and confirmation replay groups.

    ``history`` is the primary causal replay: negatives are exactly historical
    destinations from ``time < split1_first_time``.  ``history_cold`` is an
    optional calibration/stress replay that mixes in unseen IDs from that
    historical destination-ID range; it is not the primary selection replay.
    ``popularity_hard`` is also causal, but samples the historical popular
    tail.  ``test_pool`` draws from unlabeled official test candidate cells and
    is explicitly a distribution diagnostic; it must not replace the history
    replay when selecting a model.  A positive is inserted in a seeded random
    column after collision repair, so it appears exactly once in each row.
    """
    if not 2 <= int(candidate_count) <= np.iinfo(np.uint16).max:
        raise ValueError("candidate_count must be in [2, 65535]")
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    negative_strategy = _negative_strategy(negative_strategy)
    if negative_strategy == "history_cold":
        try:
            cold_fraction = float(cold_fraction)
        except (TypeError, ValueError) as exc:
            raise TypeError("cold_fraction must be a finite number") from exc
        if not np.isfinite(cold_fraction) or not 0.0 <= cold_fraction <= 1.0:
            raise ValueError("cold_fraction must be in [0, 1]")
    else:
        cold_fraction = 0.0
    if not 0.0 <= float(popularity_quantile) < 1.0:
        raise ValueError("popularity_quantile must be in [0, 1)")
    if float(popularity_power) <= 0.0:
        raise ValueError("popularity_power must be positive")
    candidate_count = int(candidate_count)
    plan = cache.split1_plan(ratios)
    requested = _group_sizes(sizes)
    config = {
        "archive_hash": cache.archive_hash,
        "scene": cache.scene,
        "seed": int(seed),
        "candidate_count": candidate_count,
        "plan": plan.as_dict(),
        "sizes": requested,
        "negative_strategy": negative_strategy,
        "popularity_quantile": float(popularity_quantile),
        "popularity_power": float(popularity_power),
        "test_pool_chunk_rows": int(test_pool_chunk_rows),
        "test_pool_max_dense_ids": int(test_pool_max_dense_ids),
    }
    if negative_strategy == "history_cold":
        config["cold_fraction"] = cold_fraction
    root = cache.root / "groups" / f"split1_{sha256_json(config)[:20]}"
    if root.is_dir():
        groups = Split1Groups(root)
        if groups.metadata.get("config_hash") != sha256_json(config):
            raise DataContractError(f"candidate-group config mismatch: {root}")
        return groups

    pool, probabilities, cold_pool, pool_metadata = _negative_pool(
        cache,
        cutoff=plan.split1_first_time,
        strategy=negative_strategy,
        cold_fraction=cold_fraction,
        popularity_quantile=float(popularity_quantile),
        popularity_power=float(popularity_power),
        test_pool_chunk_rows=int(test_pool_chunk_rows),
        test_pool_max_dense_ids=int(test_pool_max_dense_ids),
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = root.parent / f".{root.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    try:
        written: dict[str, int] = {}
        for ordinal, name in enumerate(GROUP_NAMES):
            lower, upper = plan.ranges[name]
            available = upper - lower
            rows = min(available, requested[name])
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), ordinal + 1]))
            _write_candidate_group(
                destination=stage / name,
                name=name,
                cache=cache,
                lower=lower,
                upper=upper,
                rows=rows,
                cutoff=plan.cutoffs[name],
                pool=pool,
                probabilities=probabilities,
                cold_pool=cold_pool,
                cold_fraction=cold_fraction,
                pool_metadata=pool_metadata,
                candidate_count=candidate_count,
                rng=rng,
                batch_rows=batch_rows,
            )
            written[name] = rows
        metadata = {
            "cache_version": CACHE_VERSION,
            "archive_hash": cache.archive_hash,
            "scene": cache.scene,
            "config_hash": sha256_json(config),
            "config": config,
            "plan": plan.as_dict(),
            "rows": written,
            "negative_pool_hash": sha256_array(pool),
            "negative_pool": pool_metadata,
        }
        if cold_pool is not None:
            metadata["cold_negative_pool_hash"] = sha256_array(cold_pool)
        _atomic_json(stage / "metadata.json", metadata)
        try:
            os.replace(stage, root)
        except FileExistsError:
            existing = Split1Groups(root)
            if existing.metadata.get("config_hash") != sha256_json(config):
                raise DataContractError(f"conflicting candidate groups exist: {root}")
            return existing
        return Split1Groups(root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


make_split1_groups = build_split1_groups


def _negative_strategy(value: str) -> str:
    aliases = {"historical": "history", "popularity": "popularity_hard"}
    value = aliases.get(str(value), str(value))
    if value not in NEGATIVE_STRATEGIES:
        raise ValueError(f"negative_strategy must be one of {sorted(NEGATIVE_STRATEGIES)}")
    return value


def _negative_pool(
    cache: BDataCache,
    *,
    cutoff: int,
    strategy: str,
    cold_fraction: float,
    popularity_quantile: float,
    popularity_power: float,
    test_pool_chunk_rows: int,
    test_pool_max_dense_ids: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Resolve one replay negative sampler without retaining test rows."""
    if strategy == "history":
        ids, counts = cache.history_dst_counts(cutoff)
        return ids, None, None, {
            "strategy": strategy,
            "source": "historical destinations only, edge_time < split1_first_time",
            "causal_primary_replay": True,
            "ids": int(len(ids)),
            "count_hash": sha256_array(counts),
        }
    if strategy == "history_cold":
        ids, counts = cache.history_dst_counts(cutoff)
        cold_ids = _cold_destination_pool(ids)
        if cold_fraction > 0.0 and not len(cold_ids):
            raise DataContractError(
                "historical destination range has no unseen IDs for history_cold"
            )
        return ids, None, cold_ids, {
            "strategy": strategy,
            "source": (
                "historical destinations plus unseen IDs within the historical "
                "destination numeric range, edge_time < split1_first_time"
            ),
            "causal_primary_replay": False,
            "calibration_stress_only": True,
            "warm_ids": int(len(ids)),
            "cold_ids": int(len(cold_ids)),
            "cold_fraction": cold_fraction,
            "cold_range": [int(ids[0]), int(ids[-1])],
            "count_hash": sha256_array(counts),
            "cold_ids_hash": sha256_array(cold_ids),
        }
    if strategy == "popularity_hard":
        ids, counts = cache.history_dst_counts(cutoff)
        threshold = int(np.ceil(np.quantile(counts, popularity_quantile)))
        mask = counts >= threshold
        hard_ids = np.asarray(ids[mask], dtype=np.uint32)
        hard_counts = np.asarray(counts[mask], dtype=np.float64)
        if len(hard_ids) < 2:
            raise DataContractError("popularity-hard historical pool has fewer than two ids")
        probabilities = np.power(hard_counts, popularity_power)
        probabilities /= probabilities.sum()
        return hard_ids, probabilities, None, {
            "strategy": strategy,
            "source": "historical destinations only, edge_time < split1_first_time",
            "causal_primary_replay": True,
            "popularity_quantile": popularity_quantile,
            "popularity_threshold": threshold,
            "popularity_power": popularity_power,
            "ids": int(len(hard_ids)),
            "count_hash": sha256_array(hard_counts),
        }
    ids, counts = cache.test_candidate_counts(
        chunk_rows=test_pool_chunk_rows,
        max_dense_ids=test_pool_max_dense_ids,
    )
    probabilities = np.asarray(counts, dtype=np.float64)
    probabilities /= probabilities.sum()
    return ids, probabilities, None, {
        "strategy": "test_pool",
        "source": "unlabeled official test candidate cells only",
        "causal_primary_replay": False,
        "diagnostic_only": True,
        "ids": int(len(ids)),
        "count_hash": sha256_array(counts),
    }


def _cold_destination_pool(history_ids: np.ndarray) -> np.ndarray:
    """Return unseen IDs inside a strictly historical destination-ID range."""
    history_ids = _as_uint32_array(history_ids, "historical destination ids")
    if history_ids.ndim != 1 or not len(history_ids):
        raise DataContractError("historical destination pool is empty")
    if len(history_ids) > 1 and np.any(history_ids[1:] <= history_ids[:-1]):
        raise DataContractError("historical destination IDs are not strictly sorted")
    lower, upper = int(history_ids[0]), int(history_ids[-1])
    width = upper - lower + 1
    if width > MAX_COLD_POOL_IDS:
        raise DataContractError(
            f"historical destination range has {width} IDs; limit is {MAX_COLD_POOL_IDS}"
        )
    domain = np.arange(lower, upper + 1, dtype=np.uint32)
    positions = np.searchsorted(history_ids, domain)
    known = positions < len(history_ids)
    known[known] = history_ids[positions[known]] == domain[known]
    return domain[~known]


def _group_sizes(sizes: Mapping[str, int | None] | None) -> dict[str, int]:
    defaults = {
        "train": DEFAULT_TRAIN_GROUPS,
        "valid": DEFAULT_VALID_GROUPS,
        "confirm": DEFAULT_CONFIRM_GROUPS,
    }
    if sizes is None:
        return defaults
    unknown = set(sizes).difference(GROUP_NAMES).difference({"validation"})
    if unknown:
        raise ValueError(f"unknown group sizes: {sorted(unknown)}")
    values = defaults.copy()
    for name, size in sizes.items():
        name = "valid" if name == "validation" else name
        if size is None:
            # ``None`` means all available rows, resolved after timestamp slicing.
            values[name] = np.iinfo(np.int64).max
        else:
            size = int(size)
            if size < 1:
                raise ValueError(f"group size for {name} must be positive or None")
            values[name] = size
    return values


def _write_candidate_group(
    *,
    destination: Path,
    name: str,
    cache: BDataCache,
    lower: int,
    upper: int,
    rows: int,
    cutoff: int,
    pool: np.ndarray,
    probabilities: np.ndarray | None,
    cold_pool: np.ndarray | None,
    cold_fraction: float,
    pool_metadata: Mapping[str, Any],
    candidate_count: int,
    rng: np.random.Generator,
    batch_rows: int,
) -> None:
    if rows < 1:
        raise DataContractError(f"timestamp partition {name} has no rows")
    destination.mkdir(parents=True, exist_ok=False)
    available = upper - lower
    if rows > available:
        raise ValueError("requested more candidate groups than positive rows")
    # Sampling row offsets before writing keeps every group reproducible while
    # retaining original chronological order for auditability.
    if rows == available:
        row_index = np.arange(lower, upper, dtype=np.uint64)
    else:
        row_index = np.sort(
            rng.choice(available, size=rows, replace=False).astype(np.uint64) + lower
        )
    arrays = {
        "src": np.lib.format.open_memmap(
            destination / "src.npy", mode="w+", dtype=np.uint32, shape=(rows,)
        ),
        "time": np.lib.format.open_memmap(
            destination / "time.npy", mode="w+", dtype=np.uint32, shape=(rows,)
        ),
        "candidates": np.lib.format.open_memmap(
            destination / "candidates.npy",
            mode="w+",
            dtype=np.uint32,
            shape=(rows, candidate_count),
        ),
        "labels": np.lib.format.open_memmap(
            destination / "labels.npy", mode="w+", dtype=np.uint16, shape=(rows,)
        ),
        "row_index": np.lib.format.open_memmap(
            destination / "row_index.npy", mode="w+", dtype=np.uint64, shape=(rows,)
        ),
    }
    try:
        for start in range(0, rows, batch_rows):
            stop = min(rows, start + batch_rows)
            selected = row_index[start:stop]
            selected_index = selected.astype(np.int64, copy=False)
            src = np.asarray(cache.src[selected_index], dtype=np.uint32)
            time = np.asarray(cache.time[selected_index], dtype=np.uint32)
            positive = np.asarray(cache.dst[selected_index], dtype=np.uint32)
            candidates = _sample_candidate_rows(
                pool,
                positive,
                candidate_count,
                rng,
                probabilities=probabilities,
                cold_pool=cold_pool,
                cold_fraction=cold_fraction,
            )
            labels = rng.integers(
                0, candidate_count, size=stop - start, dtype=np.uint16
            )
            candidates[np.arange(stop - start), labels] = positive
            arrays["src"][start:stop] = src
            arrays["time"][start:stop] = time
            arrays["candidates"][start:stop] = candidates
            arrays["labels"][start:stop] = labels
            arrays["row_index"][start:stop] = selected
        for values in arrays.values():
            values.flush()
        metadata = {
            "cache_version": CACHE_VERSION,
            "name": name,
            "rows": rows,
            "candidate_count": candidate_count,
            "cutoff": cutoff,
            "train_row_range": [lower, upper],
            "time_range": [int(cache.time[lower]), int(cache.time[upper - 1])],
            "negative_pool": dict(pool_metadata),
            "positive_appears_once": True,
        }
        _atomic_json(destination / "metadata.json", metadata)
    finally:
        for values in arrays.values():
            values.flush()
        arrays.clear()


def _sample_candidate_rows(
    pool: np.ndarray,
    positive: np.ndarray,
    candidate_count: int,
    rng: np.random.Generator,
    *,
    probabilities: np.ndarray | None = None,
    cold_pool: np.ndarray | None = None,
    cold_fraction: float = 0.0,
) -> np.ndarray:
    """Sample negatives from causal pools and remove positive collisions."""
    pool = _as_uint32_array(pool, "negative pool")
    positive = _as_uint32_array(positive, "positive")
    if pool.ndim != 1 or positive.ndim != 1:
        raise ValueError("pool and positive must be one-dimensional")
    if len(pool) < 2:
        raise DataContractError("negative pool has fewer than two destinations")
    if not np.isfinite(cold_fraction) or not 0.0 <= cold_fraction <= 1.0:
        raise ValueError("cold_fraction must be in [0, 1]")
    if cold_pool is not None:
        cold_pool = _as_uint32_array(cold_pool, "cold negative pool")
        if cold_pool.ndim != 1 or (cold_fraction > 0.0 and not len(cold_pool)):
            raise DataContractError("cold negative pool is empty or malformed")
    elif cold_fraction:
        raise ValueError("cold_fraction requires a cold negative pool")
    if probabilities is None:
        def draw(size: int | tuple[int, ...]) -> np.ndarray:
            return rng.integers(0, len(pool), size=size)
    else:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != pool.shape or not np.isfinite(probabilities).all():
            raise ValueError("negative-pool probabilities must match pool ids")
        if np.any(probabilities < 0.0) or not probabilities.sum() > 0.0:
            raise ValueError("negative-pool probabilities must be non-negative and nonzero")
        probabilities = probabilities / probabilities.sum()

        def draw(size: int | tuple[int, ...]) -> np.ndarray:
            return rng.choice(len(pool), size=size, replace=True, p=probabilities)

    # Per-row replacement is intentional: the requirement is that the true
    # positive does not repeat.  It also resembles official candidate rows,
    # which can contain duplicate negative ids.
    choices = draw((len(positive), candidate_count))
    candidates = pool[choices]
    if cold_pool is not None and cold_fraction:
        cold_mask = rng.random(candidates.shape) < cold_fraction
        candidates[cold_mask] = cold_pool[
            rng.integers(0, len(cold_pool), size=int(cold_mask.sum()))
        ]
    collisions = candidates == positive[:, None]
    while np.any(collisions):
        candidates[collisions] = pool[draw(int(collisions.sum()))]
        collisions = candidates == positive[:, None]
    return candidates


def _labels(labels: np.ndarray, rows: int, candidate_count: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != rows:
        raise ValueError(f"labels must have shape ({rows},)")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must have integer dtype")
    labels = labels.astype(np.int64, copy=False)
    if len(labels) and (np.any(labels < 0) or np.any(labels >= candidate_count)):
        raise ValueError("labels are outside candidate columns")
    return labels


def ranks_from_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return one-based ranks with stable original-column tie breaking."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError("scores must have shape (rows, candidate_count)")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    labels = _labels(labels, len(scores), scores.shape[1])
    positive = scores[np.arange(len(scores)), labels]
    columns = np.arange(scores.shape[1], dtype=np.int64)
    rank = 1 + (scores > positive[:, None]).sum(axis=1)
    rank += ((scores == positive[:, None]) & (columns[None, :] < labels[:, None])).sum(axis=1)
    return rank.astype(np.int32, copy=False)


class RankingMetrics:
    """Streaming MRR/top-1 accumulator with optional segment masks."""

    def __init__(self) -> None:
        self._total_n = 0
        self._total_mrr = 0.0
        self._total_top1 = 0
        self._segments: dict[str, list[float]] = {}

    def update(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        segments: Mapping[str, np.ndarray] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Accumulate one bounded score block and return its ranks."""
        rank = ranks_from_scores(scores, labels)
        reciprocal = 1.0 / rank.astype(np.float64)
        self._total_n += len(rank)
        self._total_mrr += float(reciprocal.sum())
        self._total_top1 += int((rank == 1).sum())
        for name, mask in _segment_mapping(segments, len(rank)).items():
            count = int(mask.sum())
            values = self._segments.setdefault(str(name), [0.0, 0.0, 0.0])
            values[0] += count
            if count:
                values[1] += float(reciprocal[mask].sum())
                values[2] += int((rank[mask] == 1).sum())
        return rank

    def result(self) -> dict[str, Any]:
        return {
            "n": self._total_n,
            "mrr": _divide_or_none(self._total_mrr, self._total_n),
            "top1": _divide_or_none(self._total_top1, self._total_n),
            "segments": {
                name: {
                    "n": int(values[0]),
                    "mrr": _divide_or_none(values[1], int(values[0])),
                    "top1": _divide_or_none(values[2], int(values[0])),
                }
                for name, values in sorted(self._segments.items())
            },
        }


def ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    segments: Mapping[str, np.ndarray] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute MRR/top-1 for one score block, including optional segments."""
    metrics = RankingMetrics()
    metrics.update(scores, labels, segments=segments)
    return metrics.result()


mrr_top1 = ranking_metrics


def _segment_mapping(
    segments: Mapping[str, np.ndarray] | np.ndarray | None, rows: int
) -> Mapping[str, np.ndarray]:
    if segments is None:
        return {}
    if isinstance(segments, Mapping):
        result: dict[str, np.ndarray] = {}
        for name, values in segments.items():
            values = np.asarray(values, dtype=bool)
            if values.shape != (rows,):
                raise ValueError(f"segment {name!r} must have shape ({rows},)")
            result[str(name)] = values
        return result
    values = np.asarray(segments)
    if values.shape != (rows,):
        raise ValueError(f"segment ids must have shape ({rows},)")
    return {str(value): values == value for value in np.unique(values)}


def _divide_or_none(numerator: float, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator) / denominator


def _self_test() -> None:
    """Exercise cache, strict cutoffs, groups, features and streaming metrics."""
    with tempfile.TemporaryDirectory(prefix="b_rank_data_features_") as temporary:
        temporary_path = Path(temporary)
        archive_path = temporary_path / "data_B.zip"
        header = ",".join(TEST_COLUMNS)
        test_row = "1,30," + ",".join(str(10 + index % 4) for index in range(100))
        train_rows = [
            "1,10,1,0",
            "1,11,2,0",
            "2,13,3,0",
            "2,14,4,0",
            "1,10,10,1",
            "1,11,11,1",
            "2,12,12,1",
            "2,13,13,1",
            "1,10,14,1",
            "2,11,15,1",
        ]
        test = header + "\n" + test_row + "\n"
        train = ",".join(TRAIN_COLUMNS) + "\n" + "\n".join(train_rows) + "\n"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for scene in sorted(SCENES):
                archive.writestr(f"{scene}/train.csv", train)
                archive.writestr(f"{scene}/test.csv", test)
        cache = BDataCache.build_or_open(archive_path, "dataset3", temporary_path / "cache")
        assert cache.train_rows == len(train_rows)
        assert cache.history_end(10) == 4
        store = FeatureStore.build_or_open(cache, 10)
        features = store.features(
            np.array([1], np.uint32),
            np.array([10], np.uint32),
            np.array([[10] * 100], dtype=np.uint32),
        )
        assert features.shape == (1, 100, len(FEATURE_NAMES))
        assert features.dtype == np.float32
        assert store.source_hot_mask(np.array([1, 99], np.uint32)).tolist() == [True, False]
        warm_ids, probabilities, cold_ids, cold_metadata = _negative_pool(
            cache,
            cutoff=10,
            strategy="history_cold",
            cold_fraction=1.0,
            popularity_quantile=0.80,
            popularity_power=0.75,
            test_pool_chunk_rows=1,
            test_pool_max_dense_ids=100,
        )
        assert probabilities is None and cold_ids.tolist() == [12]
        assert cold_metadata["cold_range"] == [10, 14]
        cold_candidates = _sample_candidate_rows(
            warm_ids,
            np.array([10], dtype=np.uint32),
            4,
            np.random.default_rng(7),
            cold_pool=cold_ids,
            cold_fraction=1.0,
        )
        assert np.array_equal(cold_candidates, np.full((1, 4), 12, np.uint32))
        for strategy in ("history", "history_cold", "popularity_hard", "test_pool"):
            groups = build_split1_groups(
                cache,
                seed=7,
                sizes={"train": 1, "valid": 1, "confirm": 1},
                negative_strategy=strategy,
            )
            for name in GROUP_NAMES:
                group = groups[name]
                group.validate_positive_uniqueness()
                group_store = FeatureStore.build_or_open(cache, group.cutoff)
                batch = next(group.iter_feature_batches(group_store, batch_rows=1))
                assert batch.features.shape == (1, 100, len(FEATURE_NAMES))
            if strategy == "history_cold":
                metadata = groups.metadata
                assert metadata["negative_pool"]["calibration_stress_only"] is True
                assert "cold_negative_pool_hash" in metadata
        first_test = next(iter_test_chunks(archive_path, "dataset3", chunk_rows=1))
        assert first_test.candidates.shape == (1, 100)
        score = np.zeros((2, 3), dtype=np.float32)
        score[0, 1] = 1.0
        metrics = ranking_metrics(score, np.array([1, 2]), segments={"all": [True, True]})
        assert metrics["n"] == 2 and metrics["top1"] == 0.5
        assert sha256_array(cache.src) == sha256_array(cache.src)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run a synthetic NumPy-only check")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("data_features self-test passed")
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
