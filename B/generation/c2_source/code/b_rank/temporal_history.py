#!/usr/bin/env python3
"""Bounded causal history tensors for Dataset3 and Dataset4.

This module is intentionally NumPy-only.  It consumes *training* edge arrays
already held by the caller and never opens a CSV/ZIP file, reads labels, or
writes test-sized arrays.  A Jittor temporal ranker can pass each bounded
training or official-test query chunk to :meth:`TemporalHistory.lookup` and
receive the following padding-zero tensors:

``source_indices``
    ``(B,)`` dense source ids.
``candidate_indices``
    ``(B, C)`` dense destination/item ids.
``history_item_indices``
    ``(B, L)`` most-recent source history, right-aligned in chronological
    order (the newest valid item is at the right).
``log_time_deltas``
    ``(B, L)`` ``log1p(query_time - history_time)`` for valid history items,
    otherwise zero.

``history_mask`` is included as an explicit attention mask.  Index zero is
reserved for padding/unknown values, so it is also recoverable from
``history_item_indices != 0``.

Use ``id_mode='shared'`` for Dataset3 when source and destination ids share a
node embedding table.  Use ``id_mode='bipartite'`` for Dataset4 to keep source
and item embedding tables separate.  Both modes are explicit configuration;
the module does not infer a scene or inspect test data.

The index is exact: a history edge is returned only when
``edge_time < query_time``.  Equal-time edges are excluded together.  For a
causal validation boundary, pass ``cutoff`` to :meth:`build`; with no supplied
vocabulary, both the index and its vocabularies then use only training edges
with ``time < cutoff``.
For a train/valid/confirm protocol, export ``train_index.vocabulary`` and
pass it as ``vocabulary=...`` to later indexes.  This keeps Jittor embedding
rows stable; ids first seen after the training cutoff remain zero/OOV, and an
OOV history item is masked rather than treated as a learned item.

Complexity
----------
For ``N`` selected training edges, construction is ``O(N log N)`` time and
``O(N + U + I)`` persistent memory, where ``U`` and ``I`` are source and item
vocabulary sizes.  A ``B``-row, ``C``-candidate query chunk costs
``O(B log N + B log U + BC log I + BL)`` time and ``O(B(C + L))`` bounded
working memory.  The Dataset4 audit has 16,408,399 train edges: this index's
``uint64`` search keys plus ``uint32`` item/time arrays occupy about 250 MiB;
construction additionally needs one ``intp`` permutation (about 125 MiB),
apart from caller-owned train memmaps.  Dataset4 test has 232,253,800
candidate cells, so callers must keep using chunks (for example ``B=8192``)
and must never concatenate test rows into a full candidate matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np


ID_MODES = frozenset(("shared", "bipartite"))
UINT32_MAX = np.iinfo(np.uint32).max
INT32_MAX = np.iinfo(np.int32).max


class TemporalHistoryError(ValueError):
    """Raised when temporal-history inputs violate the causal data contract."""


@dataclass(frozen=True)
class TemporalBatch:
    """Dense, bounded inputs for a candidate-conditioned temporal attention model.

    All dense ids use ``int32`` because Jittor embedding lookup accepts integer
    tensors and Dataset3/4 vocabularies fit in signed 32-bit range.  Zero is a
    shared padding/unknown index.  Histories are right-aligned chronological
    sequences, so a masked attention layer can preserve recency order directly.
    """

    source_indices: np.ndarray
    candidate_indices: np.ndarray
    history_item_indices: np.ndarray
    log_time_deltas: np.ndarray
    history_mask: np.ndarray


def _as_uint32(values: Any, name: str, ndim: int) -> np.ndarray:
    """Validate an integer array without copying official uint32 memmaps."""
    array = np.asarray(values)
    if array.ndim != ndim:
        raise TemporalHistoryError(f"{name} must be {ndim}-dimensional, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TemporalHistoryError(f"{name} must use an integer dtype, got {array.dtype}")
    if array.size:
        if np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
            raise TemporalHistoryError(f"{name} must not contain negative ids or times")
        if int(array.max()) > UINT32_MAX:
            raise TemporalHistoryError(f"{name} contains a value outside uint32")
    return array.astype(np.uint32, copy=False)


def _as_cutoff(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, np.integer)):
        raise TemporalHistoryError("cutoff must be an integer timestamp or None")
    if value < 0 or value > UINT32_MAX:
        raise TemporalHistoryError("cutoff must be inside uint32 range")
    return int(value)


def _is_non_decreasing(values: np.ndarray) -> bool:
    return len(values) < 2 or bool(np.all(values[:-1] <= values[1:]))


def _dense_indices(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Map raw ids to 1-based dense ids; unseen raw ids become zero."""
    flat = values.reshape(-1)
    output = np.zeros(flat.shape, dtype=np.int32)
    if len(flat) and len(ids):
        positions = np.searchsorted(ids, flat)
        inside = positions < len(ids)
        matched = np.zeros(flat.shape, dtype=bool)
        matched[inside] = ids[positions[inside]] == flat[inside]
        output[matched] = positions[matched].astype(np.int32, copy=False) + 1
    return output.reshape(values.shape)


def _query_arrays(
    source: Any, time: Any, candidates: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = _as_uint32(source, "source", 1)
    time = _as_uint32(time, "time", 1)
    candidates = _as_uint32(candidates, "candidates", 2)
    if len(source) != len(time) or len(source) != len(candidates):
        raise TemporalHistoryError("source, time and candidates must have matching rows")
    if candidates.shape[1] < 1:
        raise TemporalHistoryError("candidates must have at least one column")
    return source, time, candidates


def _training_arrays(
    source: Any, destination: Any, time: Any, cutoff: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and, when requested, retain exactly training edges ``time < cutoff``."""
    source = _as_uint32(source, "source", 1)
    destination = _as_uint32(destination, "destination", 1)
    time = _as_uint32(time, "time", 1)
    if len(source) != len(destination) or len(source) != len(time):
        raise TemporalHistoryError("source, destination and time must have matching rows")
    cutoff = _as_cutoff(cutoff)
    if cutoff is None:
        return source, destination, time
    if _is_non_decreasing(time):
        stop = int(np.searchsorted(time, cutoff, side="left"))
        return source[:stop], destination[:stop], time[:stop]
    keep = time < cutoff
    return source[keep], destination[keep], time[keep]


@dataclass(frozen=True)
class TemporalVocabulary:
    """Immutable 1-based embedding vocabularies exported by a train index.

    ``source_ids`` and ``item_ids`` contain sorted raw ids, without the zero
    pad/OOV entry.  Shared mode requires equal values in both arrays, while
    bipartite mode intentionally keeps them separate.
    """

    id_mode: str
    source_ids: np.ndarray
    item_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.id_mode not in ID_MODES:
            raise TemporalHistoryError(
                f"id_mode must be one of {sorted(ID_MODES)}, got {self.id_mode!r}"
            )
        source_ids = _as_uint32(self.source_ids, "source_ids", 1).copy()
        item_ids = _as_uint32(self.item_ids, "item_ids", 1).copy()
        if len(source_ids) > INT32_MAX or len(item_ids) > INT32_MAX:
            raise TemporalHistoryError("vocabulary exceeds Jittor-compatible int32 indices")
        if (len(source_ids) > 1 and np.any(source_ids[1:] <= source_ids[:-1])) or (
            len(item_ids) > 1 and np.any(item_ids[1:] <= item_ids[:-1])
        ):
            raise TemporalHistoryError("vocabulary ids must be strictly increasing")
        if self.id_mode == "shared" and not np.array_equal(source_ids, item_ids):
            raise TemporalHistoryError("shared vocabulary must use identical source and item ids")
        source_ids.setflags(write=False)
        item_ids.setflags(write=False)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "item_ids", item_ids)

    @classmethod
    def from_training_edges(
        cls,
        source: Any,
        destination: Any,
        time: Any,
        *,
        id_mode: str,
        cutoff: int | None = None,
    ) -> "TemporalVocabulary":
        """Create a stable vocabulary from allowed training edges only."""
        if id_mode not in ID_MODES:
            raise TemporalHistoryError(
                f"id_mode must be one of {sorted(ID_MODES)}, got {id_mode!r}"
            )
        source, destination, _ = _training_arrays(source, destination, time, cutoff)
        source_ids = np.unique(source).astype(np.uint32, copy=False)
        destination_ids = np.unique(destination).astype(np.uint32, copy=False)
        if id_mode == "shared":
            shared_ids = np.union1d(source_ids, destination_ids).astype(
                np.uint32, copy=False
            )
            return cls(id_mode, shared_ids, shared_ids)
        return cls(id_mode, source_ids, destination_ids)


class TemporalHistory:
    """Static training-edge index with strict ``edge_time < query_time`` lookup.

    Call :meth:`build` once from allowed training edges.  Then call
    :meth:`lookup` for a bounded training replay batch or an official-test
    chunk.  ``transform_chunk`` and ``iter_chunks`` accept any object exposing
    ``src`` (or ``source``), ``time`` and ``candidates`` attributes, which
    keeps this module independent of the CSV/cache implementation.
    """

    def __init__(
        self,
        *,
        id_mode: str,
        history_size: int,
        vocabulary: TemporalVocabulary,
        source_group_ids: np.ndarray,
        source_starts: np.ndarray,
        edge_keys: np.ndarray,
        history_items: np.ndarray,
        history_times: np.ndarray,
    ) -> None:
        self.id_mode = id_mode
        self.history_size = history_size
        self._vocabulary = vocabulary
        self._source_vocab = vocabulary.source_ids
        self._item_vocab = vocabulary.item_ids
        self._source_group_ids = source_group_ids
        self._source_starts = source_starts
        self._edge_keys = edge_keys
        self._history_items = history_items
        self._history_times = history_times
        for values in (
            self._source_vocab,
            self._item_vocab,
            self._source_group_ids,
            self._source_starts,
            self._edge_keys,
            self._history_items,
            self._history_times,
        ):
            values.setflags(write=False)

    @classmethod
    def build(
        cls,
        source: Any,
        destination: Any,
        time: Any,
        *,
        history_size: int,
        id_mode: str,
        cutoff: int | None = None,
        vocabulary: TemporalVocabulary | None = None,
        build_chunk_rows: int = 1_000_000,
    ) -> "TemporalHistory":
        """Build an immutable index from training edges only.

        ``cutoff`` keeps exactly ``time < cutoff`` before any vocabulary or
        sorting work.  It is the intended guard for temporal validation.  If
        ``vocabulary`` is supplied, its embedding rows are reused exactly;
        post-training raw ids become zero/OOV.  The three input arrays may be
        NumPy memmaps; only sorted index arrays are owned by the result.
        """
        if id_mode not in ID_MODES:
            raise TemporalHistoryError(
                f"id_mode must be one of {sorted(ID_MODES)}, got {id_mode!r}"
            )
        history_size = int(history_size)
        if history_size < 1:
            raise TemporalHistoryError("history_size must be positive")
        build_chunk_rows = int(build_chunk_rows)
        if build_chunk_rows < 1:
            raise TemporalHistoryError("build_chunk_rows must be positive")

        source, destination, time = _training_arrays(source, destination, time, cutoff)

        source_group_ids = np.unique(source).astype(np.uint32, copy=False)
        if len(source_group_ids) > INT32_MAX:
            raise TemporalHistoryError("source groups exceed Jittor-compatible int32 indices")
        if vocabulary is None:
            destination_ids = np.unique(destination).astype(np.uint32, copy=False)
            if id_mode == "shared":
                shared_ids = np.union1d(source_group_ids, destination_ids).astype(
                    np.uint32, copy=False
                )
                vocabulary = TemporalVocabulary(id_mode, shared_ids, shared_ids)
            else:
                vocabulary = TemporalVocabulary(id_mode, source_group_ids, destination_ids)
        else:
            if not isinstance(vocabulary, TemporalVocabulary):
                raise TemporalHistoryError("vocabulary must be a TemporalVocabulary instance")
            if vocabulary.id_mode != id_mode:
                raise TemporalHistoryError(
                    "vocabulary id_mode must match the temporal-history id_mode"
                )
        source_vocab = vocabulary.source_ids
        item_vocab = vocabulary.item_ids

        rows = len(source)
        if not rows:
            return cls(
                id_mode=id_mode,
                history_size=history_size,
                vocabulary=vocabulary,
                source_group_ids=source_group_ids,
                source_starts=np.zeros(1, dtype=np.int64),
                edge_keys=np.empty(0, dtype=np.uint64),
                history_items=np.empty(0, dtype=np.int32),
                history_times=np.empty(0, dtype=np.uint32),
            )

        # Official training files are time-monotone, so stable source sorting
        # preserves chronological ties with less work.  The lexsort fallback
        # keeps this helper correct for a synthetic or reordered train array.
        if _is_non_decreasing(time):
            order = np.argsort(source, kind="stable")
        else:
            order = np.lexsort((time, source))
        ordered_source = source[order]
        starts = np.searchsorted(ordered_source, source_group_ids, side="left")
        source_starts = np.empty(len(source_group_ids) + 1, dtype=np.int64)
        source_starts[:-1] = starts
        source_starts[-1] = rows
        del ordered_source

        edge_keys = np.empty(rows, dtype=np.uint64)
        history_items = np.empty(rows, dtype=np.int32)
        history_times = np.empty(rows, dtype=np.uint32)
        for start in range(0, rows, build_chunk_rows):
            stop = min(start + build_chunk_rows, rows)
            positions = order[start:stop]
            ordered_source = source[positions]
            ordered_destination = destination[positions]
            ordered_time = time[positions]
            group_indices = np.searchsorted(source_group_ids, ordered_source)
            item_indices = np.searchsorted(item_vocab, ordered_destination)
            if np.any(source_group_ids[group_indices] != ordered_source):
                raise RuntimeError("temporal-history source grouping lost a training id")
            edge_keys[start:stop] = (
                (group_indices.astype(np.uint64, copy=False) + 1) << np.uint64(32)
            ) | ordered_time.astype(np.uint64, copy=False)
            item_known = item_indices < len(item_vocab)
            item_known[item_known] = (
                item_vocab[item_indices[item_known]]
                == ordered_destination[item_known]
            )
            item_block = np.zeros(stop - start, dtype=np.int32)
            item_block[item_known] = (
                item_indices[item_known].astype(np.int32, copy=False) + 1
            )
            history_items[start:stop] = item_block
            history_times[start:stop] = ordered_time
        del order

        return cls(
            id_mode=id_mode,
            history_size=history_size,
            vocabulary=vocabulary,
            source_group_ids=source_group_ids,
            source_starts=source_starts,
            edge_keys=edge_keys,
            history_items=history_items,
            history_times=history_times,
        )

    @property
    def source_vocab_size(self) -> int:
        """Embedding-table size including dense index zero."""
        return len(self._source_vocab) + 1

    @property
    def item_vocab_size(self) -> int:
        """Destination/item embedding-table size including dense index zero."""
        return len(self._item_vocab) + 1

    @property
    def history_rows(self) -> int:
        """Number of selected training edges held by this index."""
        return len(self._edge_keys)

    @property
    def vocabulary(self) -> TemporalVocabulary:
        """Return the immutable train vocabulary for a later temporal index."""
        return self._vocabulary

    def lookup(self, source: Any, time: Any, candidates: Any) -> TemporalBatch:
        """Return causal dense tensors for one bounded training or test chunk."""
        source, time, candidates = _query_arrays(source, time, candidates)
        rows = len(source)
        source_indices = _dense_indices(source, self._source_vocab)
        candidate_indices = _dense_indices(candidates, self._item_vocab)
        history_items = np.zeros((rows, self.history_size), dtype=np.int32)
        log_deltas = np.zeros((rows, self.history_size), dtype=np.float32)
        history_mask = np.zeros((rows, self.history_size), dtype=bool)
        if not rows or not self.history_rows:
            return TemporalBatch(
                source_indices,
                candidate_indices,
                history_items,
                log_deltas,
                history_mask,
            )

        # ``edge_keys`` are (1-based source-group, timestamp) pairs.  A left
        # search gives the first edge whose time is >= query time, therefore
        # every gathered edge is strictly earlier than the query.
        source_groups = _dense_indices(source, self._source_group_ids)
        query_keys = (
            source_groups.astype(np.uint64, copy=False) << np.uint64(32)
        ) | time.astype(np.uint64, copy=False)
        ends = np.searchsorted(self._edge_keys, query_keys, side="left")
        starts = np.zeros(rows, dtype=np.int64)
        known = source_groups != 0
        starts[known] = self._source_starts[source_groups[known] - 1]
        offsets = np.arange(self.history_size, 0, -1, dtype=np.int64)
        history_positions = ends[:, None] - offsets[None, :]
        causal_mask = known[:, None] & (history_positions >= starts[:, None])
        safe_positions = np.maximum(history_positions, 0)
        history_items[causal_mask] = self._history_items[safe_positions[causal_mask]]
        history_mask = causal_mask & (history_items != 0)
        history_times = self._history_times[safe_positions]
        deltas = time.astype(np.int64, copy=False)[:, None] - history_times.astype(
            np.int64, copy=False
        )
        if np.any(deltas[causal_mask] <= 0):
            raise RuntimeError("strict temporal lookup returned a non-past edge")
        log_deltas[history_mask] = np.log1p(deltas[history_mask]).astype(np.float32)
        return TemporalBatch(
            source_indices,
            candidate_indices,
            history_items,
            log_deltas,
            history_mask,
        )

    def transform_chunk(self, chunk: Any) -> TemporalBatch:
        """Transform one chunk exposing ``src``/``source``, ``time``, ``candidates``."""
        try:
            source = chunk.src
        except AttributeError as error:
            try:
                source = chunk.source
            except AttributeError:
                raise TemporalHistoryError(
                    "chunk must expose src or source, plus time and candidates attributes"
                ) from error
        try:
            return self.lookup(source, chunk.time, chunk.candidates)
        except AttributeError as error:
            raise TemporalHistoryError(
                "chunk must expose src or source, plus time and candidates attributes"
            ) from error

    def iter_chunks(self, chunks: Iterable[Any]) -> Iterator[tuple[Any, TemporalBatch]]:
        """Yield caller-owned chunks with their bounded temporal tensors."""
        for chunk in chunks:
            yield chunk, self.transform_chunk(chunk)


def _self_test() -> None:
    """Exercise shared/bipartite maps, strict ties, cutoff and chunk handling."""
    shared = TemporalHistory.build(
        np.array([10, 10, 10, 20, 10, 20], dtype=np.uint32),
        np.array([11, 12, 13, 10, 14, 15], dtype=np.uint32),
        np.array([2, 4, 4, 5, 7, 8], dtype=np.uint32),
        history_size=3,
        id_mode="shared",
    )
    batch = shared.lookup(
        np.array([10, 10, 20, 99], dtype=np.uint32),
        np.array([4, 7, 7, 9], dtype=np.uint32),
        np.array([[11, 20], [12, 99], [10, 15], [10, 11]], dtype=np.uint32),
    )
    np.testing.assert_array_equal(batch.source_indices, [1, 1, 7, 0])
    np.testing.assert_array_equal(batch.candidate_indices, [[2, 7], [3, 0], [1, 6], [1, 2]])
    np.testing.assert_array_equal(
        batch.history_item_indices,
        [[0, 0, 2], [2, 3, 4], [0, 0, 1], [0, 0, 0]],
    )
    np.testing.assert_array_equal(
        batch.history_mask,
        [[False, False, True], [True, True, True], [False, False, True], [False] * 3],
    )
    np.testing.assert_allclose(
        batch.log_time_deltas,
        [[0.0, 0.0, np.log1p(2)], [np.log1p(5), np.log1p(3), np.log1p(3)], [0.0, 0.0, np.log1p(2)], [0.0] * 3],
    )

    # Deliberately unordered input proves the lexsort fallback and same-time
    # strictness: source=1 at time=2 must not see the time=2 edge.
    bipartite = TemporalHistory.build(
        np.array([2, 1, 2, 1], dtype=np.uint32),
        np.array([100, 101, 102, 103], dtype=np.uint32),
        np.array([5, 3, 1, 2], dtype=np.uint32),
        history_size=2,
        id_mode="bipartite",
    )
    batch = bipartite.lookup(
        np.array([2, 1, 1], dtype=np.uint32),
        np.array([6, 3, 2], dtype=np.uint32),
        np.array([[100, 102, 999], [103, 101, 100], [103, 100, 999]], dtype=np.uint32),
    )
    np.testing.assert_array_equal(batch.source_indices, [2, 1, 1])
    np.testing.assert_array_equal(batch.candidate_indices, [[1, 3, 0], [4, 2, 1], [4, 1, 0]])
    np.testing.assert_array_equal(batch.history_item_indices, [[3, 1], [0, 4], [0, 0]])
    np.testing.assert_array_equal(batch.history_mask, [[True, True], [False, True], [False, False]])

    cutoff = TemporalHistory.build(
        np.array([2, 1, 2, 1], dtype=np.uint32),
        np.array([100, 101, 102, 103], dtype=np.uint32),
        np.array([5, 3, 1, 2], dtype=np.uint32),
        history_size=2,
        id_mode="bipartite",
        cutoff=3,
    )
    cutoff_batch = cutoff.lookup(
        np.array([2], dtype=np.uint32),
        np.array([6], dtype=np.uint32),
        np.array([[100, 102]], dtype=np.uint32),
    )
    np.testing.assert_array_equal(cutoff_batch.candidate_indices, [[0, 1]])
    np.testing.assert_array_equal(cutoff_batch.history_item_indices, [[0, 1]])

    # The later index has more causal history, but reuses the train-cutoff
    # embedding rows.  Destination 100 is new after the cutoff, so it stays
    # index zero and its history position is masked rather than re-numbered.
    stable_vocabulary = TemporalVocabulary.from_training_edges(
        np.array([2, 1, 2, 1], dtype=np.uint32),
        np.array([100, 101, 102, 103], dtype=np.uint32),
        np.array([5, 3, 1, 2], dtype=np.uint32),
        id_mode="bipartite",
        cutoff=3,
    )
    later = TemporalHistory.build(
        np.array([2, 1, 2, 1], dtype=np.uint32),
        np.array([100, 101, 102, 103], dtype=np.uint32),
        np.array([5, 3, 1, 2], dtype=np.uint32),
        history_size=2,
        id_mode="bipartite",
        cutoff=6,
        vocabulary=stable_vocabulary,
    )
    later_batch = later.lookup(
        np.array([2], dtype=np.uint32),
        np.array([6], dtype=np.uint32),
        np.array([[100, 102]], dtype=np.uint32),
    )
    assert later.vocabulary is stable_vocabulary
    np.testing.assert_array_equal(later_batch.candidate_indices, [[0, 1]])
    np.testing.assert_array_equal(later_batch.history_item_indices, [[1, 0]])
    np.testing.assert_array_equal(later_batch.history_mask, [[True, False]])

    class Chunk:
        src = np.array([2], dtype=np.uint32)
        time = np.array([6], dtype=np.uint32)
        candidates = np.array([[100, 102, 999]], dtype=np.uint32)

    chunk = Chunk()
    emitted = list(bipartite.iter_chunks([chunk]))
    assert len(emitted) == 1 and emitted[0][0] is chunk
    np.testing.assert_array_equal(
        emitted[0][1].history_item_indices, [[3, 1]]
    )

    class FeatureChunk:
        source = np.array([1], dtype=np.uint32)
        time = np.array([3], dtype=np.uint32)
        candidates = np.array([[103, 101, 100]], dtype=np.uint32)

    np.testing.assert_array_equal(
        bipartite.transform_chunk(FeatureChunk()).history_item_indices, [[0, 4]]
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run NumPy synthetic checks")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is provided; import TemporalHistory from model code")
    _self_test()
    print("temporal_history self-test: PASS")


if __name__ == "__main__":
    _main()
