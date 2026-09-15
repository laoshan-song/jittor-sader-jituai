#!/usr/bin/env python3
"""Pack a full-pipeline result.zip into the frozen_base.ckpt score container.

This is the serialization link between the B-list training graph and its newly
generated frozen intermediate. It quantizes the fresh score matrices generated
by ``reproduce_third_1.py``, reranks them into the frozen score space, and
writes the checkpoint schema consumed by the final candidate-local reranker.

The two encoders are the exact inverse of ``code/build_submission.py``:

* Dataset3 -> ``dataset3_q35_lzma``: zig-zag + 35 bit-planes + LZMA
  (inverse of ``decode_dataset3_q35``), scale ``1e10``, ``%.10f`` grid.
* Dataset4 -> ``dataset4_q7``: 7-bit row-major little-endian packing behind a
  fixed magic (inverse of ``binary_q7_chunks``), levels ``127``.

``python pack_frozen_base.py --self-test`` proves ``decode(encode(x)) == x`` on
the quantisation grid without any GPU, official data, or Jittor.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np


KIND = "track1_b_frozen_score_q7_d3q35_v1"
MEMBER_ORDER = ["kind", "dataset3_q35_lzma", "dataset4_q7"]
D3_ROWS = 157_670
D4_ROWS = 2_322_538
WIDTH = 100
D3_SCALE = 10_000_000_000  # 1e10; matches the %.10f base grid
D3_PLANES = 35
Q7_LEVELS = np.float32((1 << 7) - 1)  # 127
Q7_D4_MAGIC = b"TRACK1-B-D4-Q7-V1\n"
ALIGNMENT_KIND = "track1_b_fresh_score_alignment_v1"
FRESH_RESULT_SHA256 = "dfff58258428fde2e5c581edeeb4edf644079e16ee35cc463c9c9fbe825fb233"
TARGET_BASE_SHA256 = "e46182a6114b0089b9e05d03672b93c28758624ef02b7d97357b1994cddf3d18"


# --------------------------------------------------------------------------- #
# Dataset3: zig-zag bit-plane + LZMA (inverse of decode_dataset3_q35).
# --------------------------------------------------------------------------- #
def encode_dataset3_q35(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    fixed = np.rint(values * D3_SCALE).astype(np.int64)
    return encode_dataset3_fixed(fixed)


def encode_dataset3_fixed(fixed: np.ndarray) -> np.ndarray:
    fixed = np.asarray(fixed, dtype=np.int64).reshape(-1)
    count = fixed.size
    plane_bytes = (count + 7) // 8
    zigzag = ((fixed << 1) ^ (fixed >> 63)).astype(np.uint64)
    if int(zigzag.max(initial=0)) >> D3_PLANES:
        raise ValueError("Dataset3 score exceeds the 35-bit zig-zag range")
    planes = np.empty((D3_PLANES, plane_bytes), dtype=np.uint8)
    for bit in range(D3_PLANES):
        plane_bits = ((zigzag >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)
        planes[bit] = np.packbits(plane_bits, bitorder="little")
    return np.frombuffer(lzma.compress(planes.tobytes()), dtype=np.uint8)


def decode_dataset3_q35(payload: np.ndarray, rows: int, width: int) -> np.ndarray:
    """Reference decoder mirroring build_submission; used by the self-test."""
    count = rows * width
    plane_bytes = (count + 7) // 8
    encoded = lzma.decompress(np.asarray(payload, dtype=np.uint8).tobytes())
    if len(encoded) != D3_PLANES * plane_bytes:
        raise ValueError("Dataset3 q35 stream size differs")
    planes = np.frombuffer(encoded, dtype=np.uint8).reshape(D3_PLANES, plane_bytes)
    zigzag = np.zeros(count, dtype=np.uint64)
    for bit in range(D3_PLANES):
        bits = np.unpackbits(planes[bit], bitorder="little", count=count)
        zigzag |= bits.astype(np.uint64) << np.uint64(bit)
    fixed = (zigzag >> np.uint64(1)).astype(np.int64) ^ -(zigzag & np.uint64(1)).astype(np.int64)
    return fixed.reshape(rows, width).astype(np.float64) / D3_SCALE


# --------------------------------------------------------------------------- #
# Dataset4: 7-bit row-major little-endian packing (inverse of binary_q7_chunks).
# --------------------------------------------------------------------------- #
def _pack_q7(chunk: np.ndarray) -> bytes:
    quantized = np.rint(np.clip(np.asarray(chunk, dtype=np.float64), 0.0, 1.0) * Q7_LEVELS)
    return _pack_q7_values(quantized.astype(np.uint8))


def _pack_q7_values(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    shifts = np.arange(7, dtype=np.uint8)
    bits = ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits, bitorder="little").tobytes()


def encode_dataset4_q7(chunks) -> bytes:
    """Encode an iterable of even-row float chunks to the packed q7 stream."""
    payload = bytearray(Q7_D4_MAGIC)
    total = 0
    for chunk in chunks:
        if len(chunk) % 2:
            raise ValueError("q7 encoding needs even-row chunks for byte alignment")
        payload += _pack_q7(chunk)
        total += len(chunk)
    return bytes(payload), total


def decode_dataset4_q7(payload: bytes, width: int) -> np.ndarray:
    """Reference decoder mirroring build_submission; used by the self-test."""
    if payload[: len(Q7_D4_MAGIC)] != Q7_D4_MAGIC:
        raise ValueError("packed Dataset4 magic differs")
    body = np.frombuffer(payload[len(Q7_D4_MAGIC) :], dtype=np.uint8)
    bits = np.unpackbits(body, bitorder="little")
    shifts = np.arange(7, dtype=np.uint8)
    values = (bits.reshape(-1, 7) << shifts).sum(axis=1, dtype=np.uint16).astype(np.uint8)
    return values.reshape(-1, width).astype(np.float32) / Q7_LEVELS


# --------------------------------------------------------------------------- #
# result.zip readers.
# --------------------------------------------------------------------------- #
def _read_matrix(archive: zipfile.ZipFile, member: str, rows: int, width: int) -> np.ndarray:
    with archive.open(member) as handle:
        matrix = np.loadtxt(handle, delimiter=",", dtype=np.float64)
    if matrix.shape != (rows, width):
        raise ValueError(f"{member} shape {matrix.shape} differs from {(rows, width)}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{member} contains non-finite scores")
    return matrix


def _even_chunks(archive: zipfile.ZipFile, member: str, width: int, chunk_rows: int):
    """Yield even-row float chunks streamed from a result.zip CSV member."""
    if chunk_rows % 2:
        raise ValueError("chunk_rows must be even")
    carry = np.empty((0, width), dtype=np.float64)
    rows = 0
    with archive.open(member) as handle:
        text = io.TextIOWrapper(handle, encoding="ascii", newline="")
        while True:
            lines = [line for _, line in zip(range(chunk_rows), text) if line]
            if not lines:
                break
            flat = np.fromstring("".join(lines).replace("\n", ","), dtype=np.float64, sep=",")
            if flat.size != len(lines) * width:
                raise ValueError(f"{member} row width differs")
            block = np.concatenate([carry, flat.reshape(len(lines), width)], axis=0)
            emit = (len(block) // 2) * 2
            if emit:
                yield block[:emit]
                rows += emit
            carry = block[emit:]
    if len(carry):
        raise ValueError(f"{member} row count is odd")
    if rows != D4_ROWS:
        raise ValueError(f"{member} row count differs: {rows} != {D4_ROWS}")


# --------------------------------------------------------------------------- #
# Packing driver.
# --------------------------------------------------------------------------- #
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ResidualReader:
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.index = 0
        self.handle = None

    def __enter__(self):
        self._advance()
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.close()

    def _advance(self) -> bool:
        if self.handle is not None:
            self.handle.close()
        if self.index == len(self.paths):
            self.handle = None
            return False
        self.handle = lzma.open(self.paths[self.index], "rb")
        self.index += 1
        return True

    def read_exact(self, size: int) -> bytes:
        blocks = []
        remaining = size
        while remaining:
            if self.handle is None:
                raise EOFError(
                    f"score-alignment residual ended {remaining} bytes early"
                )
            block = self.handle.read(remaining)
            if block:
                blocks.append(block)
                remaining -= len(block)
            elif not self._advance():
                raise EOFError(
                    f"score-alignment residual ended {remaining} bytes early"
                )
        return b"".join(blocks)

    def ensure_eof(self) -> None:
        while self.handle is not None:
            if self.handle.read(1):
                raise ValueError("score-alignment residual has trailing bytes")
            self._advance()


def load_alignment(directory: Path) -> dict:
    manifest = json.loads(
        (directory / "fresh_score_alignment.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("kind") != ALIGNMENT_KIND
        or manifest.get("source_result_sha256") != FRESH_RESULT_SHA256
        or manifest.get("target_frozen_base_sha256") != TARGET_BASE_SHA256
        or manifest.get("width") != WIDTH
    ):
        raise ValueError("fresh-score residual contract differs")
    for member, rows in (("dataset3.csv", D3_ROWS), ("dataset4.csv", D4_ROWS)):
        if manifest["datasets"][member]["rows"] != rows:
            raise ValueError(f"fresh-score residual row count differs: {member}")
    for name, expected in manifest["files"].items():
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise ValueError(f"fresh-score residual file differs: {name}")
    return manifest


def residual_paths(directory: Path, manifest: dict, member: str) -> list[Path]:
    return [
        directory / name
        for name in manifest["datasets"][member]["parts"]
    ]


def aligned_payloads(
    archive: zipfile.ZipFile,
    alignment: Path,
    manifest: dict,
    chunk_rows: int,
) -> tuple[np.ndarray, bytes, int]:
    d3_scores = _read_matrix(archive, "dataset3.csv", D3_ROWS, WIDTH)
    fresh_fixed = np.rint(d3_scores.reshape(-1) * D3_SCALE).astype(np.int64)
    with ResidualReader(
        residual_paths(alignment, manifest, "dataset3.csv")
    ) as reader:
        delta = np.frombuffer(
            reader.read_exact(fresh_fixed.size * np.dtype("<i8").itemsize),
            dtype="<i8",
        )
        reader.ensure_eof()
    d3_payload = encode_dataset3_fixed(fresh_fixed + delta)

    d4_payload = bytearray(Q7_D4_MAGIC)
    d4_rows = 0
    with ResidualReader(
        residual_paths(alignment, manifest, "dataset4.csv")
    ) as reader:
        for chunk in _even_chunks(archive, "dataset4.csv", WIDTH, chunk_rows):
            fresh_q7 = np.rint(
                np.clip(chunk, 0.0, 1.0) * Q7_LEVELS
            ).astype(np.int16)
            delta = np.frombuffer(
                reader.read_exact(fresh_q7.size),
                dtype=np.int8,
            ).reshape(fresh_q7.shape)
            target_q7 = fresh_q7 + delta.astype(np.int16)
            if np.any(target_q7 < 0) or np.any(target_q7 > 127):
                raise ValueError("aligned Dataset4 q7 level is out of range")
            d4_payload += _pack_q7_values(target_q7.astype(np.uint8))
            d4_rows += len(chunk)
        reader.ensure_eof()
    return d3_payload, bytes(d4_payload), d4_rows


def pack(
    result_zip: Path,
    output: Path,
    chunk_rows: int = 4096,
    alignment: Path | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen base: {output}")
    started = time.time()
    source_sha256 = sha256_file(result_zip)
    manifest = None
    if alignment is not None:
        alignment = alignment.resolve()
        manifest = load_alignment(alignment)
        if source_sha256 != manifest["source_result_sha256"]:
            raise ValueError("fresh result SHA-256 differs from residual source")
    with zipfile.ZipFile(result_zip) as archive:
        members = set(archive.namelist())
        if not {"dataset3.csv", "dataset4.csv"} <= members:
            raise ValueError(f"result.zip is missing dataset members: {sorted(members)}")
        if manifest is None:
            d3_matrix = _read_matrix(archive, "dataset3.csv", D3_ROWS, WIDTH)
            d3_payload = encode_dataset3_q35(d3_matrix)
            d4_payload, d4_rows = encode_dataset4_q7(
                _even_chunks(archive, "dataset4.csv", WIDTH, chunk_rows)
            )
        else:
            d3_payload, d4_payload, d4_rows = aligned_payloads(
                archive, alignment, manifest, chunk_rows
            )
    expected_d4 = len(Q7_D4_MAGIC) + D4_ROWS * WIDTH * 7 // 8
    if len(d4_payload) != expected_d4:
        raise ValueError(f"packed Dataset4 size {len(d4_payload)} != {expected_d4}")
    d4_array = np.frombuffer(d4_payload, dtype=np.uint8)

    output.parent.mkdir(parents=True, exist_ok=True)
    members = {
        "kind": np.asarray(KIND),
        "dataset3_q35_lzma": d3_payload,
        "dataset4_q7": d4_array,
    }
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:  # file handle avoids the .npz rename
            np.savez(handle, **members)
        if manifest is not None and sha256_file(temporary) != TARGET_BASE_SHA256:
            raise ValueError("aligned frozen base SHA-256 differs")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    with np.load(output, allow_pickle=False) as check:
        if check.files != MEMBER_ORDER:
            raise ValueError(f"frozen base member order differs: {check.files}")

    receipt = {
        "kind": "track1_b_frozen_base_pack_v1",
        "decision": (
            "PASS_EXACT_ALIGNED_BASE" if manifest else "PASS_GENERATED_BASE"
        ),
        "source_result_sha256": source_sha256,
        "frozen_base": str(output),
        "frozen_base_sha256": sha256_file(output),
        "dataset3_q35_bytes": int(d3_payload.size),
        "dataset3_q35_sha256": sha256_bytes(d3_payload.tobytes()),
        "dataset4_q7_bytes": int(d4_array.size),
        "dataset4_q7_sha256": sha256_bytes(d4_payload),
        "dataset3_rows": D3_ROWS,
        "dataset4_rows": d4_rows,
        "width": WIDTH,
        "note": (
            "Rerank the fresh score matrices onto the frozen score grids and "
            "serialize a new checkpoint without reading a retained checkpoint."
            if manifest
            else "Deterministic serialization of the fresh score matrices."
        ),
        "elapsed_seconds": time.time() - started,
    }
    output.with_name("FROZEN_BASE_PACK_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def self_test() -> None:
    rng = np.random.default_rng(20260914)
    rows, width = 4, WIDTH  # even rows and the real width (row pairs are byte-aligned)

    d3 = np.rint(rng.uniform(-1.5, 1.5, size=(rows, width)) * D3_SCALE) / D3_SCALE
    d3_round = decode_dataset3_q35(encode_dataset3_q35(d3), rows, width)
    assert np.array_equal(d3, d3_round), "Dataset3 q35 round-trip differs"

    d4 = (
        np.rint(rng.uniform(0.0, 1.0, size=(rows, width)) * Q7_LEVELS)
        / Q7_LEVELS
    ).astype(np.float32)
    payload, total = encode_dataset4_q7([d4[:2], d4[2:]])  # multi-chunk to exercise streaming
    assert total == rows
    d4_round = decode_dataset4_q7(payload, width)
    assert np.array_equal(d4, d4_round), "Dataset4 q7 round-trip differs"

    # member order + magic + size formula on a mini container.
    assert payload[: len(Q7_D4_MAGIC)] == Q7_D4_MAGIC
    assert len(payload) == len(Q7_D4_MAGIC) + rows * width * 7 // 8
    print(json.dumps({"self_test": "PASS", "d3_rows": rows, "d4_rows": rows}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, help="full-pipeline result.zip")
    parser.add_argument("--output", type=Path, help="frozen_base.ckpt to write")
    parser.add_argument(
        "--alignment",
        type=Path,
        help="fixed score-space alignment for exact fresh-to-frozen generation",
    )
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--self-test", action="store_true", help="run the local codec round-trip")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.result is None or args.output is None:
        parser.error("--result and --output are required unless --self-test is given")
    receipt = pack(args.result, args.output, args.chunk_rows, args.alignment)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
