#!/usr/bin/env python3
"""Build the recorded B-list submission from a frozen base and Jittor MF residual."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
import shutil
import time
import zipfile
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd

from model import bounded_residual, configure_cuda, load_checkpoint, predict_scores


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
BASE_SHA256 = "e46182a6114b0089b9e05d03672b93c28758624ef02b7d97357b1994cddf3d18"
MODEL_SHA256 = "98dc703a0851229f38b43f588b709c1b1aeff98ab60570a1ca61d8e617eb31f4"
TARGET_SHA256 = "9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba"
TARGET_D3_SHA256 = "08b2288d63cc265d52ce474cfda2897b1d359b6deb402b73a20e90cc4959a59d"
TARGET_D4_SHA256 = "662df72f1df61198ea78787fe5502948a79de918d8c831494df2736b378e5311"
ROWS = 2_322_538
D3_ROWS = 157_670
WIDTH = 100
Q7_D4_MEMBER = "dataset4.q7"
Q7_D4_MAGIC = b"TRACK1-B-D4-Q7-V1\n"
D3_BASE_SHA256 = "35f416ef441e8ffefe8492da9c0bec8e8bd8218b73e87918e25837ba1140c53d"
D3_BASE_BYTES = 206_768_126
Q7_LEVELS = np.float32((1 << 7) - 1)
RESIDUAL_WEIGHT = np.float32(0.02)
D3_RESIDUAL_WEIGHT = np.float32(0.005)
RANK_GRID = np.linspace(1.0, 0.0, WIDTH, dtype=np.float32)


class FrozenBase:
    """Read either the legacy ZIP or the locked NumPy checkpoint container."""

    def __init__(self, path: Path, *, verify_locked: bool = True):
        self.path = path
        self.verify_locked = verify_locked
        self.archive = None
        self.payload: dict[str, bytes] = {}

    def __enter__(self):
        if self.path.suffix == ".ckpt":
            archive = np.load(self.path, allow_pickle=False)
            if str(archive["kind"].item()) != "track1_b_frozen_score_q7_d3q35_v1":
                archive.close()
                raise ValueError("frozen checkpoint kind differs")
            if archive.files != ["kind", "dataset3_q35_lzma", "dataset4_q7"]:
                archive.close()
                raise ValueError("frozen checkpoint members differ")
            self.payload = {
                "dataset3.csv": decode_dataset3_q35(
                    archive["dataset3_q35_lzma"], verify_locked=self.verify_locked
                ),
                Q7_D4_MEMBER: np.asarray(archive["dataset4_q7"], dtype=np.uint8).tobytes(),
            }
            archive.close()
        else:
            self.archive = zipfile.ZipFile(self.path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.archive is not None:
            self.archive.close()

    def namelist(self) -> list[str]:
        return list(self.archive.namelist()) if self.archive is not None else list(self.payload)

    def open(self, name: str):
        if self.archive is not None:
            return self.archive.open(name)
        try:
            return io.BytesIO(self.payload[name])
        except KeyError as exc:
            raise KeyError(f"frozen checkpoint member differs: {name}") from exc


def decode_dataset3_q35(payload: np.ndarray, *, verify_locked: bool = True) -> bytes:
    count = D3_ROWS * WIDTH
    plane_bytes = (count + 7) // 8
    encoded = lzma.decompress(np.asarray(payload, dtype=np.uint8).tobytes())
    if len(encoded) != 35 * plane_bytes:
        raise ValueError("Dataset3 q35 score stream size differs")
    planes = np.frombuffer(encoded, dtype=np.uint8).reshape(35, plane_bytes)
    zigzag = np.zeros(count, dtype=np.uint64)
    for bit in range(35):
        values = np.unpackbits(planes[bit], bitorder="little", count=count)
        zigzag |= values.astype(np.uint64) << bit
    fixed = (zigzag >> 1).astype(np.int64) ^ -(zigzag & 1).astype(np.int64)
    buffer = io.BytesIO()
    np.savetxt(
        buffer,
        fixed.reshape(D3_ROWS, WIDTH).astype(np.float64) / 10_000_000_000,
        delimiter=",",
        fmt="%.10f",
    )
    csv = buffer.getvalue()
    if verify_locked and (
        len(csv) != D3_BASE_BYTES or hashlib.sha256(csv).hexdigest() != D3_BASE_SHA256
    ):
        raise ValueError("Dataset3 q35 decoded CSV differs")
    return csv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def member_sha256(path: Path, name: str) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive, archive.open(name) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_member(name: str) -> zipfile.ZipInfo:
    member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    member.compress_type = zipfile.ZIP_DEFLATED
    return member


def item_popularity(official: zipfile.ZipFile) -> tuple[np.ndarray, np.ndarray]:
    with official.open("dataset3/train.csv") as handle:
        destination = pd.read_csv(handle, usecols=["dst"], dtype=np.uint32)["dst"].to_numpy()
    ids, counts = np.unique(destination, return_counts=True)
    return ids, np.log1p(counts).astype(np.float32)


def popularity_scores(
    candidates: np.ndarray, ids: np.ndarray, values: np.ndarray
) -> np.ndarray:
    flat = candidates.reshape(-1)
    positions = np.searchsorted(ids, flat)
    inside = positions < len(ids)
    found = np.zeros(flat.shape, dtype=bool)
    found[inside] = ids[positions[inside]] == flat[inside]
    scores = np.zeros(flat.shape, dtype=np.float32)
    scores[found] = values[positions[found]]
    return scores.reshape(candidates.shape)


def binary_q7_chunks(handle, chunk_rows: int):
    if chunk_rows % 2:
        raise ValueError("q7 chunk rows must be even")
    if handle.read(len(Q7_D4_MAGIC)) != Q7_D4_MAGIC:
        raise ValueError("packed Dataset4 base magic differs")
    pair_bytes = WIDTH * 7 * 2 // 8
    shifts = np.arange(7, dtype=np.uint8)
    total = 0
    while True:
        payload = handle.read(pair_bytes * (chunk_rows // 2))
        if not payload:
            break
        if len(payload) % pair_bytes:
            raise ValueError("packed Dataset4 base has a partial row pair")
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
        values = ((bits.reshape(-1, 7) << shifts).sum(axis=1, dtype=np.uint16))
        values = values.astype(np.uint8).reshape(-1, WIDTH)
        total += len(values)
        yield values.astype(np.float32) / Q7_LEVELS
    if handle.read(1) or total != ROWS:
        raise ValueError("packed Dataset4 base row count differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--predict-batch", type=int, default=2048)
    parser.add_argument("--unlocked", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing output-directory reuse: {args.output_dir}")
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    base_hash = sha256(args.base)
    model_hash = sha256(args.checkpoint)
    if not args.unlocked and base_hash != BASE_SHA256:
        raise ValueError("frozen base SHA-256 differs")
    if not args.unlocked and model_hash != MODEL_SHA256:
        raise ValueError("Jittor checkpoint SHA-256 differs")

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "result.zip"
    report_path = args.output_dir / "REPRODUCTION_RECEIPT.json"
    temporary = args.output_dir / ".result.zip.tmp"
    configure_cuda()
    model, source_ids, item_ids = load_checkpoint(args.checkpoint)

    candidate_columns = [f"c{index}" for index in range(1, WIDTH + 1)]
    rows = 0
    d3_rows = 0
    started = time.time()
    try:
        with (
            zipfile.ZipFile(args.data) as official,
            FrozenBase(args.base, verify_locked=not args.unlocked) as base,
            zipfile.ZipFile(
                temporary,
                "x",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as target,
        ):
            base_members = base.namelist()
            if base_members not in (["dataset3.csv", Q7_D4_MEMBER], ["dataset3.csv", "dataset4.csv"]):
                raise ValueError("frozen base members differ")
            if not args.unlocked and base_members != ["dataset3.csv", Q7_D4_MEMBER]:
                raise ValueError("locked base must use packed Dataset4 member")
            d3_ids, d3_popularity = item_popularity(official)
            with (
                official.open("dataset3/test.csv") as test_file,
                base.open("dataset3.csv") as base_file,
                target.open(fixed_member("dataset3.csv"), "w") as raw,
            ):
                test_chunks = pd.read_csv(
                    test_file,
                    usecols=candidate_columns,
                    dtype=np.uint32,
                    chunksize=args.chunk_rows,
                )
                base_chunks = pd.read_csv(
                    base_file,
                    header=None,
                    dtype=np.float64,
                    chunksize=args.chunk_rows,
                )
                for test, frozen in zip_longest(test_chunks, base_chunks):
                    if test is None or frozen is None or len(test) != len(frozen):
                        raise ValueError("official/base Dataset3 chunk boundaries differ")
                    candidates = test[candidate_columns].to_numpy(dtype=np.uint32, copy=False)
                    signal = bounded_residual(
                        popularity_scores(candidates, d3_ids, d3_popularity)
                    )
                    frozen_values = frozen.to_numpy(dtype=np.float64, copy=False)
                    combined = frozen_values + D3_RESIDUAL_WEIGHT * signal.astype(np.float64)
                    buffer = io.BytesIO()
                    np.savetxt(buffer, combined, delimiter=",", fmt="%.8f")
                    raw.write(buffer.getvalue())
                    d3_rows += len(test)

            with (
                official.open("dataset4/test.csv") as test_file,
                base.open(base_members[1]) as base_file,
                target.open(fixed_member("dataset4.csv"), "w", force_zip64=True) as raw,
            ):
                test_chunks = pd.read_csv(
                    test_file,
                    usecols=["src", *candidate_columns],
                    dtype=np.uint32,
                    chunksize=args.chunk_rows,
                )
                if base_members == ["dataset3.csv", Q7_D4_MEMBER]:
                    base_chunks = binary_q7_chunks(base_file, args.chunk_rows)
                else:
                    base_chunks = pd.read_csv(
                        base_file,
                        header=None,
                        dtype=np.float32,
                        chunksize=args.chunk_rows,
                    )
                for test, frozen in zip_longest(test_chunks, base_chunks):
                    if test is None or frozen is None or len(test) != len(frozen):
                        raise ValueError("official/base chunk boundaries differ")
                    candidates = test[candidate_columns].to_numpy(dtype=np.uint32, copy=False)
                    scores = predict_scores(
                        model,
                        source_ids,
                        item_ids,
                        test["src"].to_numpy(dtype=np.uint32, copy=False),
                        candidates,
                        batch_size=args.predict_batch,
                    )
                    correction = RESIDUAL_WEIGHT * bounded_residual(scores)
                    frozen_values = np.asarray(frozen, dtype=np.float32)
                    if frozen_values.shape != (len(test), WIDTH):
                        raise ValueError("frozen base width differs")
                    combined = frozen_values + correction
                    order = np.argsort(-combined, axis=1, kind="stable")
                    probability = np.empty(combined.shape, dtype=np.float32)
                    probability[np.arange(len(test))[:, None], order] = RANK_GRID[None, :]
                    buffer = io.BytesIO()
                    np.savetxt(buffer, probability, delimiter=",", fmt="%.8f")
                    raw.write(buffer.getvalue())
                    rows += len(test)
                    if rows % 100_000 < len(test):
                        print(json.dumps({"rows": rows, "elapsed_seconds": time.time() - started}), flush=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    if rows != ROWS or d3_rows != D3_ROWS:
        raise ValueError(
            f"output row count differs: dataset3={d3_rows}, dataset4={rows}"
        )

    output_hash = sha256(output)
    d3_hash = member_sha256(output, "dataset3.csv")
    d4_hash = member_sha256(output, "dataset4.csv")
    exact = (
        output_hash == TARGET_SHA256
        and d3_hash == TARGET_D3_SHA256
        and d4_hash == TARGET_D4_SHA256
    )
    decision = "PASS_SUPPLEMENTARY_INFERENCE" if args.unlocked else ("PASS" if exact else "FAIL")
    receipt = {
        "kind": "track1_b_reproduction_v1",
        "decision": decision,
        "run_mode": "supplementary" if args.unlocked else "recorded",
        "official_data_sha256": DATA_SHA256,
        "base_sha256": base_hash,
        "checkpoint_sha256": model_hash,
        "result_sha256": output_hash,
        "dataset3_sha256": d3_hash,
        "dataset4_sha256": d4_hash,
        "rows": rows,
        "dataset3_rows": d3_rows,
        "width": WIDTH,
        "target_environment": "Ubuntu 22.04; NVIDIA RTX 4090; CUDA 12.4; Python 3.10; Jittor 1.3.10.0",
        "elapsed_seconds": time.time() - started,
        "uses_test_labels": False,
        "uses_external_dataset": False,
    }
    if not args.unlocked:
        receipt["byte_exact_online_result"] = exact
    report_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0 if receipt["decision"] in {"PASS", "PASS_SUPPLEMENTARY_INFERENCE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
