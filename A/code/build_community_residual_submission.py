#!/usr/bin/env python3
"""Rebuild the frozen Jittor community-residual submission from official data.

The learned component is limited to the pinned Jittor BPR user embeddings.
Dataset1 support and Dataset2 exact-group transforms are deterministic
score-space postprocessing of the frozen base result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import jittor as jt
import numpy as np


CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "dataset1"))
sys.path.insert(0, str(CODE / "dataset2"))

import d1_source_support_postprocess as d1  # noqa: E402
import d2_exact_group_postprocess as d2  # noqa: E402


ROWS = 153_420
WIDTH = 100
ITEM_COUNT = 110_368
ITEM_KEY_BASE = ITEM_COUNT + 2
TOP_FRACTION = 0.10
EXACT_WEIGHT = 0.05
RESIDUAL_WEIGHT = 0.02
NORMALIZATION_EPS = 1e-12
PAIR_BATCH = 131_071

OFFICIAL_ARCHIVE_SHA256 = (
    "898d3cbc873a446bb372352919ec346dcc0671651ef998a699d3a23d19ef7825"
)
FROZEN_BASE_RESULT_SHA256 = (
    "4c8fca6a041a94957b04a2df9f958d76098d06ab67093679fea766c364e3a28f"
)
FROZEN_CURRENT_DATASET1_SHA256 = (
    "986a38d54c58990ba90335d319645a3ae5496c310b8e859054057b5edcbed8c6"
)
FROZEN_CURRENT_DATASET2_SHA256 = (
    "36158a8c62258dec049f79d51cf8578a4b21ce7a507120b979c82d5809890e2b"
)
PRODUCTION_CHECKPOINT_SHA256 = (
    "449c45c3d32b477efa52410f5ec9024801942e264ba79873b981297005e3d256"
)
EXPECTED_RESULT_SHA256 = (
    "d36facee996b5d45806dd6e1d80f8a48883e505f57c8d8d842b9626a50e8e7ce"
)
EXPECTED_DATASET2_SHA256 = (
    "952dd4812e2f3f394176795cea89f25f2ee16fe00d8d1cc57f69e66c6ccd3b45"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, name: str) -> str:
    actual = sha256_file(path)
    require(actual == expected, f"{name} SHA-256 differs: {actual}")
    return actual


def write_json_exclusive(path: Path, payload: dict[str, object]) -> str:
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_bytes(serialized)


def inspect_checkpoint(path: Path) -> dict[str, np.ndarray]:
    require_hash(path, PRODUCTION_CHECKPOINT_SHA256, "Jittor BPR checkpoint")
    with np.load(path, allow_pickle=False) as saved:
        required = {"kind", "lineage", "users", "user", "items", "item", "ibias", "history_end", "factors"}
        require(required.issubset(saved.files), "checkpoint keys are incomplete")
        require(str(np.asarray(saved["kind"]).item()) == "bm25_bpr", "checkpoint kind differs")
        require(
            str(np.asarray(saved["lineage"]).item()) == "d2_cross_src_community_prod_bpr32_v1",
            "checkpoint lineage differs",
        )
        require(int(np.asarray(saved["history_end"]).item()) == 1_296_345_600, "checkpoint history boundary differs")
        require(int(np.asarray(saved["factors"]).item()) == 32, "checkpoint factor count differs")
        users = np.asarray(saved["users"]).copy()
        user = np.asarray(saved["user"]).copy()
        items = np.asarray(saved["items"]).copy()
        item = np.asarray(saved["item"]).copy()
        ibias = np.asarray(saved["ibias"]).copy()
        require(users.dtype == np.int64 and users.ndim == 1 and len(users), "checkpoint users differ")
        require(np.all(users[1:] > users[:-1]), "checkpoint users are not sorted")
        require(user.shape == (len(users), 32) and user.dtype == np.float32, "checkpoint user tensor differs")
        require(items.dtype == np.int64 and items.ndim == 1 and len(items), "checkpoint items differ")
        require(item.shape == (len(items), 32) and item.dtype == np.float32, "checkpoint item tensor differs")
        require(ibias.shape == (len(items),) and ibias.dtype == np.float32, "checkpoint bias differs")
        require(np.isfinite(user).all() and np.isfinite(item).all() and np.isfinite(ibias).all(), "checkpoint is non-finite")
        require(
            np.array_equal(user, np.asarray(saved["param__user__weight"]))
            and np.array_equal(item, np.asarray(saved["param__item__weight"])),
            "checkpoint public parameter parity differs",
        )
    require_hash(path, PRODUCTION_CHECKPOINT_SHA256, "Jittor BPR checkpoint after load")
    return {"users": users, "user": user}


def sorted_distinct_occurrences(
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(candidates, axis=1, kind="stable").astype(np.uint8)
    sorted_items = np.take_along_axis(candidates, order, axis=1)
    distinct = np.empty(sorted_items.shape, dtype=bool)
    distinct[:, 0] = True
    distinct[:, 1:] = sorted_items[:, 1:] != sorted_items[:, :-1]
    row_counts = distinct.sum(axis=1, dtype=np.int32)
    rows = np.repeat(np.arange(len(candidates), dtype=np.int32), row_counts)
    columns = order[distinct].astype(np.int32, copy=False)
    items = sorted_items[distinct]
    return order, sorted_items, distinct, rows, columns, items


def restore_sorted(sorted_values: np.ndarray, order: np.ndarray) -> np.ndarray:
    output = np.empty_like(sorted_values)
    np.put_along_axis(output, order, sorted_values, axis=1)
    return output


def map_users(users: np.ndarray, src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(users, src)
    known = positions < len(users)
    known[known] &= users[positions[known]] == src[known]
    return np.minimum(positions, len(users) - 1).astype(np.int32), known


def normalize_users_jittor(vectors: np.ndarray) -> np.ndarray:
    output = np.empty_like(vectors, dtype=np.float32)
    with jt.no_grad():
        for start in range(0, len(vectors), PAIR_BATCH):
            end = min(start + PAIR_BATCH, len(vectors))
            value = jt.array(np.asarray(vectors[start:end], dtype=np.float32))
            length = jt.sqrt((value * value).sum(dim=1, keepdims=True) + NORMALIZATION_EPS)
            output[start:end] = np.asarray((value / length).data, dtype=np.float32)
    require(np.isfinite(output).all(), "Jittor normalized users are non-finite")
    return output


def cosine_pairs_jittor(normalized: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    require(left.shape == right.shape, "cosine pair shapes differ")
    output = np.empty(len(left), dtype=np.float32)
    with jt.no_grad():
        for start in range(0, len(left), PAIR_BATCH):
            end = min(start + PAIR_BATCH, len(left))
            left_value = jt.array(normalized[left[start:end]])
            right_value = jt.array(normalized[right[start:end]])
            output[start:end] = np.asarray((left_value * right_value).sum(dim=1).data, dtype=np.float32)
    require(np.isfinite(output).all(), "Jittor cosine values are non-finite")
    return output


def propagate_duplicate_presence(candidates: np.ndarray, representative: np.ndarray) -> np.ndarray:
    order = np.argsort(candidates, axis=1, kind="stable").astype(np.uint8)
    sorted_items = np.take_along_axis(candidates, order, axis=1)
    sorted_representative = np.take_along_axis(representative, order, axis=1)
    sorted_presence = np.empty_like(sorted_representative)
    sorted_presence[:, 0] = sorted_representative[:, 0]
    for column in range(1, WIDTH):
        same = sorted_items[:, column] == sorted_items[:, column - 1]
        sorted_presence[:, column] = np.where(same, sorted_presence[:, column - 1], sorted_representative[:, column])
    return restore_sorted(sorted_presence, order)


def community_presence(
    src: np.ndarray, times: np.ndarray, candidates: np.ndarray, model: dict[str, np.ndarray]
) -> tuple[np.ndarray, float, dict[str, object]]:
    require(np.all(times[1:] >= times[:-1]), "official Dataset2 test is not time-stable")
    _, _, _, occurrence_rows, occurrence_columns, occurrence_items = sorted_distinct_occurrences(candidates)
    _, time_group = np.unique(times, return_inverse=True)
    occurrence_keys = time_group[occurrence_rows].astype(np.int64) * ITEM_KEY_BASE + occurrence_items.astype(np.int64)
    occurrence_cells = occurrence_rows.astype(np.int64) * WIDTH + occurrence_columns
    user_index, known_user = map_users(model["users"], src)
    normalized = normalize_users_jittor(model["user"])
    sort_index = np.argsort(occurrence_keys, kind="stable")
    keys = occurrence_keys[sort_index]
    rows = occurrence_rows[sort_index]
    cells = occurrence_cells[sort_index].astype(np.int32, copy=False)
    sorted_src = src[rows]
    sorted_user = user_index[rows]
    sorted_known = known_user[rows]
    boundaries = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    collision_sizes = np.diff(np.r_[0, boundaries, len(keys)])
    maximum_collision = int(collision_sizes.max(initial=0))
    generic_pairs = int(np.sum(collision_sizes.astype(np.int64) * (collision_sizes - 1) // 2))
    same_source_excluded = 0
    unknown_excluded = 0
    left_cells: list[np.ndarray] = []
    right_cells: list[np.ndarray] = []
    cosine_parts: list[np.ndarray] = []
    for offset in range(1, maximum_collision):
        same_key = keys[offset:] == keys[:-offset]
        if not np.any(same_key):
            continue
        left_positions = np.flatnonzero(same_key)
        right_positions = left_positions + offset
        different_source = sorted_src[right_positions] != sorted_src[left_positions]
        same_source_excluded += int(np.sum(~different_source))
        known_pair = sorted_known[left_positions] & sorted_known[right_positions]
        unknown_excluded += int(np.sum(different_source & ~known_pair))
        valid = different_source & known_pair
        if not np.any(valid):
            continue
        left_positions = left_positions[valid]
        right_positions = right_positions[valid]
        left_cells.append(cells[left_positions].copy())
        right_cells.append(cells[right_positions].copy())
        cosine_parts.append(cosine_pairs_jittor(normalized, sorted_user[left_positions], sorted_user[right_positions]))
    require(bool(cosine_parts), "no known cross-source candidate collision pairs")
    left = np.concatenate(left_cells).astype(np.int32, copy=False)
    right = np.concatenate(right_cells).astype(np.int32, copy=False)
    cosine = np.concatenate(cosine_parts).astype(np.float32, copy=False)
    require(len(cosine) <= generic_pairs, "known pairs exceed generic pairs")
    threshold = float(np.quantile(cosine, 1.0 - TOP_FRACTION, method="higher"))
    accepted = cosine >= np.float32(threshold)
    representative = np.zeros(candidates.shape, dtype=bool)
    representative.ravel()[left[accepted]] = True
    representative.ravel()[right[accepted]] = True
    presence = propagate_duplicate_presence(candidates, representative)
    diagnostics = {
        "known_cross_source_pair_count": int(len(cosine)),
        "accepted_pair_count": int(np.sum(accepted)),
        "positive_cell_rate": float(np.mean(presence)),
        "positive_row_rate": float(np.mean(np.any(presence, axis=1))),
        "generic_distinct_row_pair_count": generic_pairs,
        "same_source_pair_count_excluded": same_source_excluded,
        "unknown_user_pair_count_excluded": unknown_excluded,
        "maximum_time_item_collision_rows": maximum_collision,
    }
    return presence, threshold, diagnostics


def canonical_rows(probability: np.ndarray):
    for row in probability:
        yield (",".join(f"{value:.8f}" for value in row) + "\n").encode("ascii")


def streaming_sha256(probability: np.ndarray) -> str:
    digest = hashlib.sha256()
    for row in canonical_rows(probability):
        digest.update(row)
    return digest.hexdigest()


def zip_info(name: str) -> zipfile.ZipInfo:
    value = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    value.compress_type = zipfile.ZIP_DEFLATED
    value.create_system = 3
    value.create_version = 20
    value.extract_version = 20
    value.flag_bits = 0
    value.volume = 0
    value.internal_attr = 0
    value.external_attr = 0o600 << 16
    value.extra = b""
    value.comment = b""
    return value


def write_result_exclusive(output: Path, dataset1: bytes, probability: np.ndarray) -> str:
    require(not output.exists(), f"refusing to overwrite result: {output}")
    require(sha256_bytes(dataset1) == FROZEN_CURRENT_DATASET1_SHA256, "Dataset1 output differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".community_repro_", suffix=".zip.tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    expected_d2 = streaming_sha256(probability)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.comment = b""
            archive.writestr(zip_info("dataset1.csv"), dataset1)
            with archive.open(zip_info("dataset2.csv"), "w") as handle:
                for row in canonical_rows(probability):
                    handle.write(row)
        with zipfile.ZipFile(temporary, "r") as archive:
            require(archive.testzip() is None, "candidate ZIP CRC check failed")
            require(archive.namelist() == ["dataset1.csv", "dataset2.csv"], "candidate ZIP members differ")
            require(sha256_bytes(archive.read("dataset1.csv")) == FROZEN_CURRENT_DATASET1_SHA256, "candidate Dataset1 differs")
            require(sha256_bytes(archive.read("dataset2.csv")) == expected_d2, "candidate Dataset2 differs")
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as destination, temporary.open("rb") as source:
            shutil.copyfileobj(source, destination, length=1 << 20)
            destination.flush()
            os.fsync(destination.fileno())
        return sha256_file(output)
    finally:
        temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict[str, object]:
    require(not args.audit_output.exists(), f"refusing to overwrite audit: {args.audit_output}")
    require(not args.output.exists(), f"refusing to overwrite result: {args.output}")
    if args.data.is_file():
        require_hash(args.data, OFFICIAL_ARCHIVE_SHA256, "official archive")
    require_hash(args.baseline, FROZEN_BASE_RESULT_SHA256, "frozen base result")
    model = inspect_checkpoint(args.checkpoint)
    require(bool(jt.has_cuda), "Jittor CUDA is required by the locked reconstruction launcher")
    jt.flags.use_cuda = 1

    _, rule_hash = d1.load_rule(args.d1_config)
    d1_source, d1_candidates, d1_test_hash = d1.load_test(args.data)
    frozen_d1, frozen_d2, d1_base_probability, base_hash = d1.load_frozen_result(args.baseline)
    require(d1.serialize_probability(d1_base_probability) == frozen_d1, "base Dataset1 CSV is not canonical")
    d1_support, d1_winner, d1_maximum, d1_gap = d1.source_support(d1_source, d1_candidates)
    d1_probability, d1_changed = d1.build_candidate(d1_base_probability, d1_winner, d1_maximum, d1_gap)
    candidate_d1 = d1.serialize_probability(d1_probability)
    require(sha256_bytes(candidate_d1) == FROZEN_CURRENT_DATASET1_SHA256, "reconstructed Dataset1 hash differs")
    require(np.array_equal(d1.serialized_row_differences(frozen_d1, candidate_d1), d1_changed), "Dataset1 serialization changed unexpectedly")

    d2_d1, base_probability, d2_base_hash = d2.load_frozen_result(args.baseline)
    require(d2_d1 == frozen_d1 and d2_base_hash == base_hash, "base component closures differ")
    src, times, candidates, d2_test_hash = d2.load_test(args.data)
    exact_support = d2.exact_other_row_support(src, times, candidates)
    reference_logits = d2.qnorm(np.log(np.clip(base_probability, 1e-12, None)))
    reference_logits += np.float32(EXACT_WEIGHT) * d2.qnorm((exact_support > 0).astype(np.float32))
    reference_probability = d2.softmax(reference_logits)
    require(streaming_sha256(reference_probability) == FROZEN_CURRENT_DATASET2_SHA256, "exact reference parity differs")

    presence, threshold, community = community_presence(src, times, candidates, model)
    signal = d2.qnorm(presence.astype(np.float32))
    candidate_logits = reference_logits + np.float32(RESIDUAL_WEIGHT) * signal
    candidate_probability = d2.softmax(candidate_logits)
    require(np.isfinite(candidate_probability).all() and np.all(candidate_probability >= 0.0), "candidate probabilities are invalid")
    candidate_d2_hash = streaming_sha256(candidate_probability)
    require(candidate_d2_hash == EXPECTED_DATASET2_SHA256, "reconstructed Dataset2 hash differs")

    result_hash = write_result_exclusive(args.output, candidate_d1, candidate_probability)
    require(result_hash == EXPECTED_RESULT_SHA256, "reconstructed ZIP bytes differ")
    report = {
        "kind": "jittor_community_residual_minimal_reproduction_build_v1",
        "decision": "PASS",
        "deep_learning_framework": "Jittor",
        "official_labels_opened": False,
        "configuration_retuned": False,
        "inputs": {
            "official_archive_sha256": OFFICIAL_ARCHIVE_SHA256 if args.data.is_file() else None,
            "frozen_base_result_sha256": base_hash,
            "production_checkpoint_sha256": PRODUCTION_CHECKPOINT_SHA256,
            "dataset1_test_sha256": d1_test_hash,
            "dataset2_test_sha256": d2_test_hash,
            "dataset1_rule_sha256": rule_hash,
        },
        "formula": "qnorm(log(base))+0.05*qnorm(exact>0)+0.02*qnorm(cross_source_community_top10pct)",
        "community": {"cosine_threshold": threshold, **community},
        "output": {
            "path": str(args.output),
            "sha256": result_hash,
            "dataset1_sha256": sha256_bytes(candidate_d1),
            "dataset2_sha256": candidate_d2_hash,
            "dataset1_changed_rows": int(d1_changed.sum()),
            "dataset2_top1_changed_rows": int(np.sum(np.argmax(reference_probability, axis=1) != np.argmax(candidate_probability, axis=1))),
        },
        "environment": {"jittor": jt.__version__, "jittor_use_cuda": bool(jt.flags.use_cuda), "numpy": np.__version__},
    }
    write_json_exclusive(args.audit_output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the frozen Jittor community residual submission")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--d1-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args)
    print(json.dumps({"decision": report["decision"], "result_sha256": report["output"]["sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
