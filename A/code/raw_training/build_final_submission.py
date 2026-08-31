#!/usr/bin/env python3
"""Apply the recorded Track 1 postprocessing to a freshly inferred base ZIP.

The learned community representation is trained and normalized with Jittor.
Candidate-only support signals are deterministic postprocessing over the
official candidate matrices; no test labels are opened by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import jittor as jt
import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT / "dataset1"))
sys.path.insert(0, str(CODE_ROOT / "dataset2"))
import d1_source_support_postprocess as d1  # noqa: E402
import d2_exact_group_postprocess as d2  # noqa: E402


ROWS = 153420
WIDTH = 100
ITEM_KEY_BASE = 110370
TOP_FRACTION = 0.10
EXACT_WEIGHT = 0.05
RESIDUAL_WEIGHT = 0.02
NORMALIZATION_EPS = 1e-12
PAIR_BATCH = 131071
EXPECTED_BASE_SHA256 = "4c8fca6a041a94957b04a2df9f958d76098d06ab67093679fea766c364e3a28f"
EXPECTED_RESULT_SHA256 = "d36facee996b5d45806dd6e1d80f8a48883e505f57c8d8d842b9626a50e8e7ce"
EXPECTED_D1_SHA256 = "986a38d54c58990ba90335d319645a3ae5496c310b8e859054057b5edcbed8c6"
EXPECTED_D2_SHA256 = "952dd4812e2f3f394176795cea89f25f2ee16fe00d8d1cc57f69e66c6ccd3b45"
EXPECTED_CHECKPOINT_SHA256 = "449c45c3d32b477efa52410f5ec9024801942e264ba79873b981297005e3d256"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def inspect_checkpoint(path: Path, *, strict_release: bool) -> dict[str, np.ndarray]:
    if strict_release:
        require(sha256_file(path) == EXPECTED_CHECKPOINT_SHA256, "community checkpoint differs")
    with np.load(path, allow_pickle=False) as saved:
        required = {"kind", "users", "user", "items", "item", "ibias", "history_end", "factors"}
        require(required.issubset(saved.files), "community checkpoint keys are incomplete")
        require(str(np.asarray(saved["kind"]).item()) == "bm25_bpr", "unexpected community checkpoint kind")
        users = np.asarray(saved["users"], dtype=np.int64).copy()
        user = np.asarray(saved["user"], dtype=np.float32).copy()
        require(users.ndim == 1 and user.ndim == 2 and user.shape[0] == len(users), "invalid community user tensors")
        require(np.all(users[1:] > users[:-1]), "community user identifiers are not sorted")
        require(np.isfinite(user).all(), "community user vectors are non-finite")
        require(int(np.asarray(saved["history_end"]).item()) == 1296345600, "community history boundary differs")
        require(int(np.asarray(saved["factors"]).item()) == user.shape[1], "community factor count differs")
    return {"users": users, "user": user}


def load_base(path: Path, *, strict_release: bool) -> tuple[bytes, np.ndarray, np.ndarray]:
    if strict_release:
        require(sha256_file(path) == EXPECTED_BASE_SHA256, "base result differs")
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, "base ZIP CRC validation failed")
        require(archive.namelist() == ["dataset1.csv", "dataset2.csv"], "base ZIP members differ")
        dataset1 = archive.read("dataset1.csv")
        dataset2 = archive.read("dataset2.csv")
    d1_probability = np.loadtxt(io.BytesIO(dataset1), delimiter=",", dtype=np.float64)
    d2_probability = np.loadtxt(io.BytesIO(dataset2), delimiter=",", dtype=np.float64)
    require(d1_probability.shape == (61051, WIDTH), "base Dataset1 shape differs")
    require(d2_probability.shape == (ROWS, WIDTH), "base Dataset2 shape differs")
    require(np.isfinite(d1_probability).all() and np.isfinite(d2_probability).all(), "base scores are non-finite")
    return dataset1, d1_probability, d2_probability


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
    require(np.isfinite(output).all(), "Jittor user normalization produced non-finite values")
    return output


def cosine_pairs_jittor(normalized: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.empty(len(left), dtype=np.float32)
    with jt.no_grad():
        for start in range(0, len(left), PAIR_BATCH):
            end = min(start + PAIR_BATCH, len(left))
            left_value = jt.array(normalized[left[start:end]])
            right_value = jt.array(normalized[right[start:end]])
            output[start:end] = np.asarray((left_value * right_value).sum(dim=1).data, dtype=np.float32)
    require(np.isfinite(output).all(), "Jittor cosine computation produced non-finite values")
    return output


def sorted_distinct_occurrences(candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(candidates, axis=1, kind="stable").astype(np.uint8)
    sorted_items = np.take_along_axis(candidates, order, axis=1)
    distinct = np.empty(sorted_items.shape, dtype=bool)
    distinct[:, 0] = True
    distinct[:, 1:] = sorted_items[:, 1:] != sorted_items[:, :-1]
    row_counts = distinct.sum(axis=1, dtype=np.int32)
    rows = np.repeat(np.arange(len(candidates), dtype=np.int32), row_counts)
    columns = order[distinct].astype(np.int32, copy=False)
    items = sorted_items[distinct]
    return order, rows, columns, items, sorted_items


def restore_sorted(sorted_values: np.ndarray, order: np.ndarray) -> np.ndarray:
    output = np.empty_like(sorted_values)
    np.put_along_axis(output, order, sorted_values, axis=1)
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


def community_presence(src: np.ndarray, times: np.ndarray, candidates: np.ndarray, model: dict[str, np.ndarray]) -> tuple[np.ndarray, float, dict[str, int | float]]:
    require(np.all(times[1:] >= times[:-1]), "Dataset2 test rows are not time-stable")
    _, occurrence_rows, occurrence_columns, occurrence_items, _ = sorted_distinct_occurrences(candidates)
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
    require(bool(cosine_parts), "no cross-source candidate pairs with known users")
    left = np.concatenate(left_cells).astype(np.int32, copy=False)
    right = np.concatenate(right_cells).astype(np.int32, copy=False)
    cosine = np.concatenate(cosine_parts).astype(np.float32, copy=False)
    threshold = float(np.quantile(cosine, 1.0 - TOP_FRACTION, method="higher"))
    accepted = cosine >= np.float32(threshold)
    representative = np.zeros(candidates.shape, dtype=bool)
    representative.ravel()[left[accepted]] = True
    representative.ravel()[right[accepted]] = True
    presence = propagate_duplicate_presence(candidates, representative)
    return presence, threshold, {
        "known_cross_source_pair_count": int(len(cosine)),
        "accepted_pair_count": int(np.sum(accepted)),
        "positive_cell_rate": float(np.mean(presence)),
        "positive_row_rate": float(np.mean(np.any(presence, axis=1))),
        "generic_distinct_row_pair_count": generic_pairs,
        "same_source_pair_count_excluded": same_source_excluded,
        "unknown_user_pair_count_excluded": unknown_excluded,
        "maximum_time_item_collision_rows": maximum_collision,
    }


def canonical_rows(probability: np.ndarray):
    for row in probability:
        yield (",".join(f"{value:.8f}" for value in row) + "\n").encode("ascii")


def probability_hash(probability: np.ndarray) -> str:
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
    return value


def write_result(output: Path, dataset1: bytes, probability: np.ndarray) -> str:
    require(not output.exists(), f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".track1_build_", suffix=".zip", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.comment = b""
            archive.writestr(zip_info("dataset1.csv"), dataset1)
            with archive.open(zip_info("dataset2.csv"), "w") as handle:
                for row in canonical_rows(probability):
                    handle.write(row)
        with zipfile.ZipFile(temporary) as archive:
            require(archive.testzip() is None, "output ZIP CRC validation failed")
            require(archive.namelist() == ["dataset1.csv", "dataset2.csv"], "output ZIP members differ")
        with temporary.open("rb") as source:
            descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as destination:
                shutil.copyfileobj(source, destination, length=1 << 20)
                destination.flush()
                os.fsync(destination.fileno())
        return sha256_file(output)
    finally:
        temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict[str, object]:
    require(not args.audit_output.exists(), f"refusing to overwrite audit: {args.audit_output}")
    dataset1, base_d1, base_d2 = load_base(args.baseline, strict_release=args.strict_release)
    model = inspect_checkpoint(args.checkpoint, strict_release=args.strict_release)
    require(bool(jt.has_cuda), "Jittor CUDA is required for the release builder")
    jt.flags.use_cuda = 1
    _, rule_hash = d1.load_rule(args.d1_config)
    d1_source, d1_candidates, d1_test_hash = d1.load_test(args.data)
    d1_support, d1_winner, d1_maximum, d1_gap = d1.source_support(d1_source, d1_candidates)
    candidate_d1, changed = d1.build_candidate(base_d1, d1_winner, d1_maximum, d1_gap)
    dataset1_output = d1.serialize_probability(candidate_d1)
    src, times, candidates, d2_test_hash = d2.load_test(args.data)
    exact_support = d2.exact_other_row_support(src, times, candidates)
    reference_logits = d2.qnorm(np.log(np.clip(base_d2, 1e-12, None)))
    reference_logits += np.float32(EXACT_WEIGHT) * d2.qnorm((exact_support > 0).astype(np.float32))
    presence, threshold, community = community_presence(src, times, candidates, model)
    candidate_logits = reference_logits + np.float32(RESIDUAL_WEIGHT) * d2.qnorm(presence.astype(np.float32))
    candidate_d2 = d2.softmax(candidate_logits)
    require(np.isfinite(candidate_d2).all() and np.all(candidate_d2 >= 0.0), "final probabilities are invalid")
    d1_hash = sha256_bytes(dataset1_output)
    d2_hash = probability_hash(candidate_d2)
    result_hash = write_result(args.output, dataset1_output, candidate_d2)
    if args.strict_release:
        require(d1_hash == EXPECTED_D1_SHA256, "Dataset1 release output differs")
        require(d2_hash == EXPECTED_D2_SHA256, "Dataset2 release output differs")
        require(result_hash == EXPECTED_RESULT_SHA256, "final release ZIP differs")
    report = {
        "kind": "track1_jittor_final_submission_build_v1",
        "strict_release": bool(args.strict_release),
        "official_labels_opened": False,
        "deep_learning_framework": "Jittor",
        "formula": "qnorm(log(base))+0.05*qnorm(exact_other_row_support)+0.02*qnorm(cross_source_community_top10pct)",
        "inputs": {
            "baseline_sha256": sha256_file(args.baseline),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "dataset1_test_sha256": d1_test_hash,
            "dataset2_test_sha256": d2_test_hash,
            "dataset1_rule_sha256": rule_hash,
        },
        "community": {"cosine_threshold": threshold, **community},
        "output": {
            "path": str(args.output),
            "sha256": result_hash,
            "dataset1_sha256": d1_hash,
            "dataset2_sha256": d2_hash,
            "dataset1_changed_rows": int(changed.sum()),
            "dataset2_top1_changed_rows": int(np.sum(np.argmax(reference_logits, axis=1) != np.argmax(candidate_logits, axis=1))),
        },
        "environment": {"jittor": jt.__version__, "use_cuda": bool(jt.flags.use_cuda), "numpy": np.__version__},
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Track 1 final Jittor submission")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--d1-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--strict-release", action="store_true")
    args = parser.parse_args()
    report = build(args)
    print(json.dumps({"result_sha256": report["output"]["sha256"], "strict_release": report["strict_release"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
