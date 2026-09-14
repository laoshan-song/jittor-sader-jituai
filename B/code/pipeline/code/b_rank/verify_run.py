#!/usr/bin/env python3
"""Fail-closed validator for Dataset3/Dataset4 B-rank submissions.

The validator intentionally uses only the standard library.  It
streams both official test files and submission members, so Dataset4 never
needs to be materialized in host memory.

Canonical submission contract:
  * a ZIP containing exactly ``dataset3.csv`` then ``dataset4.csv``;
  * each member is headerless ASCII CSV with one row per official test row;
  * each row has exactly 100 ``%.8f`` probabilities which sum to one.

Run manifests are optional for exploratory checks.  When supplied they must
bind the official-data hash and any declared source-file hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
SUBMISSION_MEMBERS = ("dataset3.csv", "dataset4.csv")
TEST_MEMBERS = {"dataset3": "dataset3/test.csv", "dataset4": "dataset4/test.csv"}
INFERENCE_MANIFEST_KIND = "b_rank_inference_manifest_v1"
INFERENCE_SOURCE_FILES = frozenset(
    {
        "b_rank/main.py",
        "b_rank/data_features.py",
        "b_rank/ranker_jittor.py",
        "b_rank/verify_run.py",
    }
)
TEMPORAL_INFERENCE_MANIFEST_KIND = "b_rank_temporal_attention_inference_v1"
TEMPORAL_INFERENCE_SOURCE_FILES = frozenset(
    {
        "b_rank/temporal_infer.py",
        "b_rank/data_features.py",
        "b_rank/temporal_attention_jittor.py",
        "b_rank/temporal_history.py",
        "b_rank/temporal_validate.py",
        "b_rank/verify_run.py",
    }
)
MULTIMODEL_INFERENCE_MANIFEST_KIND = "b_rank_d34_fullhistory_multimodel_submission_v1"
MULTIMODEL_INFERENCE_SOURCE_FILES = frozenset(
    {
        "b_rank/d4_multimodel_infer.py",
        "b_rank/d4_multimodel_fit.py",
        "b_rank/d4_multimodel_shard.py",
        "b_rank/d4_transition_mf_deploy.py",
        "b_rank/d4_transition_research.py",
        "b_rank/data_features.py",
        "b_rank/fullset_ranker_jittor.py",
        "b_rank/implicit_mf_jittor.py",
        "b_rank/pairnew_transformer_jittor.py",
        "b_rank/pool_association.py",
        "b_rank/replay_score_cache.py",
        "b_rank/temporal_attention_jittor.py",
        "b_rank/temporal_history.py",
        "b_rank/temporal_infer.py",
        "b_rank/verify_run.py",
    }
)
WIDTH = 100
DEFAULT_SUM_TOLERANCE = 1e-5
MAX_OUTPUT_ROW_BYTES = 4096
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{8}\Z")


class VerificationError(ValueError):
    """Raised when a data, manifest, or submission contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256_file(path: Path) -> str:
    _require(path.is_file(), f"file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_digest(value: object, label: str) -> str:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} must be a lowercase SHA-256")
    return value


def _test_header() -> list[str]:
    return ["src", "time"] + [f"c{index}" for index in range(1, WIDTH + 1)]


def _count_official_test_rows(data_archive: Path, dataset: str) -> dict[str, int | str]:
    member = TEST_MEMBERS[dataset]
    try:
        with zipfile.ZipFile(data_archive) as archive:
            _require(member in archive.namelist(), f"official archive is missing {member}")
            with archive.open(member, "r") as handle:
                header_bytes = handle.readline(MAX_OUTPUT_ROW_BYTES + 1)
                _require(header_bytes and header_bytes.endswith(b"\n"), f"{member} has no complete header")
                _require(len(header_bytes) <= MAX_OUTPUT_ROW_BYTES, f"{member} header is unexpectedly long")
                try:
                    header = header_bytes[:-1].decode("ascii")
                except UnicodeDecodeError as error:
                    raise VerificationError(f"{member} header is not ASCII") from error
                _require(header.split(",") == _test_header(), f"{member} header differs from src,time,c1..c100")

                count = 0
                line_number = 1
                while True:
                    row = handle.readline(MAX_OUTPUT_ROW_BYTES + 1)
                    if not row:
                        break
                    line_number += 1
                    _require(len(row) <= MAX_OUTPUT_ROW_BYTES, f"{member} line {line_number} is unexpectedly long")
                    _require(row.endswith(b"\n"), f"{member} line {line_number} is not newline-terminated")
                    try:
                        fields = row[:-1].decode("ascii").split(",")
                    except UnicodeDecodeError as error:
                        raise VerificationError(f"{member} line {line_number} is not ASCII") from error
                    _require(len(fields) == WIDTH + 2, f"{member} line {line_number} has {len(fields)} fields, expected {WIDTH + 2}")
                    _require(fields[0] != "" and fields[1] != "", f"{member} line {line_number} has an empty source or time")
                    count += 1
    except zipfile.BadZipFile as error:
        raise VerificationError(f"official archive CRC or ZIP structure failed while reading {member}: {error}") from error
    return {"member": member, "rows": count, "candidate_width": WIDTH}


def _resolve_source_path(source_root: Path, name: object) -> Path:
    _require(isinstance(name, str) and name != "", "manifest source hash has an invalid path")
    _require("\\" not in name, f"manifest source path is not canonical: {name!r}")
    relative = Path(name)
    _require(not relative.is_absolute(), f"manifest source path must be relative: {name}")
    _require("." not in relative.parts and ".." not in relative.parts, f"manifest source path is not canonical: {name}")
    root = source_root.resolve()
    base = root.parent if relative.parts and relative.parts[0] == root.name else root
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise VerificationError(f"manifest source path escapes source root: {name}") from error
    _require(candidate.is_file(), f"manifest source file does not exist: {name}")
    return candidate


def _manifest_source_hashes(manifest: Mapping[str, Any]) -> Mapping[str, object] | None:
    values: list[Mapping[str, object]] = []
    for key in ("source_hashes", "source_file_sha256"):
        if key not in manifest:
            continue
        value = manifest[key]
        _require(isinstance(value, Mapping), f"manifest {key} must be an object")
        values.append(value)
    if not values:
        return None
    first = values[0]
    for value in values[1:]:
        _require(dict(value) == dict(first), "manifest source hash aliases disagree")
    _require(bool(first), "manifest source hash inventory is empty")
    return first


def _manifest_submission_hash(manifest: Mapping[str, Any]) -> str | None:
    candidates: list[object] = []
    for key in ("submission_sha256", "result_sha256", "output_sha256"):
        if key in manifest:
            candidates.append(manifest[key])
    for key in ("submission", "result", "output"):
        value = manifest.get(key)
        if isinstance(value, Mapping) and "sha256" in value:
            candidates.append(value["sha256"])
    if not candidates:
        return None
    hashes = [_require_digest(value, "manifest submission hash") for value in candidates]
    _require(len(set(hashes)) == 1, "manifest submission hash aliases disagree")
    return hashes[0]


def _require_nonnegative_int(value: object, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return value


def _verify_inference_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, object] | None,
    submission_hash: str | None,
    output_rows: Mapping[str, int],
) -> bool:
    """Validate fields that bind a manifest emitted by ``b_rank.main``."""
    if manifest.get("kind") != INFERENCE_MANIFEST_KIND:
        return False
    _require(source_hashes is not None, "inference manifest must bind its source files")
    _require(
        set(source_hashes) == INFERENCE_SOURCE_FILES,
        "inference manifest source inventory differs from the Dataset3/4 pipeline",
    )
    _require(submission_hash is not None, "inference manifest must bind its submission ZIP")

    runtime = manifest.get("jittor_runtime")
    _require(isinstance(runtime, Mapping), "inference manifest has no Jittor runtime")
    _require(
        isinstance(runtime.get("jittor"), str) and bool(runtime["jittor"]),
        "inference manifest Jittor version is invalid",
    )
    _require(
        runtime.get("has_cuda") is True and runtime.get("use_cuda") is True,
        "inference manifest does not record enabled Jittor CUDA",
    )

    gate = manifest.get("validation_gate")
    _require(isinstance(gate, Mapping), "inference manifest has no validation-gate binding")
    _require(
        isinstance(gate.get("path"), str) and bool(gate["path"]),
        "inference manifest validation-gate path is invalid",
    )
    _require_digest(gate.get("sha256"), "inference manifest validation-gate SHA-256")
    _require_digest(
        gate.get("config_sha256"), "inference manifest validation-gate config SHA-256"
    )
    _require(gate.get("decision") == "PASS", "inference manifest validation gate is not PASS")

    checkpoints = manifest.get("checkpoints")
    _require(
        isinstance(checkpoints, Mapping) and set(checkpoints) == set(TEST_MEMBERS),
        "inference manifest checkpoints must cover dataset3 and dataset4",
    )
    for dataset in TEST_MEMBERS:
        checkpoint = checkpoints[dataset]
        _require(isinstance(checkpoint, Mapping), f"inference manifest checkpoint is invalid: {dataset}")
        _require(
            isinstance(checkpoint.get("path"), str) and bool(checkpoint["path"]),
            f"inference manifest checkpoint path is invalid: {dataset}",
        )
        _require_digest(checkpoint.get("sha256"), f"inference manifest checkpoint SHA-256: {dataset}")
        config = checkpoint.get("model_config")
        _require(isinstance(config, Mapping), f"inference manifest model config is invalid: {dataset}")
        kind = config.get("kind")
        feature_dim = _require_nonnegative_int(
            config.get("feature_dim"), f"inference manifest feature_dim: {dataset}"
        )
        hidden_dim = _require_nonnegative_int(
            config.get("hidden_dim"), f"inference manifest hidden_dim: {dataset}"
        )
        _require(feature_dim > 0, f"inference manifest feature_dim must be positive: {dataset}")
        _require(
            (kind == "linear" and hidden_dim == 0)
            or (kind == "deepsets" and hidden_dim > 0),
            f"inference manifest model architecture is invalid: {dataset}",
        )
        _require_nonnegative_int(
            checkpoint.get("test_feature_cutoff"),
            f"inference manifest test feature cutoff: {dataset}",
        )

    row_counts = manifest.get("row_counts")
    _require(
        isinstance(row_counts, Mapping) and set(row_counts) == set(TEST_MEMBERS),
        "inference manifest row counts must cover dataset3 and dataset4",
    )
    for dataset in TEST_MEMBERS:
        rows = _require_nonnegative_int(
            row_counts[dataset], f"inference manifest row count: {dataset}"
        )
        _require(
            rows == output_rows[dataset],
            f"inference manifest row count differs from {dataset}.csv",
        )
    return True


def _verify_temporal_attention_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, object] | None,
    submission_hash: str | None,
    output_rows: Mapping[str, int],
) -> bool:
    """Validate the fixed Jittor temporal-attention deployment manifest."""
    if manifest.get("kind") != TEMPORAL_INFERENCE_MANIFEST_KIND:
        return False
    _require(source_hashes is not None, "temporal manifest must bind its source files")
    _require(
        set(source_hashes) == TEMPORAL_INFERENCE_SOURCE_FILES,
        "temporal manifest source inventory differs from the deployment pipeline",
    )
    _require(submission_hash is not None, "temporal manifest must bind its submission ZIP")

    runtime = manifest.get("jittor_runtime")
    _require(isinstance(runtime, Mapping), "temporal manifest has no Jittor runtime")
    _require(
        isinstance(runtime.get("jittor"), str) and bool(runtime["jittor"]),
        "temporal manifest Jittor version is invalid",
    )
    _require(
        runtime.get("has_cuda") is True and runtime.get("use_cuda") is True,
        "temporal manifest does not record enabled Jittor CUDA",
    )

    selection = manifest.get("selection")
    _require(isinstance(selection, Mapping), "temporal manifest has no validation-selection binding")
    _require(
        isinstance(selection.get("rule"), str) and bool(selection["rule"]),
        "temporal manifest selection rule is invalid",
    )
    evidence = selection.get("evidence")
    _require(
        isinstance(evidence, Mapping) and set(evidence) == set(TEST_MEMBERS),
        "temporal manifest selection evidence must cover dataset3 and dataset4",
    )

    protocol = manifest.get("training_protocol")
    _require(isinstance(protocol, Mapping), "temporal manifest has no training protocol")
    _require(protocol.get("static_features") is True, "temporal manifest must use frozen static features")
    for name in ("group_seed", "history_size", "embedding_dim", "epochs", "train_batch_rows"):
        _require_nonnegative_int(protocol.get(name), f"temporal manifest protocol {name}")
    group_sizes = protocol.get("group_sizes")
    _require(
        isinstance(group_sizes, Mapping) and set(group_sizes) == {"train", "valid", "confirm"},
        "temporal manifest group sizes are invalid",
    )
    for name in group_sizes:
        _require_nonnegative_int(group_sizes[name], f"temporal manifest group size {name}")

    checkpoints = manifest.get("checkpoints")
    _require(
        isinstance(checkpoints, Mapping) and set(checkpoints) == set(TEST_MEMBERS),
        "temporal manifest checkpoints must cover dataset3 and dataset4",
    )
    for dataset in TEST_MEMBERS:
        checkpoint = checkpoints[dataset]
        _require(isinstance(checkpoint, Mapping), f"temporal checkpoint is invalid: {dataset}")
        _require(
            isinstance(checkpoint.get("path"), str) and bool(checkpoint["path"]),
            f"temporal checkpoint path is invalid: {dataset}",
        )
        _require_digest(checkpoint.get("sha256"), f"temporal checkpoint SHA-256: {dataset}")
        _require(
            isinstance(checkpoint.get("seed"), int) and not isinstance(checkpoint["seed"], bool),
            f"temporal checkpoint seed is invalid: {dataset}",
        )
        config = checkpoint.get("model_config")
        _require(isinstance(config, Mapping), f"temporal model config is invalid: {dataset}")
        _require(config.get("kind") == "temporal_attention", f"temporal model kind is invalid: {dataset}")
        for name in ("source_count", "item_count", "embedding_dim", "feature_dim"):
            value = _require_nonnegative_int(config.get(name), f"temporal model {name}: {dataset}")
            _require(value > 0, f"temporal model {name} must be positive: {dataset}")
        _require(config.get("static_context") is True, f"temporal static context is disabled: {dataset}")
        _require(
            isinstance(config.get("static_context_pair_seen_only"), bool),
            f"temporal pair-seen gate is invalid: {dataset}",
        )
        for name in (
            "test_feature_cutoff",
            "training_feature_cutoff",
            "training_history_cutoff",
            "history_rows",
            "training_history_rows",
        ):
            _require_nonnegative_int(checkpoint.get(name), f"temporal checkpoint {name}: {dataset}")
        selected = evidence[dataset]
        _require(isinstance(selected, Mapping), f"temporal selection evidence is invalid: {dataset}")
        _require(
            isinstance(selected.get("seed"), int) and not isinstance(selected["seed"], bool),
            f"temporal selected seed is invalid: {dataset}",
        )
        _require(
            checkpoint["seed"] == selected["seed"],
            f"temporal checkpoint seed differs from selection evidence: {dataset}",
        )
        selected_pair_seen_only = selected.get("static_context_pair_seen_only")
        _require(
            isinstance(selected_pair_seen_only, bool),
            f"temporal pair-seen evidence is invalid: {dataset}",
        )
        _require(
            config["static_context_pair_seen_only"] is selected_pair_seen_only,
            f"temporal pair-seen gate differs from selection evidence: {dataset}",
        )
        _require(
            selected.get("static_context") is True,
            f"temporal selection evidence disables static context: {dataset}",
        )

    row_counts = manifest.get("row_counts")
    _require(
        isinstance(row_counts, Mapping) and set(row_counts) == set(TEST_MEMBERS),
        "temporal manifest row counts must cover dataset3 and dataset4",
    )
    for dataset in TEST_MEMBERS:
        rows = _require_nonnegative_int(
            row_counts[dataset], f"temporal manifest row count: {dataset}"
        )
        _require(
            rows == output_rows[dataset],
            f"temporal manifest row count differs from {dataset}.csv",
        )
    return True


def _verify_multimodel_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, object] | None,
    submission_hash: str | None,
    output_rows: Mapping[str, int],
) -> bool:
    """Validate the D3/D4 multi-model deployment manifest."""
    if manifest.get("kind") != MULTIMODEL_INFERENCE_MANIFEST_KIND:
        return False
    _require(source_hashes is not None, "multi-model manifest must bind its source files")
    _require(
        set(source_hashes) == MULTIMODEL_INFERENCE_SOURCE_FILES,
        "multi-model manifest source inventory differs from the deployment pipeline",
    )
    _require(submission_hash is not None, "multi-model manifest must bind its submission ZIP")

    runtime = manifest.get("jittor_runtime")
    _require(isinstance(runtime, Mapping), "multi-model manifest has no Jittor runtime")
    _require(
        isinstance(runtime.get("version"), str) and bool(runtime["version"]),
        "multi-model manifest Jittor version is invalid",
    )
    _require(
        runtime.get("has_cuda") is True and runtime.get("use_cuda") is True,
        "multi-model manifest does not record enabled Jittor CUDA",
    )

    row_counts = manifest.get("row_counts")
    _require(
        isinstance(row_counts, Mapping) and set(row_counts) == set(TEST_MEMBERS),
        "multi-model manifest row counts must cover dataset3 and dataset4",
    )
    for dataset in TEST_MEMBERS:
        rows = _require_nonnegative_int(
            row_counts[dataset], f"multi-model manifest row count: {dataset}"
        )
        _require(
            rows == output_rows[dataset],
            f"multi-model manifest row count differs from {dataset}.csv",
        )

    dataset3 = manifest.get("dataset3")
    _require(isinstance(dataset3, Mapping), "multi-model manifest has no Dataset3 binding")
    for name in ("source_zip", "source_manifest"):
        _require(
            isinstance(dataset3.get(name), str) and bool(dataset3[name]),
            f"multi-model Dataset3 {name} is invalid",
        )
    for name in ("source_zip_sha256", "source_manifest_sha256", "csv_sha256"):
        _require_digest(dataset3.get(name), f"multi-model Dataset3 {name}")
    active_d3 = _require_nonnegative_int(
        dataset3.get("active_component_count"),
        "multi-model Dataset3 active component count",
    )
    weights_d3 = dataset3.get("weights")
    _require(
        isinstance(weights_d3, Mapping) and len(weights_d3) == active_d3 >= 2,
        "multi-model Dataset3 weights do not match its active component count",
    )
    d3_values = list(weights_d3.values())
    _require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in d3_values
        )
        and abs(math.fsum(float(value) for value in d3_values) - 1.0) <= 1e-8,
        "multi-model Dataset3 weights are invalid",
    )

    dataset4 = manifest.get("dataset4")
    _require(isinstance(dataset4, Mapping), "multi-model manifest has no Dataset4 binding")
    fit_report = dataset4.get("fit_report")
    _require(isinstance(fit_report, Mapping), "multi-model Dataset4 fit report is invalid")
    _require(
        isinstance(fit_report.get("path"), str) and bool(fit_report["path"]),
        "multi-model Dataset4 fit report path is invalid",
    )
    _require_digest(fit_report.get("sha256"), "multi-model Dataset4 fit report SHA-256")

    temporal_reports = dataset4.get("temporal_reports")
    _require(
        isinstance(temporal_reports, list) and bool(temporal_reports),
        "multi-model Dataset4 temporal reports are missing",
    )
    allowed_temporal_kinds = {
        "d4_full_split1_temporal_deploy_v1",
        "d4_full_split1_testpool_temporal_deploy_v1",
    }
    for report in temporal_reports:
        _require(isinstance(report, Mapping), "multi-model temporal report is invalid")
        _require(
            isinstance(report.get("path"), str) and bool(report["path"]),
            "multi-model temporal report path is invalid",
        )
        _require_digest(report.get("sha256"), "multi-model temporal report SHA-256")
        _require(
            report.get("kind") in allowed_temporal_kinds,
            "multi-model temporal report kind is invalid",
        )

    mf_reports = dataset4.get("mf_reports")
    _require(
        isinstance(mf_reports, Mapping) and bool(mf_reports),
        "multi-model Dataset4 MF reports are missing",
    )
    for name, report in mf_reports.items():
        _require(isinstance(name, str) and bool(name), "multi-model MF name is invalid")
        _require(isinstance(report, Mapping), f"multi-model MF report is invalid: {name}")
        _require(
            isinstance(report.get("path"), str) and bool(report["path"]),
            f"multi-model MF report path is invalid: {name}",
        )
        _require_digest(report.get("sha256"), f"multi-model MF report SHA-256: {name}")

    names = dataset4.get("component_names")
    active = dataset4.get("active_components")
    weights = dataset4.get("weights")
    _require(
        isinstance(names, list)
        and len(names) == len(set(names))
        and all(isinstance(name, str) and bool(name) for name in names),
        "multi-model Dataset4 component names are invalid",
    )
    _require(
        isinstance(weights, Mapping) and set(weights) == set(names),
        "multi-model Dataset4 weights do not match its components",
    )
    values = {name: weights[name] for name in names}
    _require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in values.values()
        )
        and abs(math.fsum(float(value) for value in values.values()) - 1.0) <= 1e-8,
        "multi-model Dataset4 weights are invalid",
    )
    expected_active = [name for name in names if float(values[name]) > 1e-12]
    _require(
        isinstance(active, list)
        and active == expected_active
        and len(active) >= 2,
        "multi-model Dataset4 active components are invalid",
    )
    _require(
        dataset4.get("base_component") in names,
        "multi-model Dataset4 base component is invalid",
    )
    for name in ("seen_alpha", "new_alpha"):
        value = dataset4.get(name)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0,
            f"multi-model Dataset4 {name} is invalid",
        )
    _require_nonnegative_int(dataset4.get("test_cutoff"), "multi-model Dataset4 test cutoff")
    _require_nonnegative_int(dataset4.get("history_rows"), "multi-model Dataset4 history rows")
    _require(dataset4.get("uses_test_labels") is False, "multi-model Dataset4 uses test labels")
    return True


def _verify_manifest(
    manifest_path: Path | None,
    *,
    data_sha256: str,
    submission_sha256: str,
    source_root: Path,
    output_rows: Mapping[str, int],
) -> dict[str, Any]:
    if manifest_path is None:
        return {"provided": False, "source_hashes_verified": False}

    _require(manifest_path.is_file(), f"run manifest does not exist: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read run manifest {manifest_path}: {error}") from error
    _require(isinstance(value, Mapping), "run manifest must be a JSON object")
    manifest_data_hash = _require_digest(value.get("data_sha256"), "manifest data_sha256")
    _require(manifest_data_hash == data_sha256, "run manifest data_sha256 differs from the supplied archive")

    source_hashes = _manifest_source_hashes(value)
    verified_sources: dict[str, str] = {}
    if source_hashes is not None:
        _require(source_root.is_dir(), f"source root does not exist: {source_root}")
        for name in sorted(source_hashes):
            expected = _require_digest(source_hashes[name], f"manifest source hash for {name}")
            actual = _sha256_file(_resolve_source_path(source_root, name))
            _require(actual == expected, f"manifest source hash differs: {name}")
            verified_sources[str(name)] = actual

    expected_submission_hash = _manifest_submission_hash(value)
    if expected_submission_hash is not None:
        _require(expected_submission_hash == submission_sha256, "run manifest submission hash differs from the ZIP")
    if value.get("kind") == INFERENCE_MANIFEST_KIND:
        inference_contract = _verify_inference_manifest_contract(
            value,
            source_hashes=source_hashes,
            submission_hash=expected_submission_hash,
            output_rows=output_rows,
        )
    elif value.get("kind") == TEMPORAL_INFERENCE_MANIFEST_KIND:
        inference_contract = _verify_temporal_attention_manifest_contract(
            value,
            source_hashes=source_hashes,
            submission_hash=expected_submission_hash,
            output_rows=output_rows,
        )
    elif value.get("kind") == MULTIMODEL_INFERENCE_MANIFEST_KIND:
        inference_contract = _verify_multimodel_manifest_contract(
            value,
            source_hashes=source_hashes,
            submission_hash=expected_submission_hash,
            output_rows=output_rows,
        )
    else:
        inference_contract = False
    return {
        "provided": True,
        "path": str(manifest_path),
        "kind": value.get("kind"),
        "source_hashes_verified": source_hashes is not None,
        "source_file_count": len(verified_sources),
        "submission_hash_bound": expected_submission_hash is not None,
        "inference_manifest_contract_verified": inference_contract,
    }


def _parse_probability(token: str, dataset: str, line_number: int, column: int) -> float:
    _require(DECIMAL_RE.fullmatch(token) is not None, f"{dataset}.csv line {line_number} column {column} is not canonical %.8f")
    value = float(token)
    _require(math.isfinite(value) and 0.0 <= value <= 1.0, f"{dataset}.csv line {line_number} column {column} is outside [0, 1]")
    return value


def _validate_submission_member(archive: zipfile.ZipFile, dataset: str, expected_rows: int, tolerance: float) -> dict[str, Any]:
    member = f"{dataset}.csv"
    info = archive.getinfo(member)
    _require(not info.is_dir(), f"submission member is a directory: {member}")
    _require(not (info.flag_bits & 0x1), f"encrypted submission member is not allowed: {member}")
    digest = hashlib.sha256()
    rows = 0
    try:
        with archive.open(info, "r") as handle:
            while True:
                line = handle.readline(MAX_OUTPUT_ROW_BYTES + 1)
                if not line:
                    break
                row_number = rows + 1
                _require(len(line) <= MAX_OUTPUT_ROW_BYTES, f"{member} line {row_number} is unexpectedly long")
                _require(line.endswith(b"\n") and not line.endswith(b"\r\n"), f"{member} line {row_number} must end with LF")
                _require(b"\r" not in line, f"{member} line {row_number} contains CR")
                digest.update(line)
                try:
                    fields = line[:-1].decode("ascii").split(",")
                except UnicodeDecodeError as error:
                    raise VerificationError(f"{member} line {row_number} is not ASCII") from error
                _require(len(fields) == WIDTH, f"{member} line {row_number} has {len(fields)} fields, expected {WIDTH}")
                values = [_parse_probability(token, dataset, row_number, column) for column, token in enumerate(fields, start=1)]
                probability_sum = math.fsum(values)
                _require(abs(probability_sum - 1.0) <= tolerance, f"{member} line {row_number} sums to {probability_sum:.12g}, tolerance is {tolerance:.12g}")
                rows += 1
                _require(rows <= expected_rows, f"{member} has more rows than official {dataset}/test.csv")
    except zipfile.BadZipFile as error:
        raise VerificationError(f"submission ZIP CRC failed while reading {member}: {error}") from error

    _require(rows == expected_rows, f"{member} has {rows} rows, official {dataset}/test.csv has {expected_rows}")
    return {
        "member": member,
        "rows": rows,
        "width": WIDTH,
        "csv_sha256": digest.hexdigest(),
        "crc_verified_by_full_stream": True,
    }


def verify_run(
    data_archive: str | Path,
    submission: str | Path,
    manifest: str | Path | None = None,
    *,
    source_root: str | Path = CODE_ROOT,
    expected_data_sha256: str = EXPECTED_DATA_SHA256,
    sum_tolerance: float = DEFAULT_SUM_TOLERANCE,
) -> dict[str, Any]:
    """Verify a B-rank ZIP and return a machine-readable PASS report.

    Raises ``VerificationError`` on every contract violation.  The function
    only streams CSV members; it never loads either Dataset4 test/output table
    as an array.  A supplied manifest must contain ``data_sha256`` and may
    contain ``source_hashes`` (or ``source_file_sha256``) as a path-to-hash
    object relative to ``source_root``.
    """

    data_path = Path(data_archive).resolve()
    submission_path = Path(submission).resolve()
    manifest_path = Path(manifest).resolve() if manifest is not None else None
    root = Path(source_root).resolve()
    _require(math.isfinite(sum_tolerance) and 0.0 < sum_tolerance < 0.01, "sum_tolerance must be in (0, 0.01)")
    expected_hash = _require_digest(expected_data_sha256, "expected data SHA-256")

    data_sha256 = _sha256_file(data_path)
    _require(data_sha256 == expected_hash, "official data archive SHA-256 differs from the expected value")
    official = {dataset: _count_official_test_rows(data_path, dataset) for dataset in TEST_MEMBERS}

    _require(submission_path.is_file(), f"submission ZIP does not exist: {submission_path}")
    submission_sha256 = _sha256_file(submission_path)
    try:
        with zipfile.ZipFile(submission_path) as archive:
            names = archive.namelist()
            _require(names == list(SUBMISSION_MEMBERS), f"submission ZIP members must be exactly {list(SUBMISSION_MEMBERS)}")
            outputs = {
                dataset: _validate_submission_member(archive, dataset, int(official[dataset]["rows"]), sum_tolerance)
                for dataset in TEST_MEMBERS
            }
    except zipfile.BadZipFile as error:
        raise VerificationError(f"submission ZIP is invalid: {error}") from error

    manifest_report = _verify_manifest(
        manifest_path,
        data_sha256=data_sha256,
        submission_sha256=submission_sha256,
        source_root=root,
        output_rows={dataset: int(outputs[dataset]["rows"]) for dataset in TEST_MEMBERS},
    )
    return {
        "kind": "b_rank_submission_verification_v1",
        "decision": "PASS",
        "data": {
            "path": str(data_path),
            "sha256": data_sha256,
            "expected_sha256": expected_hash,
            "tests": official,
        },
        "submission": {
            "path": str(submission_path),
            "sha256": submission_sha256,
            "members": list(SUBMISSION_MEMBERS),
            "outputs": outputs,
        },
        "manifest": manifest_report,
        "sum_tolerance": sum_tolerance,
    }


def _self_test() -> dict[str, Any]:
    """Exercise the streaming validator on a one-row synthetic archive."""

    header = ",".join(_test_header()) + "\n"
    row = "1,2," + ",".join("3" for _ in range(WIDTH)) + "\n"
    probabilities = (",".join("0.01000000" for _ in range(WIDTH)) + "\n").encode("ascii")
    with tempfile.TemporaryDirectory(prefix="b_rank_verify_") as temporary:
        directory = Path(temporary)
        data = directory / "data.zip"
        result = directory / "result.zip"
        source_root = directory / "code"
        manifest = directory / "run_manifest.json"
        with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("dataset3/test.csv", header + row)
            archive.writestr("dataset4/test.csv", header + row)
        with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("dataset3.csv", probabilities)
            archive.writestr("dataset4.csv", probabilities)
        source_hashes: dict[str, str] = {}
        for name in sorted(INFERENCE_SOURCE_FILES):
            source_file = source_root / name
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_text(f"# synthetic {name}\n", encoding="ascii")
            source_hashes[name] = _sha256_file(source_file)
        checkpoint = {
            "path": "/synthetic/checkpoint.npz",
            "sha256": "0" * 64,
            "model_config": {"kind": "linear", "feature_dim": 11, "hidden_dim": 0},
            "test_feature_cutoff": 3,
        }
        manifest.write_text(
            json.dumps(
                {
                    "kind": INFERENCE_MANIFEST_KIND,
                    "data_sha256": _sha256_file(data),
                    "source_hashes": source_hashes,
                    "submission_sha256": _sha256_file(result),
                    "jittor_runtime": {
                        "jittor": "1.3.11.0",
                        "has_cuda": True,
                        "use_cuda": True,
                    },
                    "validation_gate": {
                        "path": "/synthetic/validation_gate.json",
                        "sha256": "1" * 64,
                        "config_sha256": "2" * 64,
                        "decision": "PASS",
                    },
                    "checkpoints": {"dataset3": checkpoint, "dataset4": checkpoint},
                    "row_counts": {"dataset3": 1, "dataset4": 1},
                }
            ),
            encoding="utf-8",
        )
        report = verify_run(
            data,
            result,
            manifest,
            source_root=source_root,
            expected_data_sha256=_sha256_file(data),
        )
        temporal_source_hashes: dict[str, str] = {}
        for name in sorted(TEMPORAL_INFERENCE_SOURCE_FILES):
            source_file = source_root / name
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_text(f"# synthetic {name}\n", encoding="ascii")
            temporal_source_hashes[name] = _sha256_file(source_file)
        temporal_checkpoints = {}
        temporal_evidence = {}
        for dataset in TEST_MEMBERS:
            pair_seen_only = dataset == "dataset4"
            temporal_checkpoints[dataset] = {
                "path": f"/synthetic/{dataset}.npz",
                "sha256": "3" * 64,
                "seed": 7,
                "model_config": {
                    "kind": "temporal_attention",
                    "source_count": 2,
                    "item_count": 3,
                    "embedding_dim": 32,
                    "feature_dim": 11,
                    "static_context": True,
                    "static_context_pair_seen_only": pair_seen_only,
                    "dropout": 0.0,
                    "time_scale": 0.25,
                },
                "test_feature_cutoff": 3,
                "training_feature_cutoff": 2,
                "training_history_cutoff": 3,
                "history_rows": 4,
                "training_history_rows": 3,
            }
            temporal_evidence[dataset] = {
                "seed": 7,
                "static_context": True,
                "static_context_pair_seen_only": pair_seen_only,
            }
        manifest.write_text(
            json.dumps(
                {
                    "kind": TEMPORAL_INFERENCE_MANIFEST_KIND,
                    "data_sha256": _sha256_file(data),
                    "source_hashes": temporal_source_hashes,
                    "submission_sha256": _sha256_file(result),
                    "jittor_runtime": {
                        "jittor": "1.3.11.0",
                        "has_cuda": True,
                        "use_cuda": True,
                    },
                    "selection": {"rule": "synthetic", "evidence": temporal_evidence},
                    "training_protocol": {
                        "group_seed": 7,
                        "group_sizes": {"train": 1, "valid": 1, "confirm": 1},
                        "history_size": 32,
                        "embedding_dim": 32,
                        "epochs": 5,
                        "train_batch_rows": 256,
                        "static_features": True,
                    },
                    "checkpoints": temporal_checkpoints,
                    "row_counts": {"dataset3": 1, "dataset4": 1},
                }
            ),
            encoding="utf-8",
        )
        temporal_report = verify_run(
            data,
            result,
            manifest,
            source_root=source_root,
            expected_data_sha256=_sha256_file(data),
        )
        _require(
            temporal_report["manifest"]["inference_manifest_contract_verified"],
            "temporal manifest self-test did not validate its contract",
        )
        report["temporal_manifest_contract"] = "PASS"

        multimodel_source_hashes: dict[str, str] = {}
        for name in sorted(MULTIMODEL_INFERENCE_SOURCE_FILES):
            source_file = source_root / name
            source_file.parent.mkdir(parents=True, exist_ok=True)
            if not source_file.exists():
                source_file.write_text(f"# synthetic {name}\n", encoding="ascii")
            multimodel_source_hashes[name] = _sha256_file(source_file)
        manifest.write_text(
            json.dumps(
                {
                    "kind": MULTIMODEL_INFERENCE_MANIFEST_KIND,
                    "data_sha256": _sha256_file(data),
                    "source_hashes": multimodel_source_hashes,
                    "submission_sha256": _sha256_file(result),
                    "row_counts": {"dataset3": 1, "dataset4": 1},
                    "jittor_runtime": {
                        "version": "1.3.11.0",
                        "has_cuda": True,
                        "use_cuda": True,
                    },
                    "dataset3": {
                        "source_zip": "/synthetic/source.zip",
                        "source_zip_sha256": "4" * 64,
                        "source_manifest": "/synthetic/source.manifest.json",
                        "source_manifest_sha256": "5" * 64,
                        "csv_sha256": "6" * 64,
                        "active_component_count": 2,
                        "weights": {"d3_a": 0.5, "d3_b": 0.5},
                    },
                    "dataset4": {
                        "fit_report": {
                            "path": "/synthetic/fit.json",
                            "sha256": "7" * 64,
                        },
                        "temporal_reports": [
                            {
                                "path": "/synthetic/temporal.json",
                                "sha256": "8" * 64,
                                "kind": "d4_full_split1_temporal_deploy_v1",
                            }
                        ],
                        "mf_reports": {
                            "fullhistory_mf_seed10": {
                                "path": "/synthetic/mf.json",
                                "sha256": "9" * 64,
                            }
                        },
                        "component_names": [
                            "temporal_h32_seed10",
                            "fullhistory_mf_seed10",
                        ],
                        "active_components": [
                            "temporal_h32_seed10",
                            "fullhistory_mf_seed10",
                        ],
                        "weights": {
                            "temporal_h32_seed10": 0.5,
                            "fullhistory_mf_seed10": 0.5,
                        },
                        "base_component": "fullhistory_mf_seed10",
                        "seen_alpha": 1.0,
                        "new_alpha": 1.0,
                        "test_cutoff": 3,
                        "history_rows": 4,
                        "uses_test_labels": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        multimodel_report = verify_run(
            data,
            result,
            manifest,
            source_root=source_root,
            expected_data_sha256=_sha256_file(data),
        )
        _require(
            multimodel_report["manifest"]["inference_manifest_contract_verified"],
            "multi-model manifest self-test did not validate its contract",
        )
        report["multimodel_manifest_contract"] = "PASS"
    report["self_test"] = "PASS"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed Dataset3/4 B-rank submission verifier")
    parser.add_argument("--data", type=Path, help="official data_B.zip")
    parser.add_argument("--submission", type=Path, help="candidate submission ZIP")
    parser.add_argument("--manifest", type=Path, help="optional run manifest JSON")
    parser.add_argument("--require-manifest", action="store_true", help="reject when --manifest is absent")
    parser.add_argument("--source-root", type=Path, default=CODE_ROOT, help="root used for manifest source hashes (default: code/)")
    parser.add_argument("--expected-data-sha256", default=EXPECTED_DATA_SHA256, help="official data archive SHA-256")
    parser.add_argument("--sum-tolerance", type=float, default=DEFAULT_SUM_TOLERANCE, help="maximum per-row probability-sum error")
    parser.add_argument("--self-test", action="store_true", help="run a small synthetic streaming self-test")
    args = parser.parse_args()

    try:
        if args.self_test:
            _require(args.data is None and args.submission is None and args.manifest is None, "--self-test cannot be combined with data arguments")
            report = _self_test()
        else:
            _require(args.data is not None and args.submission is not None, "--data and --submission are required")
            _require(not args.require_manifest or args.manifest is not None, "--require-manifest needs --manifest")
            report = verify_run(
                args.data,
                args.submission,
                args.manifest,
                source_root=args.source_root,
                expected_data_sha256=args.expected_data_sha256,
                sum_tolerance=args.sum_tolerance,
            )
            if args.require_manifest:
                _require(
                    report["manifest"]["inference_manifest_contract_verified"],
                    "--require-manifest needs a complete Dataset3/4 inference manifest",
                )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0
    except (OSError, VerificationError, zipfile.BadZipFile) as error:
        report = {
            "kind": "b_rank_submission_verification_v1",
            "decision": "REJECT",
            "error": str(error),
        }
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stdout, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
