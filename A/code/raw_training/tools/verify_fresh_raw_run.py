#!/usr/bin/env python3
"""Bind a fresh raw training receipt to its model directory and inference ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
import main as track_main  # noqa: E402


RECEIPTS = {"training_manifest.json", "raw_training_verification.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in RECEIPTS
    }


def source_inventory() -> dict[str, str]:
    return {
        str(path.relative_to(CODE_ROOT)): sha256_file(path)
        for path in sorted(CODE_ROOT.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json"}
    }


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def require_inventory(value: object, actual: dict[str, str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} inventory is missing")
    expected = {str(name): str(digest) for name, digest in value.items()}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
        raise ValueError(
            f"{label} inventory differs: missing={missing[:8]} extra={extra[:8]} changed={changed[:8]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a full fresh raw train-and-infer run")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    models = args.models.resolve()
    output_dir = args.output_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fresh-run receipt: {output}")
    if not data.is_file() or not models.is_dir() or not output_dir.is_dir():
        raise FileNotFoundError("data, models and output directory must exist")

    data_hash = sha256_file(data)
    artifacts = model_inventory(models)
    training = load_json(models / "training_manifest.json")
    if training.get("kind") != "track1_jittor_training_manifest_v2":
        raise ValueError("unexpected training manifest kind")
    if training.get("data_sha256") != data_hash:
        raise ValueError("training manifest data hash differs")
    require_inventory(training.get("model_file_sha256"), artifacts, "training model")
    current_sources = source_inventory()
    require_inventory(training.get("source_file_sha256"), current_sources, "training source")
    if training.get("requirements_sha256") != sha256_file(CODE_ROOT.parent / "requirements.txt"):
        raise ValueError("training requirements hash differs")

    raw = load_json(models / "raw_training_verification.json")
    if raw.get("kind") != "track1_raw_training_verification_v1":
        raise ValueError("unexpected raw training verification kind")
    if raw.get("decision") != "PASS_FRESH_RAW_NOT_HISTORICAL_PARITY":
        raise ValueError("raw verification decision differs")
    if raw.get("data_sha256") != data_hash:
        raise ValueError("raw verification data hash differs")
    if raw.get("model_file_count") != len(artifacts):
        raise ValueError("raw verification model count differs")
    if raw.get("historical_exact_parity_asserted") is not False:
        raise ValueError("fresh raw receipt must not assert historical parity")

    inference = load_json(output_dir / "inference_manifest.json")
    if inference.get("kind") != "track1_inference_manifest_v1":
        raise ValueError("unexpected inference manifest kind")
    if inference.get("data_sha256") != data_hash:
        raise ValueError("inference manifest data hash differs")
    require_inventory(inference.get("model_file_sha256"), artifacts, "inference model")
    require_inventory(inference.get("source_file_sha256"), current_sources, "inference source")
    if inference.get("requirements_sha256") != sha256_file(CODE_ROOT.parent / "requirements.txt"):
        raise ValueError("inference requirements hash differs")
    if inference.get("strict_release") is not False:
        raise ValueError("fresh raw inference must not assert strict release parity")
    result = output_dir / "result.zip"
    result_report = track_main.validate_submission(result, strict_release=False)
    inferred_result = inference.get("result")
    if not isinstance(inferred_result, dict) or inferred_result.get("sha256") != result_report["sha256"]:
        raise ValueError("inference manifest result hash differs")

    report = {
        "kind": "track1_fresh_raw_run_verification_v1",
        "decision": "PASS_FRESH_RAW_NOT_HISTORICAL_PARITY",
        "data_sha256": data_hash,
        "models": str(models),
        "model_file_count": len(artifacts),
        "output_dir": str(output_dir),
        "result": result_report,
        "historical_exact_parity_asserted": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
