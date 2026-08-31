#!/usr/bin/env python3
"""Fail-closed receipt for a fresh Track 1 raw-model training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


DATASET2_META_CHECKPOINTS = (
    "pool_ranker_seed20260816_jittor.npz",
    "pool_ranker_seed20260817_jittor.npz",
    "pool_ranker_seed20260818_jittor.npz",
    "set64_seed20262725_jittor.npz",
    "set64_seed20262726_jittor.npz",
    "set64_seed20262727_jittor.npz",
    "set96_seed20263725_jittor.npz",
    "set96_seed20263726_jittor.npz",
    "set96_seed20263727_jittor.npz",
    "multi_set64_seed20265701_jittor.npz",
    "multi_set64_seed20265702_jittor.npz",
    "multi_set64_seed20265703_jittor.npz",
    "multi_set96_seed20265801_jittor.npz",
    "multi_set96_seed20265802_jittor.npz",
    "multi_set96_seed20265803_jittor.npz",
    "multi_transformer2_set64_seed20265701_jittor.npz",
    "multi_transformer2_set64_seed20265702_jittor.npz",
    "multi_transformer2_set64_seed20265703_jittor.npz",
    "multi_transformer2_set96_seed20265801_jittor.npz",
    "multi_transformer2_set96_seed20265802_jittor.npz",
    "multi_transformer2_set96_seed20265803_jittor.npz",
    "warm_residual_set64_seed20265701_jittor.npz",
    "warm_residual_set64_seed20265702_jittor.npz",
    "warm_residual_set64_seed20265703_jittor.npz",
    "warm_residual_set96_seed20265801_jittor.npz",
    "warm_residual_set96_seed20265802_jittor.npz",
    "warm_residual_set96_seed20265803_jittor.npz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path, *, excluded: set[str] | None = None) -> dict[str, str]:
    excluded = excluded or set()
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in excluded
    }


def source_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json"}
    }


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def require_hashes(expected: object, actual: dict[str, str], label: str) -> None:
    if not isinstance(expected, dict):
        raise ValueError(f"{label} hash inventory is missing")
    normalized = {str(name): str(digest) for name, digest in expected.items()}
    if normalized != actual:
        missing = sorted(set(normalized) - set(actual))
        extra = sorted(set(actual) - set(normalized))
        changed = sorted(
            name for name in set(normalized) & set(actual) if normalized[name] != actual[name]
        )
        raise ValueError(
            f"{label} inventory differs: missing={missing[:8]} extra={extra[:8]} changed={changed[:8]}"
        )


def check_legacy_zip(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("fresh Legacy ZIP CRC validation failed")
        if archive.namelist() != ["dataset2.csv"]:
            raise ValueError("fresh Legacy ZIP member names differ")
        with archive.open("dataset2.csv") as handle:
            digest = hashlib.sha256()
            rows = 0
            for line in handle:
                digest.update(line)
                if len(line.rstrip(b"\n").split(b",")) != 100:
                    raise ValueError(f"fresh Legacy row {rows} does not have 100 columns")
                rows += 1
    if rows != 153420:
        raise ValueError(f"fresh Legacy ZIP row count differs: {rows}")
    return {"dataset2_csv_sha256": digest.hexdigest(), "rows": rows, "columns": 100}


def require_model_layout(models: Path) -> None:
    base = models / "base_models"
    dataset1 = base / "dataset1"
    dataset2 = base / "dataset2"
    required = [
        dataset1 / "d1_cf_seed_20260705.pkl",
        dataset1 / "d1_cf_seed_20260715.pkl",
        base / "legacy_dataset2_base.zip",
        dataset2 / "model_multvae_prod_jittor.npz",
        dataset2 / "model_recvae_prod_jittor.npz",
        dataset2 / "model_recvae_h800z400_prod_jittor.npz",
        dataset2 / "model_bm25bpr_prod_jittor.npz",
        models / "community" / "model_bpr32_prod_community_jittor.npz",
    ]
    required.extend(dataset2 / name for name in DATASET2_META_CHECKPOINTS)
    missing = [str(path.relative_to(models)) for path in required if not path.is_file()]
    if missing:
        raise ValueError("fresh raw checkpoint layout is incomplete: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a retained fresh raw Jittor training run")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    models = args.models.resolve()
    output = args.output.resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    if not models.is_dir():
        raise FileNotFoundError(models)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {output}")

    require_model_layout(models)

    training_path = models / "training_manifest.json"
    training = load_json(training_path)
    if training.get("kind") != "track1_jittor_training_manifest_v2":
        raise ValueError("unexpected training manifest kind")
    data_hash = sha256_file(data)
    if training.get("data_sha256") != data_hash:
        raise ValueError("official data hash differs from training manifest")
    if not isinstance(training.get("jittor_runtime"), dict):
        raise ValueError("Jittor runtime identity is missing from training manifest")
    model_hashes = inventory(models, excluded={"training_manifest.json"})
    require_hashes(training.get("model_file_sha256"), model_hashes, "model")
    code_root = Path(__file__).resolve().parents[1]
    require_hashes(training.get("source_file_sha256"), source_inventory(code_root), "source")
    if training.get("requirements_sha256") != sha256_file(code_root.parent / "requirements.txt"):
        raise ValueError("requirements hash differs from training manifest")

    components = models / "base_models" / "legacy_components"
    component_path = components / "manifest.json"
    component = load_json(component_path)
    if component.get("kind") != "fresh_legacy_dataset2_jittor_components_v2":
        raise ValueError("unexpected Legacy component manifest kind")
    if component.get("data_sha256") != data_hash:
        raise ValueError("official data hash differs from Legacy component manifest")
    required = (
        "cf_embedding_full_d128.npz",
        "friend/m2.pkl",
        "friend/r2.pkl",
        "friend/result.zip",
        "ours/m2.pkl",
        "ours/r2.pkl",
        "ours/result.zip",
        "cf/m2.pkl",
        "cf/r2.pkl",
        "cf/result.zip",
        "multdae/model_jittor.npz",
        "multdae/official_raw.npy",
    )
    missing = [name for name in required if not (components / name).is_file()]
    if missing:
        raise ValueError("retained Legacy component files are missing: " + ", ".join(missing))
    component_hashes = inventory(components, excluded={"manifest.json"})
    require_hashes(component.get("component_file_sha256"), component_hashes, "Legacy component")

    legacy = models / "base_models" / "legacy_dataset2_base.zip"
    legacy_report = check_legacy_zip(legacy)
    if component.get("legacy_output_sha256") != sha256_file(legacy):
        raise ValueError("fresh Legacy ZIP hash differs from component manifest")
    if component.get("legacy_dataset2_csv_sha256") != legacy_report["dataset2_csv_sha256"]:
        raise ValueError("fresh Legacy CSV hash differs from component manifest")
    if bool(component.get("historical_exact_parity_asserted")):
        raise ValueError("fresh raw manifest must not assert historical exact parity")

    receipt = {
        "kind": "track1_raw_training_verification_v1",
        "decision": "PASS_FRESH_RAW_NOT_HISTORICAL_PARITY",
        "data_sha256": data_hash,
        "models": str(models),
        "model_file_count": len(model_hashes),
        "legacy_components": str(components),
        "legacy_component_file_count": len(component_hashes),
        "legacy": legacy_report,
        "historical_exact_parity_asserted": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
