#!/usr/bin/env python3
"""Static and archive-boundary audit for the Track 1 delivery package."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path


BANNED_FRAMEWORKS = {"torch", "tensorflow", "paddle", "mindspore", "mxnet", "jax", "keras"}
EXTERNAL_SCORE_REFERENCES = ("1.4351/" + "result.zip", "craft_" + "baseline", "Jittor" + "Geometric")
REQUIRED = (
    "code",
    "requirements.txt",
    "README.md",
    "提交说明文档.pdf",
    "submission_metadata.json",
    "run_train.sh",
    "run_inference.sh",
    "run_verify.sh",
    "run_raw.sh",
    "run_fresh_inference.sh",
    "MANIFEST.sha256",
)
REQUIRED_SOURCES = (
    "code/main.py",
    "code/build_result.py",
    "code/build_set_v2_result.py",
    "code/build_final_submission.py",
    "code/train_community_bpr_jittor.py",
    "code/dataset1/train_d1_cf_jittor.py",
    "code/dataset1/score_d1_ensemble_jittor.py",
    "code/dataset2/train_vae_jittor.py",
    "code/dataset2/train_bpr_jittor.py",
    "code/legacy_dataset2/rebuild_from_raw.py",
    "code/tools/check_environment.py",
    "code/tools/check_raw_pipeline_contract.py",
    "code/tools/prepare_cuda_runtime.sh",
    "code/tools/record_release_receipt.py",
    "code/tools/verify_raw_training.py",
    "code/tools/verify_fresh_raw_run.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ignored(relative: Path) -> bool:
    """Match the delivery writer's exclusion of interpreter-generated cache."""
    return any(part == "__pycache__" for part in relative.parts) or relative.suffix in {".pyc", ".pyo"}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".", 1)[0])
    return modules


def inspect_legacy_artifact(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("legacy score artifact CRC validation failed")
        if archive.namelist() != ["dataset2.csv"]:
            raise RuntimeError("legacy score artifact members differ")
        with archive.open("dataset2.csv") as handle:
            reader = csv.reader(line.decode("ascii") for line in handle)
            rows = 0
            for row in reader:
                if len(row) != 100:
                    raise RuntimeError(f"legacy score row {rows} has {len(row)} columns")
                rows += 1
    if rows != 153420:
        raise RuntimeError(f"legacy score artifact has {rows} rows")
    return {"path": str(path), "sha256": sha256_file(path), "rows": rows, "columns": 100}


def verify_manifest(root: Path) -> dict[str, int]:
    """Validate the package inventory produced by build_delivery_package.py."""
    manifest = root / "MANIFEST.sha256"
    # Paths in a release inventory may include the required Chinese PDF name.
    rows = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected: dict[str, str] = {}
    for row in rows:
        digest, separator, name = row.partition("  ")
        if not separator or len(digest) != 64 or not name.startswith("./"):
            raise RuntimeError(f"invalid manifest row: {row!r}")
        if name in expected:
            raise RuntimeError(f"duplicate manifest path: {name}")
        expected[name] = digest
    actual = {
        "./" + str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest and not ignored(path.relative_to(root))
    }
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
        raise RuntimeError(
            "manifest differs: "
            f"missing={missing[:8]} extra={extra[:8]} changed={changed[:8]}"
        )
    return {"files": len(actual)}


def check_raw_pipeline_contract(root: Path) -> dict[str, object]:
    """Run the no-Jittor raw-source preflight from the delivery tree itself."""
    path = root / "code" / "tools" / "check_raw_pipeline_contract.py"
    spec = importlib.util.spec_from_file_location("track1_raw_pipeline_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load raw pipeline contract checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.check(root / "code")
    if report.get("status") != "PASS":
        raise RuntimeError("raw pipeline contract failed: " + json.dumps(report, sort_keys=True))
    return report


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    missing = [name for name in REQUIRED if not (root / name).exists()]
    missing_sources = [name for name in REQUIRED_SOURCES if not (root / name).is_file()]
    if missing or missing_sources:
        raise RuntimeError(json.dumps({"missing": missing, "missing_sources": missing_sources}, ensure_ascii=True))
    pdf = root / "提交说明文档.pdf"
    if not pdf.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("submission PDF has an invalid header")
    raw_runner = (root / "run_raw.sh").read_text(encoding="utf-8")
    raw_commands = "\n".join(
        line for line in raw_runner.splitlines() if not line.lstrip().startswith("#")
    )
    if "--strict-release" in raw_commands:
        raise RuntimeError("raw runner must not claim a historical release hash")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    required_dependencies = {
        "jittor==1.3.10.0",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "numba==0.66.0",
        "nvidia-cudnn-cu12==8.9.7.29",
    }
    missing_dependencies = sorted(required_dependencies - set(requirements))
    if missing_dependencies:
        raise RuntimeError("pinned runtime dependencies are missing: " + ", ".join(missing_dependencies))
    metadata = json.loads((root / "submission_metadata.json").read_text(encoding="utf-8"))
    expected_metadata = {
        "team": "sader",
        "rank": "7",
        "score": "1.521072794155721",
        "contact": "\u5b8b\u6587\u97ec",
        "wechat": "laoshan_song",
        "phone": "18984157192",
    }
    if metadata != expected_metadata:
        raise RuntimeError("submission metadata differs from the declared Track 1 A-list record")
    forbidden_imports: dict[str, list[str]] = {}
    external_score_references: dict[str, list[str]] = {}
    jittor_sources = 0
    for source in sorted((root / "code").rglob("*.py")):
        content = source.read_text(encoding="utf-8")
        modules = imported_modules(source)
        banned = sorted(modules & BANNED_FRAMEWORKS)
        if banned:
            forbidden_imports[str(source.relative_to(root))] = banned
        if "jittor" in modules:
            jittor_sources += 1
        markers = [marker for marker in EXTERNAL_SCORE_REFERENCES if marker in content]
        if markers:
            external_score_references[str(source.relative_to(root))] = markers
    if forbidden_imports:
        raise RuntimeError("non-Jittor deep-learning imports: " + json.dumps(forbidden_imports, ensure_ascii=True))
    if external_score_references:
        raise RuntimeError("external score or non-Jittor model references: " + json.dumps(external_score_references, ensure_ascii=True))
    data_files = [str(path.relative_to(root)) for path in root.rglob("data_A.zip")]
    if data_files:
        raise RuntimeError("official data was packaged: " + json.dumps(data_files, ensure_ascii=True))
    symlinks = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError("symlinks are not permitted in the delivery tree: " + json.dumps(symlinks, ensure_ascii=True))
    raw_contract = check_raw_pipeline_contract(root)
    legacy = root / "code" / "artifacts" / "release_models" / "base_models" / "legacy_dataset2_base.zip"
    if not legacy.is_file():
        raise RuntimeError("release legacy score-plane artifact is missing")
    return {
        "kind": "track1_package_static_audit_v1",
        "status": "PASS_WITH_LINEAGE_DISCLOSURE",
        "root": str(root),
        "jittor_source_files": jittor_sources,
        "forbidden_framework_imports": forbidden_imports,
        "external_score_references": external_score_references,
        "official_data_packaged": False,
        "symlinks": [],
        "legacy_release_score_plane": inspect_legacy_artifact(legacy),
        "raw_pipeline_contract": raw_contract,
        "raw_runner_hash_gated": False,
        "strict_lineage_ready": False,
        "required_runtime_dependencies": sorted(required_dependencies),
        "submission_metadata": metadata,
        "manual_review_note": "The historical legacy Dataset2 score plane is a release inference artifact. Fresh raw Jittor rebuilding exists, but archived member-state parity is not asserted.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict-lineage", action="store_true")
    parser.add_argument("--verify-manifest", action="store_true")
    args = parser.parse_args()
    report = audit(args.root)
    if args.verify_manifest:
        report["manifest"] = verify_manifest(args.root)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite audit: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.strict_lineage:
        raise SystemExit("strict lineage audit intentionally fails until archived legacy member parity is recovered")


if __name__ == "__main__":
    main()
