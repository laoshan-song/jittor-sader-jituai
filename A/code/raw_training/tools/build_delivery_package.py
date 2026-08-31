#!/usr/bin/env python3
"""Fail-closed builder for a fully auditable Track 1 delivery ZIP.

The contest requires raw-data reproducibility. A retained release score plane
or an old strict-inference receipt is not sufficient authorization to build a
delivery archive.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


EXPECTED_RESULT_SHA256 = "d36facee996b5d45806dd6e1d80f8a48883e505f57c8d8d842b9626a50e8e7ce"
PACKAGE_NAME = "contest1_sader_007.zip"
ROOT_FILES = (
    "README.md",
    "requirements.txt",
    "submission_metadata.json",
    "run_train.sh",
    "run_inference.sh",
    "run_verify.sh",
    "run_raw.sh",
    "run_fresh_inference.sh",
    "提交说明文档.pdf",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_receipt(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("release receipt must be a JSON object")
    digest = value.get("result_sha256", value.get("sha256"))
    if digest != EXPECTED_RESULT_SHA256:
        raise ValueError("release receipt does not attest the recorded A-list result SHA-256")
    if value.get("strict_release") is not True:
        raise ValueError("release receipt must record strict_release=true")
    if value.get("kind") != "track1_strict_release_inference_receipt_v2":
        raise ValueError("release receipt has no source-and-model identity binding")
    for key in ("code_file_sha256", "release_model_file_sha256"):
        inventory = value.get(key)
        if not isinstance(inventory, dict) or not inventory:
            raise ValueError(f"release receipt lacks {key}")
    if not isinstance(value.get("requirements_sha256"), str):
        raise ValueError("release receipt lacks requirements identity")
    return value


def ignored(relative: Path) -> bool:
    return any(part == "__pycache__" for part in relative.parts) or relative.suffix in {".pyc", ".pyo"}


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if ignored(relative):
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        elif path.is_symlink():
            raise ValueError(f"delivery source contains a symlink: {relative}")


def write_manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest:
            relative = "./" + str(path.relative_to(root)).replace("\\", "/")
            rows.append(f"{sha256_file(path)}  {relative}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def file_inventory(root: Path, suffixes: set[str] | None = None) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and (suffixes is None or path.suffix in suffixes)
    }


def require_receipt_identity(source: Path, receipt: dict[str, object]) -> None:
    expected_code = receipt["code_file_sha256"]
    expected_models = receipt["release_model_file_sha256"]
    actual_code = file_inventory(source / "code", {".py", ".json"})
    actual_models = file_inventory(source / "code" / "artifacts" / "release_models")
    if expected_code != actual_code:
        raise ValueError("release receipt code inventory differs from delivery source")
    if expected_models != actual_models:
        raise ValueError("release receipt model inventory differs from delivery source")
    if receipt["requirements_sha256"] != sha256_file(source / "requirements.txt"):
        raise ValueError("release receipt requirements hash differs from delivery source")


def load_audit_module(source: Path):
    location = source / "code" / "tools" / "audit_package.py"
    spec = importlib.util.spec_from_file_location("delivery_audit", location)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load package audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(source: Path, receipt_path: Path, output: Path) -> None:
    if output.name != PACKAGE_NAME:
        raise ValueError(f"output must be named {PACKAGE_NAME}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    receipt = load_receipt(receipt_path)
    source = source.resolve()
    missing = [name for name in ROOT_FILES if not (source / name).is_file()]
    if missing or not (source / "code").is_dir():
        raise FileNotFoundError("delivery source is incomplete: " + ", ".join(missing))
    require_receipt_identity(source, receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="contest1_sader_007_") as temporary:
        root = Path(temporary)
        for name in ROOT_FILES:
            shutil.copy2(source / name, root / name)
        copy_tree(source / "code", root / "code")
        (root / "release_inference_verification.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_manifest(root)
        audit = load_audit_module(root)
        report = audit.audit(root)
        report["manifest"] = audit.verify_manifest(root)
        if report.get("strict_lineage_ready") is not True:
            raise RuntimeError(
                "raw historical lineage is incomplete; refusing to package a "
                "release-score reconstruction as a contest delivery"
            )
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.release_receipt, args.output)
    print(json.dumps({"package": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
