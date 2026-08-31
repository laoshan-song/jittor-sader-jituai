#!/usr/bin/env python3
"""Fail-closed static audit for the compact exact-inference package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import fitz

EXPECTED = {
    "base_result.zip": "4c8fca6a041a94957b04a2df9f958d76098d06ab67093679fea766c364e3a28f",
    "models/model_bpr32_prod_community_jittor.npz": "449c45c3d32b477efa52410f5ec9024801942e264ba79873b981297005e3d256",
}
OFFICIAL_DATA_SHA256 = "898d3cbc873a446bb372352919ec346dcc0671651ef998a699d3a23d19ef7825"
RESULT_SHA256 = "d36facee996b5d45806dd6e1d80f8a48883e505f57c8d8d842b9626a50e8e7ce"
BLOCKED = {"torch", "tensorflow", "paddle", "mindspore", "mxnet", "sklearn", "keras", "jax", "flax"}
REQUIRED = {
    "README.md",
    "requirements.txt",
    "submission_metadata.json",
    "base_result.zip",
    "models/model_bpr32_prod_community_jittor.npz",
    "experiments/d1_source_support_locked.json",
    "code/main.py",
    "code/README.md",
    "code/build_community_residual_submission.py",
    "code/dataset1/d1_source_support_postprocess.py",
    "code/dataset2/d2_exact_group_postprocess.py",
    "code/tools/audit_exact_package.py",
    "code/tools/build_exact_package.py",
    "code/tools/check_exact_environment.py",
    "code/tools/make_submission_pdf.py",
    "code/tools/submission_report.html",
    "code/tools/prepare_cuda_runtime.sh",
    "code/raw_training/main.py",
    "code/raw_training/dataset1/train_d1_cf_jittor.py",
    "code/raw_training/dataset2/train_vae_jittor.py",
    "code/raw_training/dataset2/train_bpr_jittor.py",
    "code/raw_training/train_community_bpr_jittor.py",
    "code/raw_training/tools/check_raw_pipeline_contract.py",
    "code/raw_training/tools/check_environment.py",
    "run_inference.sh",
    "run_verify.sh",
    "run_train.sh",
    "run_fresh_inference.sh",
    "run_raw.sh",
    "\u63d0\u4ea4\u8bf4\u660e\u6587\u6863.pdf",
    "MANIFEST.sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".", 1)[0])
    return found


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(root: Path, data: Path | None) -> dict[str, object]:
    root = root.resolve()
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(REQUIRED <= actual, "required package members are missing")
    require("result.zip" not in actual, "final result.zip must not be packaged")
    require("data_A.zip" not in actual, "official data must not be packaged")
    require(not any("__pycache__" in name for name in actual), "bytecode cache is packaged")
    for relative, expected in EXPECTED.items():
        require(sha256(root / relative) == expected, f"pinned artifact hash differs: {relative}")
    with zipfile.ZipFile(root / "base_result.zip") as archive:
        require(archive.testzip() is None, "base result CRC check failed")
        require(archive.namelist() == ["dataset1.csv", "dataset2.csv"], "base result members differ")
    source_files = sorted((root / "code").rglob("*.py"))
    found_imports = set().union(*(imports(path) for path in source_files))
    require("jittor" in found_imports, "Jittor import is absent")
    require(not (found_imports & BLOCKED), "blocked deep-learning framework import found")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    require("jittor==1.3.10.0" in requirements, "Jittor requirement differs")
    require("pandas==2.2.3" in requirements, "Pandas requirement differs")
    require("numba==0.66.0" in requirements, "Numba requirement differs")
    require("jittor-geometric>=0.1.0" in requirements, "JittorGeometric requirement differs")
    require("scikit-learn==1.5.2" in requirements, "scikit-learn requirement differs")
    require("nvidia-cudnn-cu12==8.9.7.29" in requirements, "cuDNN requirement differs")
    require("weasyprint==66.0" in requirements, "WeasyPrint requirement differs")
    require("PyMuPDF==1.25.5" in requirements, "PyMuPDF requirement differs")
    require("matplotlib==3.10.3" in requirements, "matplotlib requirement differs")
    report_html = (root / "code/tools/submission_report.html").read_text(encoding="utf-8")
    require(report_html.count("<svg") == 0, "journal report must not contain decorative SVG figures")
    require("<img" not in report_html.lower(), "journal report HTML contains raster images")
    require("class=\"running\"" not in report_html, "journal report contains obsolete running headers")
    require("data-latex=" in report_html, "journal report must retain LaTeX equation sources")
    chinese_characters = len(re.findall(r"[\u4e00-\u9fff]", report_html))
    require(chinese_characters >= 15_000, f"journal report is too short: {chinese_characters} Chinese characters")
    report_pdf_path = root / "提交说明文档.pdf"
    report_pdf = report_pdf_path.read_bytes()
    require(report_pdf.startswith(b"%PDF-"), "submission document is not a PDF")
    with fitz.open(report_pdf_path) as document:
        require(document.page_count >= 10, "submission document must contain at least ten pages")
        extracted = []
        for index, page in enumerate(document):
            require(not page.get_images(full=True), f"submission document contains a raster image on page {index + 1}")
            blocks = [block for block in page.get_text("blocks") if block[4].strip()]
            require(bool(blocks), f"submission document page {index + 1} is blank")
            for block in blocks:
                require(
                    page.rect.contains(fitz.Rect(block[:4])),
                    f"submission document text escapes page bounds on page {index + 1}",
                )
            extracted.append(page.get_text("text"))
    expected_chinese = len(re.findall(r"[\u4e00-\u9fff]", report_html))
    actual_chinese = len(re.findall(r"[\u4e00-\u9fff]", "".join(extracted)))
    require(
        actual_chinese >= int(expected_chinese * 0.98),
        f"submission document Chinese coverage is too low: {actual_chinese}/{expected_chinese}",
    )
    builder = (root / "code/build_community_residual_submission.py").read_text(encoding="utf-8")
    require(OFFICIAL_DATA_SHA256 in builder, "official archive hash pin is absent")
    require(RESULT_SHA256 in builder, "final result hash pin is absent")
    metadata = json.loads((root / "submission_metadata.json").read_text(encoding="utf-8"))
    require(metadata == {
        "contact": "\u5b8b\u6587\u97ec", "phone": "18984157192", "rank": "7",
        "score": "1.521072794155721", "team": "sader", "track": "contest1",
        "wechat": "laoshan_song",
    }, "submission metadata differs")
    data_hash = None
    if data is not None:
        require(data.is_file(), f"official archive does not exist: {data}")
        data_hash = sha256(data)
        require(data_hash == OFFICIAL_DATA_SHA256, "official archive hash differs")
    return {
        "decision": "PASS",
        "package_root": str(root),
        "official_data_sha256": data_hash,
        "required_file_count": len(REQUIRED),
        "source_imports": sorted(found_imports),
        "deep_learning_framework": "Jittor",
        "result_sha256_to_reconstruct": RESULT_SHA256,
        "official_labels_opened": False,
        "final_result_packaged": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.data)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
