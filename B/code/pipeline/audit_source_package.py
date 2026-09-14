#!/usr/bin/env python3
"""Fail-closed audit for a source-only reproduction directory or ZIP."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path


FORBIDDEN_SUFFIXES = {".csv", ".pkl", ".pickle", ".pyc", ".pyo", ".npy", ".npz"}
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", "cache", "caches", "data", "answers", "assets"}
FORBIDDEN_NAMES = re.compile(r"(^|[_-])(d3ans|d4ans|answer|answers|submission|prediction|oof)([_().-]|$)", re.I)


def audit_names(names: list[str], sizes: dict[str, int]) -> dict:
    violations: list[str] = []
    for raw in names:
        path = Path(raw.replace("\\", "/"))
        lowered = {part.lower() for part in path.parts[:-1]}
        if lowered & FORBIDDEN_PARTS:
            violations.append(f"forbidden directory: {raw}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden artifact type: {raw}")
        if path.suffix.lower() not in {".py", ".md"} and FORBIDDEN_NAMES.search(path.name):
            violations.append(f"forbidden artifact name: {raw}")
    total = sum(sizes.values())
    return {
        "kind": "source_only_package_audit_v2",
        "decision": "PASS" if not violations and total < 100_000_000 else "FAIL",
        "file_count": len(names),
        "uncompressed_bytes": total,
        "under_100mb_uncompressed": total < 100_000_000,
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--zip", dest="archive", type=Path)
    args = parser.parse_args()
    if (args.root is None) == (args.archive is None):
        raise ValueError("provide exactly one of --root or --zip")
    if args.root:
        root = args.root.resolve()
        files = [path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
        names = [path.relative_to(root).as_posix() for path in files]
        sizes = {name: path.stat().st_size for name, path in zip(names, files)}
    else:
        with zipfile.ZipFile(args.archive) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            sizes = {info.filename: info.file_size for info in infos}
    result = audit_names(names, sizes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
