#!/usr/bin/env python3
"""Generate the deterministic source-only package manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "SOURCE_MANIFEST.json"
DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path != OUTPUT
    )
    payload = {
        "kind": "d3d4_official_source_manifest_v2",
        "official_data_sha256": DATA_SHA256,
        "uses_teacher_answers": False,
        "uses_answer_derived_cache": False,
        "contains_official_data": False,
        "contains_model_weights": False,
        "files": {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "manifest": str(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
