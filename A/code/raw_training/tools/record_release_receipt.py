#!/usr/bin/env python3
"""Record a strict release-inference verification receipt for packaging."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
import main as track_main  # noqa: E402


EXPECTED_DATA_SHA256 = "898d3cbc873a446bb372352919ec346dcc0671651ef998a699d3a23d19ef7825"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(root: Path, suffixes: set[str] | None = None) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and (suffixes is None or path.suffix in suffixes)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    result = args.result.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite release receipt: {output}")
    data_sha256 = sha256_file(data)
    if data_sha256 != EXPECTED_DATA_SHA256:
        raise ValueError("official data archive hash differs from the recorded A-list input")
    report = track_main.validate_submission(result, strict_release=True)
    release_models = CODE_ROOT / "artifacts" / "release_models"
    receipt = {
        "kind": "track1_strict_release_inference_receipt_v2",
        "strict_release": True,
        "official_data_sha256": data_sha256,
        "result_filename": result.name,
        "result_sha256": report["sha256"],
        "rows": report["rows"],
        "code_file_sha256": file_inventory(CODE_ROOT, {".py", ".json"}),
        "release_model_file_sha256": file_inventory(release_models),
        "requirements_sha256": sha256_file(CODE_ROOT.parent / "requirements.txt"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
