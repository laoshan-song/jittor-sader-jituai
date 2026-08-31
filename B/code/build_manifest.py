#!/usr/bin/env python3
"""Write deterministic SHA-256 manifest for the package root."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path != OUTPUT)
    OUTPUT.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

