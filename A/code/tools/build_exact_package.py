#!/usr/bin/env python3
"""Create a deterministic outer ZIP for the compact exact package."""

from __future__ import annotations

import argparse
import stat
import zipfile
from pathlib import Path


LIMIT = 100 * 1024 * 1024


def info(name: str, mode: int, directory: bool = False) -> zipfile.ZipInfo:
    value = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    value.create_system = 3
    value.external_attr = ((mode & 0o777) << 16) | (0x10 if directory else 0)
    value.compress_type = zipfile.ZIP_STORED if directory else zipfile.ZIP_DEFLATED
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise ValueError(f"source directory does not exist: {root}")
    if output.exists():
        raise FileExistsError(output)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    relative = [path.relative_to(root).as_posix() for path in files]
    if "result.zip" in relative or "data_A.zip" in relative:
        raise ValueError("source must not package final result or official data")
    directories = {root.name + "/"}
    for name in relative:
        parts = name.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add(root.name + "/" + "/".join(parts[:index]) + "/")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(directories):
            archive.writestr(info(name, 0o755, directory=True), b"")
        for path, name in zip(files, relative):
            entry = info(root.name + "/" + name, stat.S_IMODE(path.stat().st_mode))
            if path.suffix.lower() == ".zip":
                entry.compress_type = zipfile.ZIP_STORED
            archive.writestr(entry, path.read_bytes())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("outer ZIP failed CRC validation")
    if output.stat().st_size >= LIMIT:
        raise RuntimeError(f"package exceeds 100 MiB: {output.stat().st_size}")
    print(f"wrote {output} bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
