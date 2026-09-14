#!/usr/bin/env python3
"""Fail-closed standard-library verifier for a v26 submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
DECIMAL = re.compile(rb"(?:0|[1-9][0-9]*)\.[0-9]{8}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(data: Path) -> dict[str, int]:
    result = {}
    with zipfile.ZipFile(data) as archive:
        for scene in ("dataset3", "dataset4"):
            with archive.open(f"{scene}/test.csv") as handle:
                header = handle.readline()
                if not header.startswith(b"src,time,c1,"):
                    raise ValueError(f"invalid official header: {scene}")
                result[scene] = sum(bool(line.strip()) for line in handle)
    return result


def check_csv(archive: zipfile.ZipFile, name: str, expected: int) -> dict:
    count = 0
    maximum_error = 0.0
    with archive.open(name) as handle:
        for count, line in enumerate(handle, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"{name} row {count} is not newline terminated")
            fields = line[:-1].split(b",")
            if len(fields) != 100:
                raise ValueError(f"{name} row {count} has {len(fields)} fields")
            values = []
            for field in fields:
                if DECIMAL.fullmatch(field) is None:
                    raise ValueError(f"{name} row {count} has invalid decimal")
                value = float(field)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} row {count} has invalid probability")
                values.append(value)
            maximum_error = max(maximum_error, abs(sum(values) - 1.0))
            if maximum_error > 1e-5:
                raise ValueError(f"{name} row {count} does not sum to one")
    if count != expected:
        raise ValueError(f"{name} has {count} rows, expected {expected}")
    return {"rows": count, "maximum_sum_error": maximum_error}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.data.resolve()) != DATA_SHA256:
        raise ValueError("official data hash differs")
    expected = rows(args.data.resolve())
    with zipfile.ZipFile(args.submission.resolve()) as archive:
        if tuple(archive.namelist()) != ("dataset3.csv", "dataset4.csv"):
            raise ValueError("submission root must contain dataset3.csv,dataset4.csv")
        if archive.testzip() is not None:
            raise ValueError("submission CRC failed")
        result = {scene: check_csv(archive, f"{scene}.csv", expected[scene])
                  for scene in expected}
    output = {"decision": "PASS", "data_sha256": DATA_SHA256,
              "submission_sha256": sha256(args.submission.resolve()),
              "members": result}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
