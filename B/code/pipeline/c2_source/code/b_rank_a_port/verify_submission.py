#!/usr/bin/env python3
"""Fail-closed verifier for the B-rank multi-model submission."""

import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path


EXPECTED_DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
MEMBERS = ("dataset3.csv", "dataset4.csv")
TEST_MEMBERS = ("dataset3/test.csv", "dataset4/test.csv")
DECIMAL = re.compile(rb"(?:0|[1-9][0-9]*)\.[0-9]{8}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def official_rows(data: Path) -> dict[str, int]:
    output = {}
    with zipfile.ZipFile(data) as archive:
        for member in TEST_MEMBERS:
            with archive.open(member) as handle:
                header = handle.readline()
                require(header.startswith(b"src,time,c1,"), f"invalid header: {member}")
                output[member.split("/")[0]] = sum(1 for line in handle if line.strip())
    return output


def verify_member(archive: zipfile.ZipFile, name: str, expected_rows: int) -> dict:
    rows = 0
    maximum_sum_error = 0.0
    with archive.open(name) as handle:
        for rows, line in enumerate(handle, start=1):
            require(line.endswith(b"\n"), f"{name} row {rows} is not newline terminated")
            fields = line[:-1].split(b",")
            require(len(fields) == 100, f"{name} row {rows} has {len(fields)} fields")
            values = []
            for field in fields:
                require(DECIMAL.fullmatch(field) is not None, f"{name} row {rows} has invalid decimal")
                value = float(field)
                require(math.isfinite(value) and value >= 0.0, f"{name} row {rows} has invalid probability")
                values.append(value)
            error = abs(sum(values) - 1.0)
            maximum_sum_error = max(maximum_sum_error, error)
            require(error <= 1e-5, f"{name} row {rows} sums to {sum(values):.12f}")
    require(rows == expected_rows, f"{name} has {rows} rows, expected {expected_rows}")
    return {"rows": rows, "maximum_sum_error": maximum_sum_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    data = args.data.resolve()
    submission = args.submission.resolve()
    manifest_path = args.manifest.resolve()
    require(sha256(data) == EXPECTED_DATA_SHA256, "official data hash differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("kind") == "b_rank_a_port_submission_v1", "manifest kind differs")
    require(manifest.get("data_sha256") == EXPECTED_DATA_SHA256, "manifest data hash differs")
    require(manifest.get("submission_sha256") == sha256(submission), "manifest submission hash differs")
    source_hashes = manifest.get("source_hashes")
    require(isinstance(source_hashes, dict) and source_hashes, "manifest source hashes missing")
    for name, expected in source_hashes.items():
        path = args.source_root.resolve() / name
        require(path.is_file(), f"source file missing: {name}")
        require(sha256(path) == expected, f"source hash differs: {name}")

    expected_rows = official_rows(data)
    with zipfile.ZipFile(submission) as archive:
        require(tuple(archive.namelist()) == MEMBERS, "submission members or order differ")
        require(archive.testzip() is None, "submission CRC failed")
        results = {
            name: verify_member(archive, name, expected_rows[name[:-4]])
            for name in MEMBERS
        }
    require(manifest.get("row_counts") == expected_rows, "manifest row counts differ")
    output = {
        "kind": "b_rank_a_port_verification_v1",
        "decision": "PASS",
        "data_sha256": EXPECTED_DATA_SHA256,
        "submission_sha256": sha256(submission),
        "members": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
