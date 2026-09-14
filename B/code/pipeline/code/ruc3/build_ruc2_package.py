#!/usr/bin/env python3
"""Build and minimally verify b_ruc2.zip from frozen D3 and new D4 shards."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path


EXPECTED = {"dataset3.csv": 157_670, "dataset4.csv": 2_322_538}


def count_rows(handle: io.BufferedReader) -> int:
    return sum(block.count(b"\n") for block in iter(lambda: handle.read(8 << 20), b""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.baseline) as baseline, zipfile.ZipFile(
        args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
        allowZip64=True,
    ) as output:
        if set(baseline.namelist()) != set(EXPECTED):
            raise ValueError("baseline ZIP members differ")
        with baseline.open("dataset3.csv") as source, output.open(
            "dataset3.csv", "w", force_zip64=True
        ) as destination:
            shutil.copyfileobj(source, destination, length=8 << 20)
        with output.open("dataset4.csv", "w", force_zip64=True) as destination:
            for shard in args.shard:
                with shard.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=8 << 20)

    counts = {}
    samples = {}
    with zipfile.ZipFile(args.output) as archive:
        if set(archive.namelist()) != set(EXPECTED):
            raise ValueError("output ZIP members differ")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member: {bad}")
        for name, expected in EXPECTED.items():
            with archive.open(name) as handle:
                counts[name] = count_rows(handle)
            if counts[name] != expected:
                raise ValueError(f"{name} row count differs: {counts[name]}")
            with archive.open(name) as handle:
                first = next(csv.reader(io.TextIOWrapper(handle, encoding="ascii")))
            values = [float(value) for value in first]
            if len(values) != 100 or abs(sum(values) - 1.0) > 2e-6:
                raise ValueError(f"{name} first row format differs")
            samples[name] = {"columns": len(values), "sum": sum(values)}

    report = {
        "kind": "b_ruc2_submission_package_v1",
        "decision": "PASS",
        "output": str(args.output.resolve()),
        "members": sorted(EXPECTED),
        "row_counts": counts,
        "first_row_samples": samples,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
