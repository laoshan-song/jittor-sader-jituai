#!/usr/bin/env python3
"""Build a larger D4 replay score cache from the frozen model inventory.

This intentionally reuses checkpoints listed in an existing cache manifest.  It
does not train or rerun the v12 stack; only inference is expanded to additional
time-ordered replay rows so a new rank-aligned model can be trained separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--output-cache", type=Path, required=True)
    result.add_argument("--strategy", choices=("history", "test_pool"), required=True)
    result.add_argument("--valid-groups", type=int, default=120_000)
    result.add_argument("--confirm-groups", type=int, default=30_000)
    result.add_argument("--batch-rows", type=int, default=512)
    return result


def main() -> int:
    args = parser().parse_args()
    manifest = json.loads(args.source_manifest.read_text())
    metadata = manifest.get("metadata", manifest)
    command = [
        sys.executable,
        "-m",
        "b_rank.d4_multimodel_fit",
        "--data",
        str(args.data),
        "--cache-dir",
        str(args.cache_dir),
        "--run-dir",
        str(args.run_dir),
        "--batch-rows",
        str(args.batch_rows),
        "--group-seed",
        str(metadata["group_seed"]),
        "--valid-groups",
        str(args.valid_groups),
        "--confirm-groups",
        str(args.confirm_groups),
        "--score-cache-only",
        "--score-cache-dir",
        str(args.output_cache),
        "--score-only-strategy",
        args.strategy,
    ]
    for record in metadata["temporal_models"]:
        command.extend(
            (
                "--temporal",
                record["name"],
                str(record["history_size"]),
                record["path"],
            )
        )
    for record in metadata["mf_models"]:
        command.extend(("--mf", record["name"], record["path"]))
    for record in metadata["transition_mf_models"]:
        command.extend(("--transition-mf", record["name"], record["path"]))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
