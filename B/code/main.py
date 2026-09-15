#!/usr/bin/env python3
"""Two public B-list reproduction commands."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("verify", "reproduce"),
        help=(
            "verify reconstructs the retained byte-exact result; reproduce "
            "rebuilds and aligns the complete fresh data_B.zip chain"
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu",
        type=int,
        help=(
            "optional physical GPU index; omitted preserves CUDA_VISIBLE_DEVICES "
            "or uses the first CUDA-visible device"
        ),
    )
    args = parser.parse_args()
    scripts = {
        "verify": ["run_verify.sh", args.data, args.output],
        "reproduce": ["run_reproduce.sh", args.data, args.output],
    }
    command = scripts[args.command]
    if args.gpu is not None:
        command.append(args.gpu)
    return subprocess.run(
        ["bash", str(ROOT / command[0]), *map(str, command[1:])],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
