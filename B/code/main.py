#!/usr/bin/env python3
"""Single command entry point for audit, inference, and training."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "infer", "train", "fresh-infer", "generate-base"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", type=Path)
    args = parser.parse_args()
    scripts = {
        "verify": ["run_verify.sh", args.data, args.output, args.gpu],
        "infer": ["run_inference.sh", args.data, args.output, args.gpu],
        "train": ["run_train.sh", args.data, args.output, args.gpu],
        "fresh-infer": ["run_fresh_inference.sh", args.data, args.models, args.output, args.gpu],
        "generate-base": ["run_generate_base.sh", args.data, args.output, args.gpu],
    }
    command = scripts[args.command]
    if args.command == "fresh-infer" and args.models is None:
        parser.error("fresh-infer requires --models")
    return subprocess.run(["bash", str(ROOT / command[0]), *map(str, command[1:])], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
