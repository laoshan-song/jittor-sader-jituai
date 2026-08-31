#!/usr/bin/env python3
"""Single command entry point for the Track 1 reproduction package.

The ``verify`` command is the recommended path for the recorded A-list
result.  It runs the manifest and static audits before reconstructing the
submission.  The remaining commands delegate to the retained raw Jittor
training and inference protocol without changing its arguments or outputs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(script: str, *arguments: Path | str) -> None:
    """Run one documented package launcher with this Python interpreter."""
    environment = os.environ.copy()
    environment.setdefault("PYTHON", sys.executable)
    command = ["bash", str(ROOT / script), *(str(argument) for argument in arguments)]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def add_data_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, required=True, help="official data_A.zip")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track 1 Jittor reproduction entry point",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="audit and exactly reconstruct the recorded result")
    add_data_output(verify)

    infer = commands.add_parser("infer", help="reconstruct the recorded result without static audit")
    add_data_output(infer)

    train = commands.add_parser("train", help="train retained raw Jittor components from official data")
    train.add_argument("--data", type=Path, required=True, help="official data_A.zip")
    train.add_argument("--models", type=Path, required=True, help="new model output directory")
    train.add_argument("--dataset", choices=("dataset1", "dataset2", "all"), default="all")
    train.add_argument("--cuda", action="store_true", help="require CUDA-capable Jittor")

    fresh = commands.add_parser("fresh-infer", help="run fresh inference from a raw Jittor model directory")
    fresh.add_argument("--data", type=Path, required=True, help="official data_A.zip")
    fresh.add_argument("--models", type=Path, required=True, help="raw Jittor model directory")
    fresh.add_argument("--output", type=Path, required=True, help="new inference output directory")

    raw = commands.add_parser("raw", help="run raw Jittor training and fresh inference with receipts")
    raw.add_argument("--data", type=Path, required=True, help="official data_A.zip")
    raw.add_argument("--models", type=Path, required=True, help="new raw model output directory")
    raw.add_argument("--output", type=Path, required=True, help="new inference output directory")

    args = parser.parse_args()
    if args.command == "verify":
        run_script("run_verify.sh", args.data, args.output)
    elif args.command == "infer":
        run_script("run_inference.sh", args.data, args.output)
    elif args.command == "train":
        command = [args.data, args.models, "--dataset", args.dataset]
        if args.cuda:
            command.append("--cuda")
        run_script("run_train.sh", *command)
    elif args.command == "fresh-infer":
        run_script("run_fresh_inference.sh", args.data, args.models, args.output)
    elif args.command == "raw":
        run_script("run_raw.sh", args.data, args.models, args.output)
    else:  # argparse constrains the command, but keep this fail-closed.
        raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
