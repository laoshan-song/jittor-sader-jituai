#!/usr/bin/env python3
"""Train one dataset1 high-order collaborative member with Jittor."""

import argparse
import os
from pathlib import Path

import jittor as jt

import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--groups", type=int, default=80000)
    parser.add_argument("--valid", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is required")
    run.DATA = str(args.data.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(output.parent)
    try:
        score = run.fit_scene(
            "dataset1",
            args.groups,
            args.epochs,
            args.batch,
            args.seed,
            False,
            False,
            False,
            1e-3,
            "ce",
            args.valid,
            True,
        )
        Path("m1.pkl").replace(output)
    finally:
        os.chdir(previous)
    print(f"validation_mrr={score:.9f} wrote={output}", flush=True)


if __name__ == "__main__":
    main()
