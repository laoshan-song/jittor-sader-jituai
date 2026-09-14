#!/usr/bin/env python3
"""Train one B-rank graph/temporal member and its structural ranker."""

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED_DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scene", choices=("dataset3", "dataset4"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("cf", "hist_cf", "raw"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--groups", type=int, default=120000)
    parser.add_argument("--valid-groups", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--rank-groups", type=int, default=160000)
    parser.add_argument("--rank-valid", type=int, default=30000)
    parser.add_argument("--rank-epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing output reuse: {output_dir}")
    output_dir.mkdir(parents=True)
    os.environ["JT_USE_CUDA"] = "1"

    import run

    run.DATA = str(args.data.resolve())
    data_sha256 = sha256(args.data.resolve())
    if data_sha256 != EXPECTED_DATA_SHA256:
        raise ValueError("official data hash differs")
    previous = Path.cwd()
    os.chdir(output_dir)
    try:
        use_cf = args.variant != "raw"
        use_hist = args.variant == "hist_cf"
        prepared = run.prepare_training_scene(args.scene)
        validation_mrr = run.fit_scene(
            args.scene,
            args.groups,
            args.epochs,
            args.batch,
            args.seed,
            use_hist,
            False,
            False,
            1e-3,
            "ce",
            args.valid_groups,
            use_cf,
            prepared,
        )
        ranker_validation_mrr = run.train_fast_scene(
            args.scene,
            args.rank_groups,
            args.rank_valid,
            args.rank_epochs,
            args.batch,
            args.seed + 1000,
            prepared,
        )
    finally:
        os.chdir(previous)

    suffix = args.scene[-1]
    model = output_dir / f"m{suffix}.pkl"
    ranker = output_dir / f"r{suffix}.pkl"
    report = {
        "kind": "b_rank_a_port_member_v1",
        "scene": args.scene,
        "variant": args.variant,
        "seed": args.seed,
        "segment_fractions": os.environ.get(
            "B_SEGMENT_FRACTIONS", "0.50,0.10,0.15,0.10,0.15"
        ),
        "data_sha256": data_sha256,
        "source_hashes": {
            "run.py": sha256(Path(run.__file__).resolve()),
            "train_member.py": sha256(Path(__file__).resolve()),
        },
        "validation_mrr": validation_mrr,
        "ranker_validation_mrr": ranker_validation_mrr,
        "model_sha256": sha256(model),
        "ranker_sha256": sha256(ranker),
        "config": {
            "groups": args.groups,
            "valid_groups": args.valid_groups,
            "epochs": args.epochs,
            "rank_groups": args.rank_groups,
            "rank_valid": args.rank_valid,
            "rank_epochs": args.rank_epochs,
            "batch": args.batch,
        },
    }
    (output_dir / "member_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
