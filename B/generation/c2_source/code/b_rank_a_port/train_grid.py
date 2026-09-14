#!/usr/bin/env python3
"""Train one scene's complete three-variant, three-seed member grid."""

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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=("raw", "cf", "hist_cf"),
        default=("raw", "cf", "hist_cf"),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=(20260810, 20260811, 20260812)
    )
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--valid-groups", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--rank-groups", type=int, required=True)
    parser.add_argument("--rank-valid", type=int, required=True)
    parser.add_argument("--rank-epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    data = args.data.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing output reuse: {output_root}")
    data_sha256 = sha256(data)
    if data_sha256 != EXPECTED_DATA_SHA256:
        raise ValueError("official data hash differs")
    identities = [(variant, seed) for variant in args.variants for seed in args.seeds]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate variant/seed identity")

    os.environ["JT_USE_CUDA"] = "1"
    import run

    run.DATA = str(data)
    prepared = run.prepare_training_scene(args.scene)
    output_root.mkdir(parents=True)
    source_hashes = {
        "run.py": sha256(Path(run.__file__).resolve()),
        "train_grid.py": sha256(Path(__file__).resolve()),
    }
    previous = Path.cwd()
    reports = []
    for variant, seed in identities:
        output_dir = output_root / f"{args.scene}_{variant}_{seed}"
        output_dir.mkdir()
        os.chdir(output_dir)
        try:
            validation_mrr = run.fit_scene(
                args.scene,
                args.groups,
                args.epochs,
                args.batch,
                seed,
                variant == "hist_cf",
                False,
                False,
                1e-3,
                "ce",
                args.valid_groups,
                variant != "raw",
                prepared,
            )
            ranker_validation_mrr = run.train_fast_scene(
                args.scene,
                args.rank_groups,
                args.rank_valid,
                args.rank_epochs,
                args.batch,
                seed + 1000,
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
            "variant": variant,
            "seed": seed,
            "segment_fractions": os.environ.get(
                "B_SEGMENT_FRACTIONS", "0.50,0.10,0.15,0.10,0.15"
            ),
            "data_sha256": data_sha256,
            "source_hashes": source_hashes,
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
        report_path = output_dir / "member_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reports.append({"path": str(report_path), **report})
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    summary = {
        "kind": "b_rank_a_port_grid_v1",
        "scene": args.scene,
        "members": reports,
        "source_hashes": source_hashes,
        "data_sha256": data_sha256,
    }
    (output_root / "grid_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
