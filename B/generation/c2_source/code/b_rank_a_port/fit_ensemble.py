#!/usr/bin/env python3
"""Fit scene-specific multi-model fusion and gate it on untouched confirmation."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ["JT_USE_CUDA"] = "1"

import ensemble_core as core


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
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta-groups", type=int, default=50000)
    parser.add_argument("--valid-groups", type=int, default=30000)
    parser.add_argument("--confirm-groups", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing output reuse: {args.output}")
    if sha256(args.data.resolve()) != EXPECTED_DATA_SHA256:
        raise ValueError("official data hash differs")
    model_dirs = [path.resolve() for path in args.models]
    member_reports = []
    identities = set()
    labels = set()
    for model_dir in model_dirs:
        report_path = model_dir / "member_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"invalid member directory: {model_dir}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("kind") != "b_rank_a_port_member_v1"
            or report.get("scene") != args.scene
            or report.get("data_sha256") != EXPECTED_DATA_SHA256
        ):
            raise ValueError(f"invalid member report: {report_path}")
        suffix = args.scene[-1]
        if sha256(model_dir / f"m{suffix}.pkl") != report.get("model_sha256"):
            raise ValueError(f"base model hash differs: {model_dir}")
        if sha256(model_dir / f"r{suffix}.pkl") != report.get("ranker_sha256"):
            raise ValueError(f"ranker hash differs: {model_dir}")
        identity = (report.get("variant"), report.get("seed"))
        if identity in identities or model_dir.name in labels:
            raise ValueError("duplicate member identity or directory label")
        identities.add(identity)
        labels.add(model_dir.name)
        member_reports.append(report)
    seeds = {int(report["seed"]) for report in member_reports}
    required_seeds = {20260810, 20260811, 20260812}
    required_identities = {
        (variant, seed)
        for variant in ("raw", "cf", "hist_cf")
        for seed in required_seeds
    }
    if seeds != required_seeds or identities != required_identities:
        raise ValueError("ensemble requires the complete three-variant, three-seed grid")
    source_hash_sets = {
        tuple(sorted(report.get("source_hashes", {}).items()))
        for report in member_reports
    }
    if len(source_hash_sets) != 1 or not next(iter(source_hash_sets)):
        raise ValueError("member source snapshots differ or are missing")

    (
        _train,
        _test,
        max_node,
        use_src_freq,
        pool,
        freq,
        src_freq,
        initial_history,
        segments,
    ) = core.scene_data(args.data, args.scene)

    segment_groups = {
        "meta_train": args.meta_groups,
        "validation": args.valid_groups,
        "confirmation": args.confirm_groups,
    }
    scored = {}
    component_names = None
    for offset, name in enumerate(("meta_train", "validation", "confirmation")):
        src, time, candidates, labels = core.sample_segment(
            segments, pool, name, segment_groups[name], args.seed + offset
        )
        names, components, strata = core.component_scores(
            scene=args.scene,
            model_dirs=model_dirs,
            history=core.segment_history(initial_history, segments, name),
            max_node=max_node,
            use_src_freq=use_src_freq,
            freq=freq,
            src_freq=src_freq,
            src=src,
            time=time,
            candidates=candidates,
            labels=labels,
            batch=args.batch,
        )
        if component_names is None:
            component_names = names
        elif names != component_names:
            raise ValueError("component inventory changed across segments")
        scored[name] = {"components": components, "labels": labels, "strata": strata}

    meta_weights, meta_mrr = core.tune_convex(
        scored["meta_train"]["components"], scored["meta_train"]["labels"]
    )
    validation_components = scored["validation"]["components"]
    validation_labels = scored["validation"]["labels"]
    candidates = {"meta_fitted": meta_weights}
    for index, name in enumerate(component_names):
        weights = np.zeros(len(component_names), dtype=np.float64)
        weights[index] = 1.0
        candidates[f"single:{name}"] = weights
    selected_name, selected_weights = max(
        candidates.items(),
        key=lambda item: core.mrr(
            core.mixed_score(item[1], validation_components), validation_labels
        ),
    )

    report_segments = {}
    for name, values in scored.items():
        mixed = core.mixed_score(selected_weights, values["components"])
        segment_report = {}
        masks = {
            "overall": np.ones(len(values["labels"]), dtype=bool),
            "source_hot": values["strata"]["source_hot"],
            "pair_seen": values["strata"]["pair_seen"],
        }
        for stratum, mask in masks.items():
            if not np.asarray(mask).any():
                raise ValueError(f"empty {stratum} stratum in {name}")
            individual = {
                component: core.mrr(score, values["labels"], mask)
                for component, score in zip(component_names, values["components"])
            }
            ensemble = core.mrr(mixed, values["labels"], mask)
            best = max(individual.values())
            segment_report[stratum] = {
                "rows": int(np.asarray(mask).sum()),
                "individual_mrr": individual,
                "best_individual_mrr": best,
                "ensemble_mrr": ensemble,
                "ensemble_delta_vs_best": ensemble - best,
            }
        segment_report["source_hot"]["threshold"] = float(
            values["strata"]["source_hot_threshold"]
        )
        report_segments[name] = segment_report

    confirmation = report_segments["confirmation"]
    decision_checks = {
        "overall": confirmation["overall"]["ensemble_delta_vs_best"] >= -0.0005,
        "source_hot": confirmation["source_hot"]["ensemble_delta_vs_best"] >= -0.005,
        "pair_seen": confirmation["pair_seen"]["ensemble_delta_vs_best"] >= -0.005,
    }
    decision = "PASS" if all(decision_checks.values()) else "NO_GO"
    report = {
        "kind": "b_rank_a_port_ensemble_v1",
        "decision": decision,
        "scene": args.scene,
        "segment_fractions": os.environ.get(
            "B_SEGMENT_FRACTIONS", "0.50,0.10,0.15,0.10,0.15"
        ),
        "component_names": component_names,
        "meta_fitted_mrr": meta_mrr,
        "selected": selected_name,
        "active_component_count": int(np.count_nonzero(selected_weights > 1e-12)),
        "weights": {
            name: float(weight)
            for name, weight in zip(component_names, selected_weights)
            if weight > 1e-12
        },
        "segments": report_segments,
        "decision_checks": decision_checks,
        "members": [
            {
                "path": str(path),
                "report_sha256": sha256(path / "member_report.json"),
                "variant": report["variant"],
                "seed": int(report["seed"]),
                "model_sha256": report["model_sha256"],
                "ranker_sha256": report["ranker_sha256"],
            }
            for path, report in zip(model_dirs, member_reports)
        ],
        "config": {
            "meta_groups": args.meta_groups,
            "valid_groups": args.valid_groups,
            "confirm_groups": args.confirm_groups,
            "seed": args.seed,
            "batch": args.batch,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if decision != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
