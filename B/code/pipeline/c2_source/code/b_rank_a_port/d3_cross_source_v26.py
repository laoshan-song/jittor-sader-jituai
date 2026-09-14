#!/usr/bin/env python3
"""Validate and deploy a sparse same-time cross-source residual for D3."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

os.environ["JT_USE_CUDA"] = "1"

import ensemble_core as core
import run


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
V21_ZIP_SHA256 = "d3b7de07c64311e80577ec753cd5572ee5edecafef5d829210147172045fe1ec"
V21_D4_SHA256 = "11283eb88c87751917cbdaadac80dd1e55a32cec0ac298c222e78e520a42aea1"
ROWS = {"dataset3": 157_670, "dataset4": 2_322_538}
WIDTH = 100
WEIGHTS = np.arange(0.0, 0.1001, 0.0025)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + np.float32(1e-6)
    )


def reciprocal_ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positive = scores[np.arange(len(labels)), labels]
    columns = np.arange(scores.shape[1])[None, :]
    rank = 1 + (scores > positive[:, None]).sum(axis=1)
    rank += ((scores == positive[:, None]) & (columns < labels[:, None])).sum(axis=1)
    return 1.0 / rank.astype(np.float64)


def cross_source_support(
    pool_src: np.ndarray,
    pool_time: np.ndarray,
    pool_candidates: np.ndarray,
    query_src: np.ndarray,
    query_time: np.ndarray,
    query_candidates: np.ndarray,
) -> np.ndarray:
    maximum = int(max(pool_candidates.max(), query_candidates.max()))
    item_base = np.uint64(maximum + 1)
    sorted_items = np.sort(pool_candidates, axis=1, kind="stable")
    distinct = np.empty(sorted_items.shape, dtype=bool)
    distinct[:, 0] = True
    distinct[:, 1:] = sorted_items[:, 1:] != sorted_items[:, :-1]
    rows = np.repeat(
        np.arange(len(sorted_items), dtype=np.int32),
        distinct.sum(axis=1, dtype=np.int32),
    )
    items = sorted_items[distinct].astype(np.uint64, copy=False)

    time_item = pool_time[rows].astype(np.uint64) * item_base + items
    time_item_ids, time_item_counts = np.unique(time_item, return_counts=True)
    source_time = (
        pool_src.astype(np.uint64) << np.uint64(32)
    ) | pool_time.astype(np.uint64)
    source_time_ids, source_time_group = np.unique(source_time, return_inverse=True)
    source_time_item = source_time_group[rows].astype(np.uint64) * item_base + items
    own_ids, own_counts = np.unique(source_time_item, return_counts=True)

    query_time_item = (
        query_time.astype(np.uint64)[:, None] * item_base
        + query_candidates.astype(np.uint64, copy=False)
    )
    total = lookup(time_item_ids, time_item_counts, query_time_item)
    query_source_time = (
        query_src.astype(np.uint64) << np.uint64(32)
    ) | query_time.astype(np.uint64)
    group = np.searchsorted(source_time_ids, query_source_time)
    matched = group < len(source_time_ids)
    matched[matched] &= source_time_ids[group[matched]] == query_source_time[matched]
    group[~matched] = len(source_time_ids)
    query_own = group.astype(np.uint64)[:, None] * item_base
    query_own = query_own + query_candidates.astype(np.uint64, copy=False)
    support = total.astype(np.int32) - lookup(own_ids, own_counts, query_own).astype(
        np.int32
    )
    if support.min(initial=0) < 0:
        raise ValueError("cross-source support became negative")
    return support


def lookup(ids: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    flat = np.asarray(query).reshape(-1)
    positions = np.searchsorted(ids, flat)
    inside = positions < len(ids)
    matched = np.zeros(len(flat), dtype=bool)
    matched[inside] = ids[positions[inside]] == flat[inside]
    output = np.zeros(len(flat), dtype=values.dtype)
    output[matched] = values[positions[matched]]
    return output.reshape(query.shape)


def pair_seen(history: np.ndarray, src: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    keys = history[:, 0].astype(np.int64) * run.BASE + history[:, 1].astype(np.int64)
    keys = np.unique(keys)
    query = src[:, None].astype(np.int64) * run.BASE + candidates.astype(np.int64)
    positions = np.searchsorted(keys, query)
    inside = positions < len(keys)
    output = np.zeros(query.shape, dtype=bool)
    output[inside] = keys[positions[inside]] == query[inside]
    return output


def metrics(
    baseline: np.ndarray, candidate: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    before = reciprocal_ranks(baseline, labels)
    after = reciprocal_ranks(candidate, labels)
    delta = after - before
    return {
        "baseline_mrr": float(before.mean()),
        "candidate_mrr": float(after.mean()),
        "delta": float(delta.mean()),
        "delta_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "positive_row_rate": float(np.mean(delta > 0.0)),
        "negative_row_rate": float(np.mean(delta < 0.0)),
        "top1_changed": float(
            np.mean(np.argmax(baseline, axis=1) != np.argmax(candidate, axis=1))
        ),
    }


def apply_policy(
    baseline: np.ndarray,
    feature: np.ndarray,
    seen: np.ndarray,
    weight: float,
    gate: str,
) -> np.ndarray:
    active = np.ones(seen.shape, dtype=bool)
    if gate == "row_no_seen":
        active &= ~np.any(seen, axis=1)[:, None]
    elif gate == "pair_new":
        active &= ~seen
    elif gate != "all":
        raise ValueError(f"unknown gate: {gate}")
    return baseline + np.float32(weight) * np.where(active, feature, 0.0)


def load_ensemble(path: Path) -> tuple[dict, list[Path], list[str], np.ndarray]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("kind") != "b_rank_a_port_ensemble_v1" or report.get("decision") != "PASS":
        raise ValueError("D3 ensemble report is not a passing fitted ensemble")
    active = list(report["weights"])
    weights = np.asarray([report["weights"][name] for name in active], dtype=np.float64)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("D3 ensemble weights do not sum to one")
    labels = {name.split(":", 1)[0] for name in active if ":" in name}
    model_dirs = [
        Path(member["path"])
        for member in report["members"]
        if Path(member["path"]).name in labels
    ]
    return report, model_dirs, active, weights


def research(args: argparse.Namespace) -> dict:
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("official data hash differs")
    report, model_dirs, active, weights = load_ensemble(args.ensemble_report)
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
    ) = core.scene_data(args.data, "dataset3")
    scored = {}
    for offset, name in enumerate(("validation", "confirmation"), start=1):
        seed = int(args.seed) + offset
        src, time, candidates, labels = core.sample_segment(
            segments, pool, name, int(args.groups), seed
        )
        names, components, _ = core.component_scores(
            scene="dataset3",
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
            batch=int(args.batch),
        )
        indices = [names.index(component) for component in active]
        baseline = core.mixed_score(weights, components[indices])
        pool_src, pool_time, pool_candidates, _ = core.sample_segment(
            segments, pool, name, len(segments[name]), seed + 10_000
        )
        support = cross_source_support(
            pool_src, pool_time, pool_candidates, src, time, candidates
        )
        feature = qnorm(np.log1p(support).astype(np.float32))
        history = core.segment_history(initial_history, segments, name)
        seen = pair_seen(history, src, candidates)
        positive = support[np.arange(len(labels)), labels]
        scored[name] = {
            "baseline": baseline,
            "feature": feature,
            "seen": seen,
            "labels": labels,
            "audit": {
                "rows": int(len(labels)),
                "candidate_cell_rate": float(np.mean(support > 0)),
                "label_rate": float(np.mean(positive > 0)),
                "label_enrichment": float(
                    np.mean(positive > 0) / max(np.mean(support > 0), np.finfo(float).eps)
                ),
                "rows_without_seen_pair": float(np.mean(~np.any(seen, axis=1))),
            },
        }

    selection = []
    validation = scored["validation"]
    for gate in ("row_no_seen", "pair_new", "all"):
        for weight in WEIGHTS:
            candidate = apply_policy(
                validation["baseline"], validation["feature"], validation["seen"],
                float(weight), gate,
            )
            selection.append({"gate": gate, "weight": float(weight), **metrics(
                validation["baseline"], candidate, validation["labels"]
            )})
    policy = max(
        selection,
        key=lambda item: (item["candidate_mrr"], -item["top1_changed"], -item["weight"]),
    )
    evaluations = {}
    for name, values in scored.items():
        candidate = apply_policy(
            values["baseline"], values["feature"], values["seen"],
            policy["weight"], policy["gate"],
        )
        evaluations[name] = metrics(values["baseline"], candidate, values["labels"])
    checks = {
        "active_weight": policy["weight"] > 0.0,
        "validation_positive": evaluations["validation"]["delta"] > 0.0,
        "confirmation_positive": evaluations["confirmation"]["delta"] > 0.0,
        "confirmation_positive_after_one_se": (
            evaluations["confirmation"]["delta"]
            > evaluations["confirmation"]["delta_se"]
        ),
        "confirmation_negative_rows_below_0_005": (
            evaluations["confirmation"]["negative_row_rate"] < 0.005
        ),
    }
    output = {
        "kind": "d3_same_time_cross_source_residual_v26",
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "data_sha256": DATA_SHA256,
        "ensemble_report": {
            "path": str(args.ensemble_report.resolve()),
            "sha256": sha256(args.ensemble_report),
        },
        "policy": {key: policy[key] for key in ("gate", "weight")},
        "checks": checks,
        "metrics": evaluations,
        "feature_audits": {name: values["audit"] for name, values in scored.items()},
        "selection": selection,
        "source_sha256": sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def read_base(archive: zipfile.ZipFile) -> np.ndarray:
    with archive.open("dataset3.csv") as handle:
        values = np.loadtxt(handle, delimiter=",", dtype=np.float32)
    if values.shape != (ROWS["dataset3"], WIDTH):
        raise ValueError("v21 Dataset3 shape differs")
    return values


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    np.exp(values, out=values)
    values /= values.sum(axis=1, keepdims=True)
    return values


def build(args: argparse.Namespace) -> dict:
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("official data hash differs")
    base_hash = sha256(args.base)
    base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    if (
        base_manifest.get("submission_sha256") != base_hash
        or base_manifest.get("data_sha256") != DATA_SHA256
    ):
        raise ValueError("base manifest does not match the supplied base ZIP")
    if args.require_reference_base and base_hash != V21_ZIP_SHA256:
        raise ValueError("reference v21 base hash differs")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("decision") != "PASS" or report.get("kind") != "d3_same_time_cross_source_residual_v26":
        raise ValueError("D3 residual report is not deployable")
    if args.output.exists():
        raise FileExistsError(args.output)
    run.DATA = str(args.data)
    train, test = run.read_scene("dataset3")
    src = test.src.to_numpy(np.int64, copy=False)
    time = test.time.to_numpy(np.int64, copy=False)
    candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    support = cross_source_support(src, time, candidates, src, time, candidates)
    feature = qnorm(np.log1p(support).astype(np.float32))
    history = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    seen = pair_seen(history, src, candidates)

    with zipfile.ZipFile(args.base) as source:
        if tuple(source.namelist()) != ("dataset3.csv", "dataset4.csv"):
            raise ValueError("base submission members differ")
        d4_hash = member_sha256(source, "dataset4.csv")
        base = read_base(source)
        logits = np.log(np.clip(base, np.float32(1e-12), None))
        candidate = apply_policy(
            logits, feature, seen, float(report["policy"]["weight"]),
            str(report["policy"]["gate"]),
        )
        probability = softmax(candidate)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
            ) as destination:
                with destination.open("dataset3.csv", "w", force_zip64=True) as raw:
                    with io.TextIOWrapper(raw, encoding="ascii", newline="\n") as text:
                        np.savetxt(text, probability, fmt="%.8f", delimiter=",")
                with source.open("dataset4.csv") as incoming:
                    with destination.open("dataset4.csv", "w", force_zip64=True) as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=8 << 20)
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)

    changed = np.argmax(base, axis=1) != np.argmax(probability, axis=1)
    output = {
        "kind": "b_rank_d34_d3_cross_source_v26_submission",
        "submission_sha256": sha256(args.output),
        "data_sha256": DATA_SHA256,
        "base_submission_sha256": base_hash,
        "base_manifest_sha256": sha256(args.base_manifest),
        "residual_report_sha256": sha256(args.report),
        "policy": report["policy"],
        "dataset3": {
            "rows": ROWS["dataset3"],
            "signal_cell_rate": float(np.mean(support > 0)),
            "signal_row_rate": float(np.mean(np.any(support > 0, axis=1))),
            "top1_changed_rate": float(np.mean(changed)),
            "top1_changed_rows": int(changed.sum()),
        },
        "dataset4": {"rows": ROWS["dataset4"], "csv_sha256": d4_hash},
        "source_sha256": sha256(Path(__file__).resolve()),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    research_parser = commands.add_parser("research")
    research_parser.add_argument("--data", type=Path, required=True)
    research_parser.add_argument("--ensemble-report", type=Path, required=True)
    research_parser.add_argument("--output", type=Path, required=True)
    research_parser.add_argument("--groups", type=int, default=30_000)
    research_parser.add_argument("--seed", type=int, default=20260810)
    research_parser.add_argument("--batch", type=int, default=256)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--data", type=Path, required=True)
    build_parser.add_argument("--base", type=Path, required=True)
    build_parser.add_argument("--base-manifest", type=Path, required=True)
    build_parser.add_argument("--report", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument(
        "--require-reference-base", action="store_true",
        help="require the frozen online v21 ZIP; omit for a fresh data-to-submission run",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    result = research(args) if args.command == "research" else build(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("decision", "PASS") == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
