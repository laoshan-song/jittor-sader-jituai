#!/usr/bin/env python3
"""Build the frozen three-member D3 residual ensemble on verified online c6."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

os.environ.update({
    "use_cutt": "0",
    "use_cutlass": "0",
    "use_nccl": "0",
    "use_mkl": "0",
})
os.environ.setdefault("JT_USE_CUDA", "1")

from d3_residual_ranker_v39 import (
EXPECTED_DATA_SHA256,
    add_structural,
    component_features,
    qnorm,
    rank_percentile,
    sha256,
)


ROWS = {"dataset3": 157_670, "dataset4": 2_322_538}
WIDTH = 100
FIXED_SCALE = np.float32(0.30)
EXPECTED_BASE_SHA256 = "627c50a7d0d941ed25af0f5928d2f72a100b087dac8e34b09472895f77fb4a94"


def member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    np.exp(values, out=values)
    values /= values.sum(axis=1, keepdims=True)
    return values


def quantiles(values: np.ndarray) -> dict[str, float]:
    result = np.quantile(np.asarray(values, dtype=np.float64), (0.01, 0.10, 0.50, 0.90, 0.99))
    return {name: float(value) for name, value in zip(("q01", "q10", "q50", "q90", "q99"), result)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--ensemble-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--transformer-model", type=Path, nargs=3, required=True)
    parser.add_argument("--transformer-report", type=Path, nargs=3, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument(
        "--allow-reproduced-base",
        action="store_true",
        help="accept a freshly trained c6 with identical policy but different bytes",
    )
    args = parser.parse_args()
    if args.output.exists() or args.report_output.exists():
        raise FileExistsError("refusing to overwrite v65 output")
    if sha256(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("official data hash differs")
    if not args.allow_reproduced_base and sha256(args.base) != EXPECTED_BASE_SHA256:
        raise ValueError("base is not verified online c6")
    validation_report = json.loads(args.validation_report.read_text())
    if (
        validation_report.get("decision") != "PASS"
        or validation_report.get("aggregation") != "mean_member"
        or not all(validation_report.get("checks", {}).values())
        or any(
            validation_report["evaluations"][split]["direct_vs_c6"]["delta"] < 0.008
            for split in ("validation", "confirmation")
        )
    ):
        raise ValueError("time-forward validation evidence is not deployable")

    sys.path.insert(0, str(args.code.resolve()))
    sys.path.insert(0, str(args.code.parent.resolve()))
    import d3_cross_source_c2_audit as d3
    import d3_multiscale_craft_gate as multiscale
    import d3_near_time_audit as near
    import jittor as jt
    from jittor import nn
    from b_rank_a_port import ensemble_core as core
    import run

    payloads = [jt.load(str(path)) for path in args.transformer_model]
    reports = [json.loads(path.read_text()) for path in args.transformer_report]
    for payload, report in zip(payloads, reports):
        if payload.get("kind") != "d3_c6_candidate_set_transformer_v49_pilot":
            raise ValueError("unexpected transformer kind")
        if int(payload["epoch"]) != 8 or int(report["selected"]["epoch"]) != 8:
            raise ValueError("final member is not frozen at epoch 8")
        if not np.isclose(float(payload["scale"]), float(FIXED_SCALE), rtol=0.0, atol=1e-7):
            raise ValueError("final member scale differs")
        if report.get("fit_splits") != ["meta_train", "validation", "confirmation"]:
            raise ValueError("final member was not fit on all replay history")

    class TransformerBlock(nn.Module):
        def __init__(self, hidden: int, heads: int) -> None:
            self.attention = jt.attention.MultiheadAttention(hidden, heads, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden)
            self.feedforward = nn.Sequential(
                nn.Linear(hidden, 2 * hidden), nn.Relu(), nn.Linear(2 * hidden, hidden)
            )
            self.norm2 = nn.LayerNorm(hidden)

        def execute(self, values):
            context, _ = self.attention(values, values, values, need_weights=False)
            values = self.norm1(values + context)
            return self.norm2(values + self.feedforward(values))

    class SetTransformerResidual(nn.Module):
        def __init__(self, payload) -> None:
            hidden = int(payload["hidden"])
            self.encoder = nn.Sequential(
                nn.Linear(int(payload["input_dim"]), hidden), nn.Relu(),
                nn.Linear(hidden, hidden), nn.Relu(),
            )
            self.blocks = nn.ModuleList([
                TransformerBlock(hidden, int(payload["heads"]))
                for _ in range(int(payload["layers"]))
            ])
            self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.Relu(), nn.Linear(hidden, 1))

        def execute(self, values):
            values = self.encoder(values)
            for block in self.blocks:
                values = block(values)
            return self.output(values).squeeze(-1)

    nets = []
    for payload in payloads:
        net = SetTransformerResidual(payload)
        net.load_state_dict({name: jt.array(value) for name, value in payload["state"].items()})
        net.eval()
        nets.append(net)

    _, model_dirs, active, ensemble_weights = d3.load_ensemble(args.ensemble_report)
    run.DATA = str(args.data.resolve())
    train, test = run.read_scene("dataset3")
    src = test.src.to_numpy(np.int64, copy=False)
    query_time = test.time.to_numpy(np.int64, copy=False)
    candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    if candidates.shape != (ROWS["dataset3"], WIDTH):
        raise ValueError("official D3 test shape differs")
    history = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    scene = core.scene_data(args.data, "dataset3")
    _, _, max_node, use_src_freq, _pool, freq, src_freq, _history0, _segments = scene
    labels_placeholder = np.zeros(len(src), dtype=np.int64)
    component_names, components, _ = core.component_scores(
        scene="dataset3", model_dirs=model_dirs, history=history,
        max_node=max_node, use_src_freq=use_src_freq, freq=freq, src_freq=src_freq,
        src=src, time=query_time, candidates=candidates, labels=labels_placeholder,
        batch=args.batch,
    )
    if component_names != list(payloads[0]["component_names"]):
        raise ValueError("official component inventory differs")
    feature_names, feature_values = component_features(components)
    del components

    with zipfile.ZipFile(args.base) as base_archive:
        if tuple(base_archive.namelist()) != ("dataset3.csv", "dataset4.csv"):
            raise ValueError("base members differ")
        with base_archive.open("dataset3.csv") as handle:
            base_probability = np.loadtxt(handle, delimiter=",", dtype=np.float32)
        base_d4_hash = member_sha256(base_archive, "dataset4.csv")
    if base_probability.shape != (ROWS["dataset3"], WIDTH):
        raise ValueError("base D3 shape differs")
    base_logits = np.log(np.clip(base_probability, np.float32(1e-12), None)).astype(np.float32)
    base_normalized = qnorm(base_logits)
    feature_names.extend(("c6_qnorm", "c6_rank", "c6_winner_gap", "pair_seen"))

    index = near.NearTimeIndex(src, query_time, candidates)
    cross_exact = index.support(src, query_time, candidates, 0)
    source_exact = index.source_support(src, query_time, candidates, 0)
    source_300_past, source_300_future = multiscale.directional_support(
        index, src, query_time, candidates, 300, True
    )
    source_session = source_300_past + source_300_future
    if not np.array_equal(
        source_session,
        index.source_support(src, query_time, candidates, 300) - source_exact,
    ):
        raise ValueError("official source-session implementation differs")
    seen = d3.pair_seen(history, src, candidates)
    feature_values.extend((
        base_normalized,
        rank_percentile(base_logits),
        base_normalized - base_normalized.max(axis=1, keepdims=True),
        seen.astype(np.float32),
    ))
    add_structural(feature_names, feature_values, "source_0_300", source_session)
    source_900_past, source_900_future = multiscale.directional_support(
        index, src, query_time, candidates, 900, True
    )
    add_structural(
        feature_names, feature_values, "source_300_900",
        source_900_past + source_900_future - source_300_past - source_300_future,
    )
    source_3600_past, source_3600_future = multiscale.directional_support(
        index, src, query_time, candidates, 3_600, True
    )
    add_structural(
        feature_names, feature_values, "source_900_3600",
        source_3600_past + source_3600_future - source_900_past - source_900_future,
    )
    source_21600_past, source_21600_future = multiscale.directional_support(
        index, src, query_time, candidates, 21_600, True
    )
    source_86400_past, source_86400_future = multiscale.directional_support(
        index, src, query_time, candidates, 86_400, True
    )
    add_structural(
        feature_names, feature_values, "source_3600_21600",
        source_21600_past + source_21600_future - source_3600_past - source_3600_future,
    )
    add_structural(
        feature_names, feature_values, "source_21600_86400",
        source_86400_past + source_86400_future - source_21600_past - source_21600_future,
    )
    add_structural(feature_names, feature_values, "cross_exact", cross_exact)
    cross_300_past, cross_300_future = multiscale.directional_support(
        index, src, query_time, candidates, 300, False
    )
    cross_900_past, cross_900_future = multiscale.directional_support(
        index, src, query_time, candidates, 900, False
    )
    add_structural(
        feature_names, feature_values, "cross_300_900",
        cross_900_past + cross_900_future - cross_300_past - cross_300_future,
    )
    cross_3600_past, cross_3600_future = multiscale.directional_support(
        index, src, query_time, candidates, 3_600, False
    )
    add_structural(
        feature_names, feature_values, "cross_900_3600",
        cross_3600_past + cross_3600_future - cross_900_past - cross_900_future,
    )
    if feature_names != list(payloads[0]["feature_names"]) or len(feature_names) != 87:
        raise ValueError("official feature inventory differs")

    duplicate_rows = np.any(
        np.diff(np.sort(candidates, axis=1, kind="stable"), axis=1) == 0,
        axis=1,
    )
    member_corrections = np.empty((len(nets), len(src), WIDTH), dtype=np.float32)
    with jt.no_grad():
        for start in range(0, len(src), args.batch):
            stop = min(start + args.batch, len(src))
            part = np.stack([value[start:stop] for value in feature_values], axis=2).astype(
                np.float32, copy=False
            )
            if not np.isfinite(part).all():
                raise ValueError("official feature batch is not finite")
            tensor = jt.array(part)
            for member_index, net in enumerate(nets):
                member_corrections[member_index, start:stop] = qnorm(
                    np.asarray(net(tensor).data, dtype=np.float32)
                )
            if start % (args.batch * 100) == 0:
                print(json.dumps({"inference_rows": int(stop)}), flush=True)
    correction = member_corrections.mean(axis=0)
    correction[duplicate_rows] = 0.0
    candidate_logits = base_logits + FIXED_SCALE * correction
    probability = softmax(candidate_logits)
    probability[duplicate_rows] = base_probability[duplicate_rows]
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("candidate probabilities are invalid")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(args.base) as source, zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
        ) as destination:
            with destination.open("dataset3.csv", "w", force_zip64=True) as raw:
                with io.TextIOWrapper(raw, encoding="ascii", newline="\n") as text:
                    np.savetxt(text, probability, fmt="%.8f", delimiter=",")
            with source.open("dataset4.csv") as incoming, destination.open(
                "dataset4.csv", "w", force_zip64=True
            ) as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=8 << 20)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    with zipfile.ZipFile(args.output) as archive:
        output_d3_hash = member_sha256(archive, "dataset3.csv")
        output_d4_hash = member_sha256(archive, "dataset4.csv")
    if output_d4_hash != base_d4_hash:
        raise ValueError("Dataset4 changed from verified c6")
    changed = np.argmax(base_probability, axis=1) != np.argmax(probability, axis=1)
    flat_members = member_corrections.reshape(len(nets), -1)
    row_sum_max_error = float(np.max(np.abs(probability.sum(axis=1) - 1.0)))
    correction_std = float(correction.std())
    top1_changed_rate = float(changed.mean())
    checks = {
        "row_sum_max_error_below_1e_6": row_sum_max_error < 1e-6,
        "dataset4_identical_to_verified_c6": output_d4_hash == base_d4_hash,
        "correction_std_between_0_20_and_0_80": 0.20 <= correction_std <= 0.80,
        "top1_changed_rate_between_0_02_and_0_10": 0.02 <= top1_changed_rate <= 0.10,
        "duplicate_rows_fall_back_exactly": bool(
            np.array_equal(probability[duplicate_rows], base_probability[duplicate_rows])
        ),
    }
    report = {
        "kind": "b_rank_d34_c6_d3_set_transformer_mean_ensemble_v65",
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "data_sha256": EXPECTED_DATA_SHA256,
        "submission_sha256": sha256(args.output),
        "base_sha256": sha256(args.base),
        "ensemble_report_sha256": sha256(args.ensemble_report),
        "validation_report_sha256": sha256(args.validation_report),
        "time_forward_deltas": {
            split: validation_report["evaluations"][split]["direct_vs_c6"]["delta"]
            for split in ("validation", "confirmation")
        },
        "transformer_model_sha256": [sha256(path) for path in args.transformer_model],
        "transformer_report_sha256": [sha256(path) for path in args.transformer_report],
        "aggregation": "mean_member_without_final_qnorm",
        "fixed_scale": float(FIXED_SCALE),
        "feature_count": len(feature_names),
        "dataset3": {
            "rows": ROWS["dataset3"],
            "csv_sha256": output_d3_hash,
            "duplicate_fallback_rows": int(duplicate_rows.sum()),
            "pair_seen_cell_rate": float(seen.mean()),
            "top1_changed_rows": int(changed.sum()),
            "top1_changed_rate": top1_changed_rate,
            "row_sum_max_error": row_sum_max_error,
            "correction_mean": float(correction.mean()),
            "correction_std": correction_std,
            "correction_quantiles": quantiles(correction),
            "member_correction_correlations": np.corrcoef(flat_members).tolist(),
            "base_margin_quantiles": quantiles(
                np.sort(base_logits, axis=1)[:, -1] - np.sort(base_logits, axis=1)[:, -2]
            ),
            "candidate_margin_quantiles": quantiles(
                np.sort(candidate_logits, axis=1)[:, -1] - np.sort(candidate_logits, axis=1)[:, -2]
            ),
        },
        "dataset4": {
            "rows": ROWS["dataset4"],
            "csv_sha256": output_d4_hash,
            "identical_to_base": True,
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
