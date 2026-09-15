#!/usr/bin/env python3
"""Build replay-aligned D4 baseline scores from freshly trained Jittor models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from b_rank import (
    pairnew_transformer_jittor as pairnew,
    replay_score_cache,
    temporal_attention_jittor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-cache", type=Path, action="append", required=True)
    parser.add_argument("--control-fit", type=Path, required=True)
    parser.add_argument("--pairnew-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing output reuse: {args.output}")

    temporal_attention_jittor.configure_cuda()
    scored, names, _ = replay_score_cache.load(args.replay_cache)
    report = json.loads(args.pairnew_report.read_text(encoding="utf-8"))
    if (
        report.get("kind") not in {
            "d4_pairnew_rank_slot_candidate_set_transformer_v12",
            "d4_pairnew_rank_slot_scaled_replay_transformer_v21",
        }
        or report.get("decision") not in {"PASS", "NO_GO"}
        or not report.get("training", {}).get("members")
    ):
        raise ValueError("pair-new report is not a deployable Jittor ensemble")
    control, indices, base_index, seen_alpha, new_alpha = pairnew._control_contract(
        args.control_fit.resolve(), names
    )
    nets = []
    for record in report["training"]["members"]:
        net, metadata = pairnew.load_checkpoint(Path(record["checkpoint"]).resolve())
        if int(metadata["hidden"]) != int(record["hidden"]):
            raise ValueError("pair-new checkpoint metadata differs")
        nets.append(net)

    static_mean = np.asarray(report["training"]["static_mean"], dtype=np.float32)
    static_std = np.asarray(report["training"]["static_std"], dtype=np.float32)
    args.output.mkdir(parents=True)
    records = {}
    for (strategy, split), (scores, _labels, seen, _segments, static) in scored.items():
        base = pairnew._control_score(
            scores, seen, control, indices, base_index, seen_alpha, new_alpha
        )
        feature = pairnew._features(
            scores, static, base, seen, static_mean, static_std
        )
        residual = pairnew._qnorm(
            np.mean(
                [pairnew._predict(net, feature, args.batch) for net in nets], axis=0
            )
        )
        baseline = pairnew._candidate_score(
            base, residual, seen, float(report["residual_alpha"])
        )
        name = f"{strategy}__{split}.npy"
        np.save(args.output / name, baseline, allow_pickle=False)
        records[f"{strategy}__{split}"] = {
            "file": name,
            "shape": list(baseline.shape),
            "dtype": str(baseline.dtype),
        }
        print(json.dumps({"saved": name, "rows": len(baseline)}), flush=True)

    manifest = {
        "kind": "ruc3_d4_replay_baseline_cache_v1",
        "decision": "PASS",
        "members": len(nets),
        "entries": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
