#!/usr/bin/env python3
"""Stream a fitted D3/D4 multi-model ensemble into a canonical submission ZIP."""

import argparse
import hashlib
import io
import json
import os
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ["JT_USE_CUDA"] = "1"

import ensemble_core as core
import run


EXPECTED_DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probabilities(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    np.exp(values, out=values)
    values /= values.sum(axis=1, keepdims=True)
    if not np.isfinite(values).all():
        raise ValueError("non-finite probability")
    return values


class PreparedScene:
    def __init__(self, data: Path, scene: str, report_path: Path):
        self.scene = scene
        self.report_path = report_path.resolve()
        self.report = json.loads(self.report_path.read_text(encoding="utf-8"))
        if self.report.get("decision") != "PASS" or self.report.get("scene") != scene:
            raise ValueError(f"invalid fitted ensemble report: {report_path}")
        self.component_names = list(self.report["component_names"])
        self.active_names = [
            name
            for name in self.component_names
            if float(self.report["weights"].get(name, 0.0)) > 1e-12
        ]
        self.weights = np.asarray(
            [float(self.report["weights"][name]) for name in self.active_names],
            dtype=np.float64,
        )
        if not self.active_names or not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(f"ensemble weights do not sum to one: {scene}")

        (
            train,
            self.test,
            self.max_node,
            self.use_src_freq,
            _pool,
            self.freq,
            self.src_freq,
            _initial_history,
            _segments,
        ) = core.scene_data(data, scene)
        self.history = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
        self.stats = run.Stats(self.history, self.max_node)
        self.members = []
        for member in self.report["members"]:
            model_dir = Path(member["path"])
            suffix = scene[-1]
            report_path = model_dir / "member_report.json"
            if sha256(report_path) != member["report_sha256"]:
                raise ValueError(f"member report hash differs: {model_dir}")
            member_report = json.loads(report_path.read_text(encoding="utf-8"))
            model_path = model_dir / f"m{suffix}.pkl"
            ranker_path = model_dir / f"r{suffix}.pkl"
            if sha256(model_path) != member_report.get("model_sha256"):
                raise ValueError(f"base model hash differs: {model_dir}")
            if sha256(ranker_path) != member_report.get("ranker_sha256"):
                raise ValueError(f"ranker hash differs: {model_dir}")
            base_name = f"{model_dir.name}:base"
            rank_name = f"{model_dir.name}:rank"
            prop_name = f"{model_dir.name}:prop"
            use_base = base_name in self.active_names
            use_rank = rank_name in self.active_names
            use_prop = prop_name in self.active_names
            if not (use_base or use_rank or use_prop):
                continue
            base, mu, sd = run.load_model(str(model_path))
            ranker = run.load_fast_model(str(ranker_path))
            feature_function, member_src_freq = run.feat_fn_for_dim(len(mu))
            state = base.state_dict()
            source_embedding = np.asarray(state["src_emb.weight"].data, dtype=np.float32)
            destination_embedding = np.asarray(state["dst_emb.weight"].data, dtype=np.float32)
            context = run.make_fast_context(
                self.history,
                self.max_node,
                source_embedding,
                destination_embedding,
                self.src_freq,
                self.stats,
            )
            self.members.append(
                {
                    "name": model_dir.name,
                    "base_name": base_name,
                    "rank_name": rank_name,
                    "prop_name": prop_name,
                    "use_base": use_base,
                    "use_rank": use_rank,
                    "use_prop": use_prop,
                    "base": base,
                    "mu": mu,
                    "sd": sd,
                    "ranker": ranker,
                    "feature_function": feature_function,
                    "member_src_freq": member_src_freq,
                    "context": context,
                }
            )

    def score(self, frame, batch: int) -> np.ndarray:
        src = frame.src.to_numpy(np.int64, copy=False)
        time = frame.time.to_numpy(np.int64, copy=False)
        candidates = frame.iloc[:, 2:].to_numpy(np.int64, copy=False)
        scores = {}
        heuristic_names = {"pair_repeat", "recent_pair", "candidate_pool", "hand_score"}
        if heuristic_names.intersection(self.active_names):
            raw = run.raw_features_numba(
                self.stats,
                src,
                time,
                candidates,
                self.freq,
                self.src_freq,
                self.use_src_freq,
                return_raw=True,
            )
            heuristic_index = 23 if self.use_src_freq else 21
            scores.update(
                {
                    "pair_repeat": core.qnorm(3.0 * raw[:, :, 0] + 2.0 * raw[:, :, 1]),
                    "recent_pair": core.qnorm(raw[:, :, 8] + raw[:, :, 9]),
                    "candidate_pool": core.qnorm(raw[:, :, 11]),
                    "hand_score": core.qnorm(raw[:, :, heuristic_index]),
                }
            )
            del raw
        feature_cache = {}
        history_ids = history_mask = None
        for member in self.members:
            base_score = None
            if member["use_base"]:
                key = (member["feature_function"].__name__, member["member_src_freq"])
                if key not in feature_cache:
                    feature_cache[key] = member["feature_function"](
                        self.stats,
                        src,
                        time,
                        candidates,
                        self.freq,
                        self.src_freq,
                        member["member_src_freq"],
                    )
                feature = run.scale(feature_cache[key], member["mu"], member["sd"])
                if history_ids is None:
                    history_ids, history_mask = run.history_ids(self.stats, src)
                base_score = run.model_scores_with_prop(
                    member["base"],
                    feature,
                    src,
                    candidates,
                    history_ids,
                    history_mask,
                    max(128, batch // 4),
                    None,
                    0.0,
                )
                scores[member["base_name"]] = core.qnorm(base_score)
            if member["use_rank"]:
                fast_feature = run.fast_rank_features(
                    member["context"], self.stats, src, time, candidates, self.freq
                )
                if fast_feature.shape[-1] != member["ranker"].dim:
                    if base_score is None:
                        raise ValueError("stacked ranker requires an active base score")
                    fast_feature = np.concatenate(
                        [fast_feature, run.base_score_extra_features(base_score)], axis=2
                    )
                rank_score = run.fast_scores(member["ranker"], fast_feature, max(128, batch))
                scores[member["rank_name"]] = core.qnorm(rank_score)
            if member["use_prop"]:
                context = member["context"]
                propagation_score = run.prop_scores(
                    src,
                    candidates,
                    context["src_emb"],
                    context["dst_emb"],
                    context["prop_src"],
                    context["prop_dst"],
                )
                scores[member["prop_name"]] = core.qnorm(propagation_score)
        if any(name not in scores for name in self.active_names):
            raise ValueError(f"active component inventory changed during inference: {self.scene}")
        return core.mixed_score(
            self.weights, np.stack([scores[name] for name in self.active_names], axis=0)
        )


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        path.name: sha256(path)
        for path in (
            root / "run.py",
            root / "ensemble_core.py",
            root / "train_member.py",
            root / "train_grid.py",
            root / "fit_ensemble.py",
            root / "infer_ensemble.py",
            root / "verify_submission.py",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset3-report", type=Path, required=True)
    parser.add_argument("--dataset4-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    data = args.data.resolve()
    output = args.output.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if sha256(data) != EXPECTED_DATA_SHA256:
        raise ValueError("official data hash differs")
    for path in (output, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing output reuse: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    rows = {}
    reports = {
        "dataset3": args.dataset3_report,
        "dataset4": args.dataset4_report,
    }
    try:
        with zipfile.ZipFile(
            temporary,
            "x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for scene in ("dataset3", "dataset4"):
                prepared = PreparedScene(data, scene, reports[scene])
                count = 0
                with archive.open(f"{scene}.csv", "w", force_zip64=True) as raw:
                    with io.TextIOWrapper(raw, encoding="ascii", newline="\n") as text:
                        for start in range(0, len(prepared.test), args.chunk_rows):
                            frame = prepared.test.iloc[start : start + args.chunk_rows]
                            np.savetxt(
                                text,
                                probabilities(prepared.score(frame, args.batch)),
                                fmt="%.8f",
                                delimiter=",",
                                newline="\n",
                            )
                            count += len(frame)
                            print(f"{scene} {count}/{len(prepared.test)}", flush=True)
                rows[scene] = count
        manifest = {
            "kind": "b_rank_a_port_submission_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_sha256": EXPECTED_DATA_SHA256,
            "submission_sha256": sha256(temporary),
            "source_hashes": source_hashes(),
            "ensemble_reports": {
                scene: {
                    "path": str(path.resolve()),
                    "sha256": sha256(path.resolve()),
                }
                for scene, path in reports.items()
            },
            "row_counts": rows,
            "format": "dataset3.csv,dataset4.csv; headerless ASCII; %.8f probabilities",
        }
        temporary.replace(output)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
