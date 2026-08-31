#!/usr/bin/env python3
"""Train the audited multi-slice DeepSets-v2 ensemble and score test."""

import argparse
import json
import zipfile
from pathlib import Path

import jittor as jt
import numpy as np
import pandas as pd

import d2_pool_ranker_eval_jittor as rank
import d2_temporal_pool_eval_jittor as pool
import d2_v2_set_incremental_audit_jittor as set_v2
import d2_v2_set_production_jittor as production


MEMBERS = (
    (64, 20265701, 8), (64, 20265702, 6), (64, 20265703, 8),
    (96, 20265801, 4), (96, 20265802, 8), (96, 20265803, 8),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--slice-models", type=Path)
    parser.add_argument("--prod-models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=6)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--candidate-seed", type=int, default=20260781)
    parser.add_argument("--cpu-inference", action="store_true")
    args = parser.parse_args()
    if not args.checkpoints and args.slice_models is None:
        parser.error("--slice-models is required when training checkpoints")

    jt.flags.use_cuda = 1
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.data) as archive:
        train = pd.read_csv(
            archive.open("dataset2/train.csv"), usecols=["src", "dst", "time"]
        )
        test = pd.read_csv(archive.open("dataset2/test.csv"))
    edges = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    test_candidates = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    candidate_audit = {}
    if args.checkpoints:
        checkpoint_paths = list(args.checkpoints)
    else:
        replay_pool = pool.build_replay_pool(test_candidates, edges[:, 1])
        features, labels = [], []
        for offset, name in enumerate(("y2009", "strict")):
            values = rank.build_slice(
                edges, test, args.slice_models, name, pool.SLICES[name], 153420,
                args.batch, args.candidate_seed + offset, replay_pool, "test-pool",
            )
            set_v2.extend(values, edges, int(test_candidates.max()))
            features.append(values["feature_v2"])
            labels.append(values["labels"])
            candidate_audit[name] = values["candidate_audit"]
            print(name, values["feature_v2"].shape, flush=True)
        feature = np.concatenate(features)
        label = np.concatenate(labels)
        del features, labels

        checkpoint_paths = []
        for hidden, seed, epochs in MEMBERS:
            net = production.train_full(
                feature, label, hidden, epochs, args.batch, seed
            )
            path = args.output_dir / f"multi_set{hidden}_seed{seed}_jittor.npz"
            production.save_checkpoint(path, net, hidden, seed, epochs)
            reloaded = production.load_checkpoint(path)
            saved = np.load(path, allow_pickle=False)
            state = reloaded.state_dict()
            for index, name in enumerate(saved["state_names"]):
                if not np.array_equal(
                    saved[f"state_{index}"], np.asarray(state[str(name)].data)
                ):
                    raise RuntimeError(
                        f"checkpoint reload mismatch: {path}:{name}"
                    )
            checkpoint_paths.append(path)
            print(f"verified {path}", flush=True)
        del feature, label

    feature = production.official_features(
        edges, test, args.prod_models, args.batch
    )
    jt.sync_all()
    if args.cpu_inference:
        jt.flags.use_cuda = 0
    nets = [production.load_checkpoint(path) for path in checkpoint_paths]
    predictions = [
        set_v2.predict_set(net, feature, args.batch) for net in nets
    ]
    set64 = pool.qnorm(np.mean(predictions[:3], axis=0))
    set96 = pool.qnorm(np.mean(predictions[3:], axis=0))
    set_all = pool.qnorm(0.5 * (set64 + set96))
    outputs = {
        "official_multislice_set64.npy": set64,
        "official_multislice_set96.npy": set96,
        "official_multislice_set_all.npy": set_all,
    }
    for name, value in outputs.items():
        np.save(args.output_dir / name, np.round(value, 2).astype(np.float32))
    top1 = [np.argmax(value, axis=1) for value in predictions]
    diagnostics = {
        "kind": "jittor_deepsets_v2_multislice",
        "training_rows": 306840,
        "training_slices": ["y2009", "strict"],
        "checkpoint_only": bool(args.checkpoints),
        "candidate_audit": candidate_audit,
        "checkpoints": [str(path) for path in checkpoint_paths],
        "shape": list(set_all.shape),
        "finite": bool(np.isfinite(set_all).all()),
        "member_top1_agreement_with_first": [
            float(np.mean(top1[0] == value)) for value in top1[1:]
        ],
        "set64_set96_top1_agreement": float(
            np.mean(np.argmax(set64, axis=1) == np.argmax(set96, axis=1))
        ),
    }
    (args.output_dir / "production_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
