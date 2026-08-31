#!/usr/bin/env python3
"""Official-data training for the packaged Jittor MF32 model family."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import jittor as jt
import numpy as np
import pandas as pd
from jittor import nn

from model import ImplicitMF, configure_cuda, dense_indices


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--negative-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing output-directory reuse: {args.output_dir}")
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    args.output_dir.mkdir(parents=True)

    with zipfile.ZipFile(args.data) as archive, archive.open("dataset4/train.csv") as handle:
        frame = pd.read_csv(handle, usecols=["src", "dst"], dtype=np.uint32)
    raw_source = frame["src"].to_numpy(dtype=np.uint32, copy=False)
    raw_item = frame["dst"].to_numpy(dtype=np.uint32, copy=False)
    source_ids = np.unique(raw_source)
    item_ids = np.unique(raw_item)
    source = dense_indices(raw_source, source_ids)
    positive = dense_indices(raw_item, item_ids)

    configure_cuda()
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ImplicitMF(len(source_ids) + 1, len(item_ids) + 1, args.embedding_dim)
    model.source.weight.assign(jt.randn(model.source.weight.shape) * 0.01)
    model.item.weight.assign(jt.randn(model.item.weight.shape) * 0.01)
    model.item_bias.weight.assign(jt.zeros(model.item_bias.weight.shape))
    optimizer = jt.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    losses = []
    started = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(source))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            negative = rng.integers(
                1,
                len(item_ids) + 1,
                size=(len(rows), args.negative_count),
                dtype=np.int32,
            )
            collision = negative == positive[rows, None]
            while collision.any():
                negative[collision] = rng.integers(
                    1, len(item_ids) + 1, size=int(collision.sum()), dtype=np.int32
                )
                collision = negative == positive[rows, None]
            candidates = np.concatenate([positive[rows, None], negative], axis=1)
            logits = model(jt.array(source[rows]), jt.array(candidates))
            loss = nn.cross_entropy_loss(logits, jt.zeros(len(rows), dtype="int32"))
            optimizer.step(loss)
            total += float(np.asarray(loss.data).item()) * len(rows)
        losses.append(total / len(source))
        print(json.dumps({"epoch": epoch + 1, "loss": losses[-1]}), flush=True)

    checkpoint = args.output_dir / "d4_implicit_mf32.npz"
    payload = {
        "kind": np.asarray("d4_implicit_mf"),
        "source_count": np.asarray(model.source_count),
        "item_count": np.asarray(model.item_count),
        "embedding_dim": np.asarray(model.embedding_dim),
        "source_ids": source_ids.astype(np.uint32),
        "item_ids": item_ids.astype(np.uint32),
    }
    payload.update(
        {
            f"param__{name}": np.asarray(value.data, dtype=np.float32).copy()
            for name, value in model.state_dict().items()
        }
    )
    np.savez_compressed(checkpoint, **payload)
    receipt = {
        "kind": "track1_b_mf32_training_v1",
        "decision": "PASS",
        "data_sha256": DATA_SHA256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_count": len(source_ids),
        "item_count": len(item_ids),
        "embedding_dim": args.embedding_dim,
        "loss": losses,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "TRAINING_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
