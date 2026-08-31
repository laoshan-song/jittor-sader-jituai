#!/usr/bin/env python3
"""Minimal Jittor port of the dataset2 MultDAE experiment."""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Preserve the caller's user-site path for Jittor compiler subprocesses, but
# keep generated kernels outside $HOME.
site_paths = [path for path in sys.path if "site-packages" in path]
if os.environ.get("PYTHONPATH"):
    site_paths.append(os.environ["PYTHONPATH"])
os.environ["PYTHONPATH"] = os.pathsep.join(site_paths)
os.environ["HOME"] = os.environ.get("ML_CACHE_ROOT", "/tmp")
if "--cuda" not in sys.argv and os.environ.get("JT_USE_CUDA") != "1":
    os.environ["nvcc_path"] = ""
os.environ.setdefault("log_silent", "1")
os.environ.setdefault("use_mpi", "0")

import jittor as jt
from jittor import nn


HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TRACK1_DATA", HERE.parent / "data_A.zip"))
DEFAULT_MODEL = HERE / "model_prod_full_jittor.npz"
DAY = 86400


class MultDAE(nn.Module):
    def __init__(self, items, hidden=256, dropout=0.5):
        self.encoder = nn.Linear(items, hidden)
        self.decoder = nn.Linear(hidden, items)
        self.dropout = nn.Dropout(dropout)
        self.encoder.weight.xavier_gauss_()
        self.decoder.weight.xavier_gauss_()
        self.encoder.bias.assign(jt.zeros(self.encoder.bias.shape))
        self.decoder.bias.assign(jt.zeros(self.decoder.bias.shape))

    def encode(self, x, corrupt=True):
        x = jt.normalize(x, p=2, dim=1, eps=1e-12)
        if corrupt:
            x = self.dropout(x)
        return jt.tanh(self.encoder(x))

    def execute(self, x):
        return self.decoder(self.encode(x, corrupt=True))


def multinomial_loss(logits, target):
    return -(nn.log_softmax(logits, dim=1) * target).sum(dim=1).mean()


def mapped(values, sorted_ids):
    values = np.asarray(values)
    idx = np.searchsorted(sorted_ids, values)
    ok = idx < len(sorted_ids)
    ok[ok] &= sorted_ids[idx[ok]] == values[ok]
    return idx, ok


class SparseRows:
    """NumPy CSR-style rows without a third-party sparse runtime dependency."""

    def __init__(self, indptr, indices, values, shape):
        self.indptr = indptr
        self.indices = indices
        self.values = values
        self.shape = shape

    def dense(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        output = np.zeros((len(rows), self.shape[1]), dtype=np.float32)
        for output_row, source_row in enumerate(rows):
            start, end = self.indptr[source_row : source_row + 2]
            # COO-to-CSR sums duplicate interactions. Retaining duplicates and
            # accumulating here produces the same dense minibatch values.
            np.add.at(output[output_row], self.indices[start:end], self.values[start:end])
        return output


def load_history(history_end):
    with zipfile.ZipFile(DATA) as z:
        train = pd.read_csv(z.open("dataset2/train.csv"), usecols=["src", "dst", "time"])
    edges = train[["src", "dst", "time"]].to_numpy(np.int64, copy=False)
    return edges[edges[:, 2] < history_end]


def interaction_matrix(edges, history_end, decay_days):
    users = np.unique(edges[:, 0])
    items = np.unique(edges[:, 1])
    ui, uok = mapped(edges[:, 0], users)
    ii, iok = mapped(edges[:, 1], items)
    assert uok.all() and iok.all()
    weight = np.ones(len(edges), np.float32)
    if decay_days > 0:
        weight = np.exp(-(edges[:, 2].max() - edges[:, 2]) / (decay_days * DAY)).astype(np.float32)
    order = np.argsort(ui, kind="stable")
    ui = ui[order]
    ii = ii[order]
    weight = weight[order]
    indptr = np.zeros(len(users) + 1, dtype=np.int64)
    np.add.at(indptr, ui + 1, 1)
    np.cumsum(indptr, out=indptr)
    matrix = SparseRows(indptr, ii, weight, (len(users), len(items)))
    return users, items, matrix


def train(args):
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    if args.cuda and jt.has_cuda:
        jt.flags.use_cuda = 1
    edges = load_history(args.history_end)
    users, items, matrix = interaction_matrix(edges, args.history_end, args.decay_days)
    model = MultDAE(len(items), args.hidden, args.dropout)
    optimizer = jt.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        order = rng.permutation(len(users))
        for lo in range(0, len(order), args.batch):
            ids = order[lo : lo + args.batch]
            x_np = matrix.dense(ids)
            x = jt.array(x_np)
            loss = multinomial_loss(model(x), x)
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.clip_grad_norm(5.0, 2)
            optimizer.step()
            total += float(np.asarray(loss.data).item()) * len(ids)
        print(f"epoch={epoch + 1} loss={total / len(users):.6f}", flush=True)

    model.eval()
    hidden = []
    with jt.no_grad():
        for lo in range(0, len(users), args.batch):
            x = jt.array(matrix.dense(np.arange(lo, min(lo + args.batch, len(users)))))
            hidden.append(np.asarray(model.encode(x, corrupt=False).data, np.float32))
    output = Path(args.output)
    np.savez_compressed(
        output,
        kind="dae",
        users=users,
        items=items,
        user=np.vstack(hidden),
        item=np.asarray(model.decoder.weight.data, np.float32),
        ibias=np.asarray(model.decoder.bias.data, np.float32),
        encoder_weight=np.asarray(model.encoder.weight.data, np.float32),
        encoder_bias=np.asarray(model.encoder.bias.data, np.float32),
        history_end=np.int64(args.history_end),
        decay_days=np.float64(args.decay_days),
    )
    print(f"wrote {output}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("train")
    a.add_argument("--history-end", type=int, default=1262044800)
    a.add_argument("--hidden", type=int, default=256)
    a.add_argument("--dropout", type=float, default=0.5)
    a.add_argument("--epochs", type=int, default=10)
    a.add_argument("--decay-days", type=float, default=365.0)
    a.add_argument("--batch", type=int, default=128)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--weight-decay", type=float, default=1e-5)
    a.add_argument("--seed", type=int, default=20260711)
    a.add_argument("--cuda", action="store_true")
    a.add_argument("--output", default=HERE / "model_dae_jittor.npz")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
