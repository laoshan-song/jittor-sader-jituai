#!/usr/bin/env python3
"""Train MultVAE or RecVAE on dataset2 using Jittor only."""

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt
from jittor import nn


DAY = 86400
ORIGINS = {
    "y2008": 1199145600,
    "y2009": 1230768000,
    "strict": 1262044800,
    "prod": 1296345600,
}


def mapped(values, sorted_ids):
    values = np.asarray(values)
    index = np.searchsorted(sorted_ids, values)
    valid = index < len(sorted_ids)
    valid[valid] &= sorted_ids[index[valid]] == values[valid]
    return index, valid


class SparseRows:
    def __init__(self, indptr, indices, values, shape):
        self.indptr = indptr
        self.indices = indices
        self.values = values
        self.shape = shape
        self.nnz = len(indices)

    def dense(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        output = np.zeros((len(rows), self.shape[1]), np.float32)
        for output_row, source_row in enumerate(rows):
            start, end = self.indptr[source_row : source_row + 2]
            np.add.at(
                output[output_row],
                self.indices[start:end],
                self.values[start:end],
            )
        return output


def load_matrix(data, origin, decay_days):
    with zipfile.ZipFile(data) as archive:
        train = pd.read_csv(
            archive.open("dataset2/train.csv"), usecols=["src", "dst", "time"]
        )
    edges = train.loc[
        train.time < origin, ["src", "dst", "time"]
    ].to_numpy(np.int64, copy=False)
    users, items = np.unique(edges[:, 0]), np.unique(edges[:, 1])
    user_index, user_valid = mapped(edges[:, 0], users)
    item_index, item_valid = mapped(edges[:, 1], items)
    assert user_valid.all() and item_valid.all()
    weights = np.exp(
        -(edges[:, 2].max() - edges[:, 2]) / (decay_days * DAY)
    ).astype(np.float32)
    order = np.argsort(user_index, kind="stable")
    user_index = user_index[order]
    item_index = item_index[order]
    weights = weights[order]
    indptr = np.zeros(len(users) + 1, np.int64)
    np.add.at(indptr, user_index + 1, 1)
    np.cumsum(indptr, out=indptr)
    matrix = SparseRows(indptr, item_index, weights, (len(users), len(items)))
    return users, items, matrix


def init_linear(layer):
    layer.weight.xavier_gauss_()
    layer.bias.assign(jt.zeros(layer.bias.shape))


class MultVAE(nn.Module):
    def __init__(self, items, hidden=600, latent=200, dropout=0.5):
        self.q = nn.Linear(items, hidden)
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.p = nn.Linear(latent, hidden)
        self.out = nn.Linear(hidden, items)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.q, self.mu, self.logvar, self.p, self.out):
            init_linear(layer)

    def encode(self, x, corrupt=True):
        x = jt.normalize(x, p=2, dim=1, eps=1e-12)
        if corrupt:
            x = self.dropout(x)
        hidden = jt.tanh(self.q(x))
        return self.mu(hidden), jt.clamp(self.logvar(hidden), -10.0, 10.0)

    def decoder_hidden(self, z):
        return jt.tanh(self.p(z))

    def execute(self, x):
        mu, logvar = self.encode(x, True)
        z = mu + jt.randn(mu.shape) * jt.exp(0.5 * logvar)
        return self.out(self.decoder_hidden(z)), mu, logvar


def swish(x):
    return x * jt.sigmoid(x)


class RecEncoder(nn.Module):
    def __init__(self, items, hidden=600, latent=200, dropout=0.5):
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([nn.Linear(items, hidden)] + [
            nn.Linear(hidden, hidden) for _ in range(4)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(5)])
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        for layer in list(self.layers) + [self.mu, self.logvar]:
            init_linear(layer)

    def execute(self, x, corrupt=True):
        x = jt.normalize(x, p=2, dim=1, eps=1e-12)
        if corrupt:
            x = self.dropout(x)
        values = []
        for layer, norm in zip(self.layers, self.norms):
            residual = sum(values[1:], values[0]) if values else 0.0
            hidden = swish(norm(layer(x if not values else values[-1]) + residual))
            values.append(hidden)
        hidden = sum(values[1:], values[0])
        return self.mu(hidden), jt.clamp(self.logvar(hidden), -10.0, 10.0)


class RecVAE(nn.Module):
    def __init__(self, items, hidden=600, latent=200, dropout=0.5):
        self.encoder = RecEncoder(items, hidden, latent, dropout)
        self.prior_encoder = RecEncoder(items, hidden, latent, dropout)
        self.decoder = nn.Linear(latent, items)
        init_linear(self.decoder)
        self.update_prior()
        for parameter in self.prior_encoder.parameters():
            parameter.stop_grad()

    def update_prior(self):
        self.prior_encoder.load_state_dict(self.encoder.state_dict())

    @staticmethod
    def log_normal(x, mu, logvar):
        return -0.5 * (
            math.log(2.0 * math.pi) + logvar + (x - mu) * (x - mu) / jt.exp(logvar)
        )

    def loss(self, x, corrupt=True, gamma=0.005):
        mu, logvar = self.encoder(x, corrupt)
        z = mu + jt.randn(mu.shape) * jt.exp(0.5 * logvar)
        logits = self.decoder(z)
        with jt.no_grad():
            old_mu, old_logvar = self.prior_encoder(x, False)
        zeros = jt.zeros_like(z)
        log_q = self.log_normal(z, mu, logvar)
        components = jt.stack(
            [
                self.log_normal(z, zeros, zeros) + math.log(0.15),
                self.log_normal(z, old_mu, old_logvar) + math.log(0.75),
                self.log_normal(z, zeros, jt.full_like(z, 10.0)) + math.log(0.10),
            ],
            dim=-1,
        )
        kl = (log_q - nn.logsumexp(components, dim=-1)).sum(dim=1)
        likelihood = (nn.log_softmax(logits, dim=1) * x).sum(dim=1)
        loss = -(likelihood - gamma * x.sum(dim=1) * kl).mean()
        return loss, -likelihood.mean(), kl.mean()


def scalar(value):
    return float(np.asarray(value.data).item())


def multvae_epoch(model, matrix, optimizer, rng, batch, updates):
    order = rng.permutation(matrix.shape[0])
    totals = np.zeros(3, np.float64)
    model.train()
    for start in range(0, len(order), batch):
        ids = order[start : start + batch]
        x = jt.array(matrix.dense(ids))
        logits, mu, logvar = model(x)
        recon = -(nn.log_softmax(logits, dim=1) * x).sum(dim=1).mean()
        kl = -0.5 * (1.0 + logvar - mu * mu - jt.exp(logvar)).sum(dim=1).mean()
        anneal = min(0.2, updates / 2000.0 * 0.2)
        loss = recon + anneal * kl
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.clip_grad_norm(5.0, 2)
        optimizer.step()
        totals += np.array([scalar(loss), scalar(recon), scalar(kl)]) * len(ids)
        updates += 1
    return totals / matrix.shape[0], updates


def set_trainable(module, enabled):
    for parameter in module.parameters():
        parameter.start_grad() if enabled else parameter.stop_grad()


def recvae_epoch(model, matrix, optimizer, rng, batch, train_encoder):
    set_trainable(model.encoder, train_encoder)
    set_trainable(model.decoder, not train_encoder)
    order = rng.permutation(matrix.shape[0])
    totals = np.zeros(3, np.float64)
    model.train()
    for start in range(0, len(order), batch):
        ids = order[start : start + batch]
        x = jt.array(matrix.dense(ids))
        loss, recon, kl = model.loss(x, corrupt=train_encoder)
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.clip_grad_norm(5.0, 2)
        optimizer.step()
        totals += np.array([scalar(loss), scalar(recon), scalar(kl)]) * len(ids)
    return totals / matrix.shape[0]


def export_model(path, model, model_type, users, items, matrix, batch, metadata):
    model.eval()
    latent = []
    with jt.no_grad():
        for start in range(0, len(users), batch):
            x = jt.array(matrix.dense(np.arange(start, min(start + batch, len(users)))))
            if model_type == "multvae":
                mu, _ = model.encode(x, False)
                value = model.decoder_hidden(mu)
                decoder = model.out
            else:
                value, _ = model.encoder(x, False)
                decoder = model.decoder
            latent.append(np.asarray(value.data, np.float32))
    payload = {
        "kind": np.array(model_type),
        "users": users,
        "items": items,
        "user": np.vstack(latent).astype(np.float32),
        "item": np.asarray(decoder.weight.data, np.float32),
        "ibias": np.asarray(decoder.bias.data, np.float32),
        **metadata,
    }
    for name, value in model.state_dict().items():
        payload["param__" + name.replace(".", "__")] = np.asarray(value.data, np.float32)
    np.savez_compressed(path, **payload)
    print(f"wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("multvae", "recvae"))
    parser.add_argument("slice", choices=ORIGINS)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=600)
    parser.add_argument("--latent", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--decay-days", type=float, default=365.0)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    if not jt.has_cuda:
        raise RuntimeError("Jittor CUDA is required")
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    origin = ORIGINS[args.slice]
    users, items, matrix = load_matrix(args.data, origin, args.decay_days)
    print(
        f"model={args.model} slice={args.slice} users={len(users)} items={len(items)} "
        f"edges={matrix.nnz}",
        flush=True,
    )
    rng = np.random.default_rng(args.seed)
    if args.model == "multvae":
        model = MultVAE(len(items), args.hidden, args.latent, args.dropout)
        optimizer = jt.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        updates = 0
        for epoch in range(args.epochs):
            values, updates = multvae_epoch(
                model, matrix, optimizer, rng, args.batch, updates
            )
            print(
                f"epoch={epoch + 1} loss={values[0]:.6f} recon={values[1]:.6f} "
                f"kl={values[2]:.6f}",
                flush=True,
            )
    else:
        model = RecVAE(len(items), args.hidden, args.latent, args.dropout)
        enc_optimizer = jt.optim.Adam(model.encoder.parameters(), lr=5e-4)
        dec_optimizer = jt.optim.Adam(model.decoder.parameters(), lr=5e-4)
        for cycle in range(args.cycles):
            for step in range(3):
                values = recvae_epoch(
                    model, matrix, enc_optimizer, rng, args.batch, True
                )
                print(
                    f"cycle={cycle + 1} encoder={step + 1} loss={values[0]:.6f} "
                    f"recon={values[1]:.6f} kl={values[2]:.6f}",
                    flush=True,
                )
            model.update_prior()
            values = recvae_epoch(
                model, matrix, dec_optimizer, rng, args.batch, False
            )
            print(
                f"cycle={cycle + 1} decoder loss={values[0]:.6f} "
                f"recon={values[1]:.6f} kl={values[2]:.6f}",
                flush=True,
            )

    export_model(
        args.output,
        model,
        args.model,
        users,
        items,
        matrix,
        args.batch,
        {
            "history_end": np.int64(origin),
            "decay_days": np.float64(args.decay_days),
            "hidden": np.int64(args.hidden),
            "latent": np.int64(args.latent),
            "seed": np.int64(args.seed),
        },
    )


if __name__ == "__main__":
    main()
