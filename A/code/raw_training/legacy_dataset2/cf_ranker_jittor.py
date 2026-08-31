#!/usr/bin/env python3
import argparse
import os
import random
import sys
import zipfile
from collections import defaultdict, deque

site_paths = [p for p in sys.path if "site-packages" in p]
old_pythonpath = os.environ.get("PYTHONPATH")
if old_pythonpath:
    site_paths.append(old_pythonpath)
os.environ["PYTHONPATH"] = os.pathsep.join(site_paths)
cache_root = os.environ.get("ML_CACHE_ROOT", "/tmp")
os.environ["HOME"] = cache_root
os.environ.setdefault("use_mpi", "0")
if os.environ.get("JT_USE_CUDA") != "1":
    os.environ["nvcc_path"] = ""
os.environ.setdefault("log_silent", "1")

import jittor as jt
from jittor import nn
import numpy as np
import pandas as pd
import numba

if os.environ.get("JT_USE_CUDA") == "1" and jt.has_cuda:
    jt.flags.use_cuda = 1


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


DATA = os.environ.get("DATA_PATH", "data_A.zip")
BASE = 1 << 32
RAW_DIM = 22
SRC_FREQ_DIM = 24
HIST_LEN = 8
# First-stage graph propagation strength, added on top of the base Jittor model.
# Re-tuned via held-out validation grid search (no retraining): dataset1 0.10->0.08.
# dataset2: 0.15 -> 0.02 -> 0.0 -> 0.02, re-tuned each time m2/r2 get retrained
# (scaling dataset2's base model to groups=100000 shifted the optimum again).
PROP_ALPHA = {"dataset1": 0.08, "dataset2": 0.0}
# Second-stage ranker strength. Re-tuned repeatedly as m2 (use_hist=True, then
# groups=100000/140000/180000) and r2 (retrained on each new set of embeddings)
# improved: 0.20 -> 0.50 -> 0.80 -> 0.60 -> 0.70 -> 0.50. Not monotonic —
# re-tune after every base/ranker retrain rather than assuming it always goes up.
RANK_BLEND = {"dataset2": 0.50}
# Tiny transductive hot-candidate correction from the official 100-candidate pools.
# It is query-normalized, so it only nudges ordering inside each row instead of changing score scale globally.
# dataset1's bonus was tuned away (0.08 -> 0.0): validation showed it no longer helps once
# PROP_ALPHA/RANK_BLEND are re-tuned. dataset2: 0.02 -> 0.03 -> 0.05, holding at 0.05.
FREQ_BONUS = {"dataset1": 0.0, "dataset2": 0.05}
# Post-hoc Jittor CF-embedding blend. The raw driver trains the full-history
# user/item embedding and passes its immutable path through CF_EMB.
CF_BLEND = {"dataset1": 0.0, "dataset2": 0.15}
CF_EMB_PATH = {"dataset2": "cf_full_d128_ours.npz"}
_cf_cache = {}


def load_cf_emb(scene):
    if scene in _cf_cache:
        return _cf_cache[scene]
    w = CF_BLEND.get(scene, 0.0)
    path = os.environ.get("CF_EMB", "") or CF_EMB_PATH.get(scene, "")
    cf = None
    if w and path and os.path.exists(path):
        z = np.load(path)
        cf = (z["user"].astype(np.float32), z["item"].astype(np.float32), z["ibias"].astype(np.float32))
    _cf_cache[scene] = cf
    return cf


def candidate_cf_bonus(src, cand, cf, scene):
    w = CF_BLEND.get(scene, 0.0)
    if not w or cf is None:
        return 0.0
    U, I, IB = cf
    cfdot = (U[src, None, :] * I[cand]).sum(axis=2) + IB[cand]
    return w * query_norm_score(cfdot)


def _ensemble_paths(scene):
    # m{d}_ens0.pkl, m{d}_ens1.pkl, ... if present, else the single m{d}.pkl.
    import glob
    d = scene[-1]
    members = sorted(glob.glob(f"m{d}_ens*.pkl"))
    return members if members else [f"m{d}.pkl"]


class Net(nn.Module):
    # Base scorer: hand-built temporal/graph features + MF-style src/dst embeddings.
    def __init__(self, dim, nodes, emb=64, use_hist=False):
        super().__init__()
        self.use_hist = bool(use_hist)
        self.src_emb = nn.Embedding(nodes, emb)
        self.dst_emb = nn.Embedding(nodes, emb)
        self.src_bias = nn.Embedding(nodes, 1)
        self.dst_bias = nn.Embedding(nodes, 1)
        self.src_emb.weight.assign(jt.randn(self.src_emb.weight.shape) * 0.01)
        self.dst_emb.weight.assign(jt.randn(self.dst_emb.weight.shape) * 0.01)
        self.src_bias.weight.assign(jt.zeros(self.src_bias.weight.shape))
        self.dst_bias.weight.assign(jt.zeros(self.dst_bias.weight.shape))
        self.layers = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Relu(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.Relu(),
            nn.Linear(64, 1),
        )
        self.emb_scale = 1.0
        self.hist_scale = 0.7

    def execute(self, x, src, dst, hist_ids=None, hist_mask=None):
        mlp = self.layers(x).squeeze(-1)
        # Jittor MF term: learns src-dst identity interaction beyond hand-built features.
        dst_vec = self.dst_emb(dst)
        dot = (self.src_emb(src) * dst_vec).sum(dim=1) * self.emb_scale
        if self.use_hist and hist_ids is not None and hist_mask is not None:
            # Sequential preference: recent destination embeddings summarize the source's short-term intent.
            hmask = hist_mask.unsqueeze(-1)
            hvec = (self.dst_emb(hist_ids) * hmask).sum(dim=1) / (hmask.sum(dim=1) + 1e-6)
            dot = dot + (hvec * dst_vec).sum(dim=1) * self.hist_scale
        bias = self.src_bias(src).squeeze(-1) + self.dst_bias(dst).squeeze(-1)
        return mlp + dot + bias


class NetAttn(nn.Module):
    # Same hand-built-feature MLP + MF backbone as Net, but replaces the plain masked-average
    # history summary with genuine target-attention over the source's recent-destination
    # sequence: query = candidate embedding, keys/values = historical destination embeddings,
    # with a learned additive penalty for real elapsed time (not just sequence position).
    def __init__(self, dim, nodes, emb=64, hist_len=HIST_LEN):
        super().__init__()
        self.emb = emb
        self.hist_len = hist_len
        self.src_emb = nn.Embedding(nodes, emb)
        self.dst_emb = nn.Embedding(nodes, emb)
        self.src_bias = nn.Embedding(nodes, 1)
        self.dst_bias = nn.Embedding(nodes, 1)
        self.src_emb.weight.assign(jt.randn(self.src_emb.weight.shape) * 0.01)
        self.dst_emb.weight.assign(jt.randn(self.dst_emb.weight.shape) * 0.01)
        self.src_bias.weight.assign(jt.zeros(self.src_bias.weight.shape))
        self.dst_bias.weight.assign(jt.zeros(self.dst_bias.weight.shape))
        self.layers = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Relu(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.Relu(),
            nn.Linear(64, 1),
        )
        self.q_proj = nn.Linear(emb, emb)
        self.k_proj = nn.Linear(emb, emb)
        self.v_proj = nn.Linear(emb, emb)
        self.attn_dropout = nn.Dropout(0.3)
        self.hist_emb_dropout = nn.Dropout(0.2)
        self.emb_scale = 1.0
        self.hist_scale = jt.array(np.array([0.7], dtype=np.float32))
        self.time_w = jt.array(np.array([0.5], dtype=np.float32))

    def execute(self, x, src, dst, hist_ids, hist_valid, hist_gap):
        mlp = self.layers(x).squeeze(-1)
        dst_vec = self.dst_emb(dst)
        dot = (self.src_emb(src) * dst_vec).sum(dim=1) * self.emb_scale

        hist_vecs = self.hist_emb_dropout(self.dst_emb(hist_ids))  # (B, L, emb)
        q = self.q_proj(dst_vec).unsqueeze(1)  # (B, 1, emb)
        k = self.k_proj(hist_vecs)  # (B, L, emb)
        v = self.v_proj(hist_vecs)  # (B, L, emb)
        scores = (q * k).sum(dim=-1) / (self.emb ** 0.5)  # (B, L)
        scores = scores - self.time_w * hist_gap
        any_valid = hist_valid.sum(dim=1, keepdims=True) > 0
        neg_inf = jt.full_like(scores, -1e9)
        scores = jt.where(hist_valid > 0, scores, neg_inf)
        attn = nn.softmax(scores, dim=-1)
        attn = jt.where(any_valid, attn, jt.zeros_like(attn))
        attn = self.attn_dropout(attn)
        ctx = (attn.unsqueeze(-1) * v).sum(dim=1)  # (B, emb)
        hist_score = (ctx * dst_vec).sum(dim=1) * self.hist_scale

        bias = self.src_bias(src).squeeze(-1) + self.dst_bias(dst).squeeze(-1)
        return mlp + dot + hist_score + bias


class FastRanker(nn.Module):
    # Second-stage scorer for hard dataset2 cases; it learns from compact structural signals only.
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.layers = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Relu(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.Relu(),
            nn.Linear(64, 1),
        )

    def execute(self, x):
        return self.layers(x).squeeze(-1)


class Stats:
    # Cached graph statistics built from allowed historical edges.
    def __init__(self, edges, max_node):
        self.max_node = int(max_node)
        n = self.max_node + 1
        self.src_cnt = np.zeros(n, np.float32)
        self.dst_cnt = np.zeros(n, np.float32)
        self.src_last = np.full(n, -1, np.int64)
        self.dst_last = np.full(n, -1, np.int64)
        self.pair_cnt = {}
        self.pair_last = {}
        self.recent_out = defaultdict(lambda: deque(maxlen=64))
        self.recent_in = defaultdict(lambda: deque(maxlen=64))
        self.recent_out_set = defaultdict(set)
        self.recent_in_set = defaultdict(set)
        # Parallel to recent_out: the timestamp of each historical (u, v) edge, for
        # attention models that need a real elapsed-time signal, not just position.
        self.recent_out_times = defaultdict(lambda: deque(maxlen=64))
        self.build(edges)

    def build(self, edges):
        order = np.argsort(edges[:, 2], kind="mergesort")
        self.pair_first = {}
        self.recent_in_times = defaultdict(lambda: deque(maxlen=64))
        for u, v, t in edges[order]:
            u = int(u)
            v = int(v)
            t = int(t)
            k = u * BASE + v
            if k not in self.pair_cnt:
                self.pair_first[k] = t
            self.pair_cnt[k] = self.pair_cnt.get(k, 0) + 1
            self.pair_last[k] = t
            self.src_cnt[u] += 1
            self.dst_cnt[v] += 1
            self.src_last[u] = t
            self.dst_last[v] = t
            self.recent_out[u].appendleft(v)
            self.recent_in[v].appendleft(u)
            self.recent_out_times[u].appendleft(t)
            self.recent_in_times[v].appendleft(t)
        for v, xs in self.recent_in.items():
            self.recent_in_set[v] = set(xs)
        for u, xs in self.recent_out.items():
            self.recent_out_set[u] = set(xs)
        # Per-dst time-of-day / day-of-week interaction histograms (vectorized, for
        # the v2 periodicity features). Kernel normalizes by dst_cnt.
        n = self.max_node + 1
        self.dst_hour_hist = np.zeros((n, 24), np.float32)
        self.dst_dow_hist = np.zeros((n, 7), np.float32)
        if len(edges):
            dsts = edges[:, 1].astype(np.int64)
            hours = ((edges[:, 2] % 86400) // 3600).astype(np.int64)
            dows = ((edges[:, 2] // 86400) % 7).astype(np.int64)
            np.add.at(self.dst_hour_hist, (dsts, hours), 1.0)
            np.add.at(self.dst_dow_hist, (dsts, dows), 1.0)
        self._arrays = None

    def get_arrays(self):
        # Lazily built, cached: padded fixed-width array views of the dict/set/deque
        # fields above, for the numba-JIT feature kernel. Built once per Stats object
        # (O(#nodes with history), not O(edges) or O(groups*100)), reused across every
        # raw_features_numba call that shares this Stats instance.
        if self._arrays is None:
            self._arrays = stats_to_arrays(self)
        return self._arrays


def recency(last, t):
    if last < 0:
        return 0.0
    return 1.0 / (1.0 + np.log1p(max(0, t - int(last)) / 86400.0))


def read_scene(scene):
    with zipfile.ZipFile(DATA) as z:
        train = pd.read_csv(z.open(f"{scene}/train.csv"))
        test = pd.read_csv(z.open(f"{scene}/test.csv"))
    return train, test


def node_max(train, test):
    cand_max = int(test.iloc[:, 2:].to_numpy(np.int64, copy=False).max())
    return max(int(train.src.max()), int(train.dst.max()), int(test.src.max()), cand_max)


def test_pool_freq(test, max_node):
    # Candidate-pool frequency is transductive but allowed: it uses only the provided test candidate ids.
    cand = test.iloc[:, 2:].to_numpy(np.int64, copy=False)
    vals = cand.ravel()
    freq = np.bincount(vals, minlength=max_node + 1).astype(np.float32)
    src = test.src.to_numpy(np.int64, copy=False)
    src_freq = {}
    for u in np.unique(src):
        xs = cand[src == u].ravel()
        uniq, cnt = np.unique(xs, return_counts=True)
        for v, c in zip(uniq[cnt > 1], cnt[cnt > 1]):
            src_freq[int(u) * BASE + int(v)] = np.log1p(float(c))
    return vals, np.log1p(freq), src_freq


def split_edges(scene, train):
    cols = ["src", "dst", "time"]
    if "split" in train.columns:
        hist = train[train["split"] == 0][cols].to_numpy(np.int64)
        target_raw = train[train["split"] == 1][cols].to_numpy(np.int64)
        target = target_raw[np.argsort(target_raw[:, 2], kind="mergesort")]
    else:
        arr_raw = train[cols].to_numpy(np.int64)
        arr = arr_raw[np.argsort(arr_raw[:, 2], kind="mergesort")]
        cut = int(len(arr) * 0.80)
        hist, target = arr[:cut], arr[cut:]
    valid_cut = max(1, int(len(target) * 0.80))
    return hist, target[:valid_cut], target[valid_cut:]


def sample_groups(pos_edges, pool, groups, seed):
    # Training mimics the official task: one true dst is mixed with 99 hard negatives from the test candidate pool.
    rng = np.random.default_rng(seed)
    n = min(groups, len(pos_edges))
    idx = np.sort(rng.choice(len(pos_edges), n, replace=False))
    pos = pos_edges[idx]
    cand = rng.choice(pool, size=(n, 100), replace=True).astype(np.int64)
    label = rng.integers(0, 100, size=n, dtype=np.int32)
    cand[np.arange(n), label] = pos[:, 1]
    bad = cand == pos[:, 1:2]
    bad[np.arange(n), label] = False
    while bad.any():
        cand[bad] = rng.choice(pool, size=int(bad.sum()), replace=True)
        bad = cand == pos[:, 1:2]
        bad[np.arange(n), label] = False
    return pos[:, 0], pos[:, 2], cand, label


def raw_features(stats, srcs, times, cands, freq, src_freq, use_src_freq):
    # Full feature set for the base model. It is slower because it includes set-overlap graph features.
    n = len(srcs)
    dim = SRC_FREQ_DIM if use_src_freq else RAW_DIM
    x = np.empty((n, 100, dim), np.float32)
    for i in range(n):
        u = int(srcs[i])
        t = int(times[i])
        cs = cands[i].astype(np.int64, copy=False)
        recent = stats.recent_out.get(u, ())
        out_set = stats.recent_out_set.get(u, set())
        in_set = stats.recent_in_set.get(u, set())
        pos = {int(v): p for p, v in enumerate(recent)}
        src_log = np.log1p(stats.src_cnt[u]) if u <= stats.max_node else 0.0
        src_gap = recency(stats.src_last[u], t) if u <= stats.max_node else 0.0
        row = x[i]
        for j, v0 in enumerate(cs):
            v = int(v0)
            k = u * BASE + v
            rk = v * BASE + u
            pc = self_get(stats.pair_cnt, k)
            rc = self_get(stats.pair_cnt, rk)
            p = pos.get(v, -1)
            dst_cnt = stats.dst_cnt[v] if v <= stats.max_node else 0.0
            src2_cnt = stats.src_cnt[v] if v <= stats.max_node else 0.0
            dst_last = stats.dst_last[v] if v <= stats.max_node else -1
            src2_last = stats.src_last[v] if v <= stats.max_node else -1
            pgap = recency(stats.pair_last.get(k, -1), t)
            rgap = recency(stats.pair_last.get(rk, -1), t)
            recent_decay = np.exp(-p / 8.0) if p >= 0 else 0.0
            vin = stats.recent_in_set.get(v, set())
            vout = stats.recent_out_set.get(v, set())
            in_hit = 1.0 if u in vin else 0.0
            twohop = len(out_set & vin)
            out_ov = len(out_set & vout)
            in_ov = len(in_set & vin)
            twohop_j = twohop / np.sqrt((len(out_set) + 1.0) * (len(vin) + 1.0))
            f = freq[v] if v < len(freq) else 0.0
            # Transductive candidate-pool prior: repeated candidates under the same src are strong cold-start clues.
            sf = src_freq.get(k, 0.0) if use_src_freq else 0.0
            h = (
                3.0 * np.log1p(pc)
                + 2.0 * pgap
                + 1.4 * recent_decay
                + 1.1 * in_hit
                + 1.4 * np.log1p(twohop)
                + 0.8 * np.log1p(out_ov)
                + 0.7 * np.log1p(in_ov)
                + 0.9 * np.log1p(rc)
                + 0.45 * np.log1p(dst_cnt)
                + 0.35 * f
                + 0.75 * sf
            )
            base = (
                np.log1p(pc),
                pgap,
                src_log,
                np.log1p(dst_cnt),
                src_gap,
                recency(dst_last, t),
                np.log1p(rc),
                rgap,
                1.0 if p >= 0 else 0.0,
                recent_decay,
                in_hit,
                f,
                1.0 if u == v else 0.0,
                np.log1p(src2_cnt),
                recency(src2_last, t),
                np.log1p(len(stats.recent_out.get(v, ()))),
                np.log1p(len(stats.recent_in.get(v, ()))),
                np.log1p(twohop),
                np.log1p(out_ov),
                np.log1p(in_ov),
                twohop_j,
                h,
            )
            if use_src_freq:
                row[j] = base[:-1] + (sf, 1.0 if sf > 0 else 0.0, h)
            else:
                row[j] = base
    return add_query_norm(x)


def self_get(d, k):
    return d.get(k, 0)


def stats_to_arrays(stats):
    # One-time conversion of a Stats object's dict/set/deque fields into padded
    # fixed-width (64) numpy arrays for the numba feature kernel. Cheap: O(#nodes
    # with any history), not O(edges) or O(groups*100).
    n = stats.max_node + 1

    recent_out_arr = np.full((n, 64), -1, np.int64)
    recent_out_len = np.zeros(n, np.int64)
    for u, xs in stats.recent_out.items():
        L = len(xs)
        recent_out_arr[u, :L] = xs
        recent_out_len[u] = L

    recent_in_arr = np.full((n, 64), -1, np.int64)
    recent_in_len = np.zeros(n, np.int64)
    for v, xs in stats.recent_in.items():
        L = len(xs)
        recent_in_arr[v, :L] = xs
        recent_in_len[v] = L

    recent_out_set_arr = np.full((n, 64), -1, np.int64)
    recent_out_set_len = np.zeros(n, np.int64)
    for u, s in stats.recent_out_set.items():
        vals = list(s)
        L = len(vals)
        recent_out_set_arr[u, :L] = vals
        recent_out_set_len[u] = L

    recent_in_set_arr = np.full((n, 64), -1, np.int64)
    recent_in_set_len = np.zeros(n, np.int64)
    for v, s in stats.recent_in_set.items():
        vals = list(s)
        L = len(vals)
        recent_in_set_arr[v, :L] = vals
        recent_in_set_len[v] = L

    if stats.pair_cnt:
        pair_keys = np.fromiter(stats.pair_cnt.keys(), dtype=np.int64, count=len(stats.pair_cnt))
        pair_counts = np.fromiter(stats.pair_cnt.values(), dtype=np.float32, count=len(stats.pair_cnt))
        pair_lasts = np.fromiter((stats.pair_last[k] for k in stats.pair_cnt.keys()), dtype=np.int64, count=len(stats.pair_cnt))
        pair_firsts = np.fromiter((stats.pair_first[k] for k in stats.pair_cnt.keys()), dtype=np.int64, count=len(stats.pair_cnt))
        order = np.argsort(pair_keys)
        pair_keys, pair_counts, pair_lasts, pair_firsts = (
            pair_keys[order], pair_counts[order], pair_lasts[order], pair_firsts[order])
    else:
        pair_keys = np.empty(0, np.int64)
        pair_counts = np.empty(0, np.float32)
        pair_lasts = np.empty(0, np.int64)
        pair_firsts = np.empty(0, np.int64)

    recent_in_times_arr = np.full((n, 64), -1, np.int64)
    for v, xs in stats.recent_in_times.items():
        L = len(xs)
        recent_in_times_arr[v, :L] = xs

    return dict(
        recent_out_arr=recent_out_arr, recent_out_len=recent_out_len,
        recent_in_arr=recent_in_arr, recent_in_len=recent_in_len,
        recent_out_set_arr=recent_out_set_arr, recent_out_set_len=recent_out_set_len,
        recent_in_set_arr=recent_in_set_arr, recent_in_set_len=recent_in_set_len,
        pair_keys=pair_keys, pair_counts=pair_counts, pair_lasts=pair_lasts,
        pair_firsts=pair_firsts, recent_in_times_arr=recent_in_times_arr,
    )


@numba.njit(cache=True, parallel=True)
def _raw_features_kernel(
    u_arr, t_arr, cand_arr,
    src_cnt, dst_cnt, src_last, dst_last,
    pair_keys, pair_counts, pair_lasts,
    recent_out_arr, recent_out_len,
    recent_in_arr, recent_in_len,
    recent_out_set_arr, recent_out_set_len,
    recent_in_set_arr, recent_in_set_len,
    freq, sf_keys, sf_vals,
    use_src_freq, max_node, out,
):
    # Numba-JIT port of raw_features' inner loop. Same algorithm/semantics as the
    # original (including its 'last-match-wins' quirk when a candidate repeats in
    # the recent_out deque), just using arrays + binary search instead of Python
    # dict/set lookups so it JIT-compiles to native code (measured 75x-195x faster).
    n = u_arr.shape[0]
    n_pairs = pair_keys.shape[0]
    n_sf = sf_keys.shape[0]
    n_freq = freq.shape[0]
    for i in numba.prange(n):
        u = u_arr[i]
        t = t_arr[i]
        if u <= max_node:
            src_log = np.log1p(src_cnt[u])
            sl = src_last[u]
            if sl < 0:
                src_gap = 0.0
            else:
                gap = t - sl
                if gap < 0:
                    gap = 0
                src_gap = 1.0 / (1.0 + np.log1p(gap / 86400.0))
            u_out_len = recent_out_set_len[u]
            u_in_len = recent_in_set_len[u]
            u_recent_len = recent_out_len[u]
        else:
            src_log = 0.0
            src_gap = 0.0
            u_out_len = 0
            u_in_len = 0
            u_recent_len = 0

        for j in range(100):
            v = cand_arr[i, j]
            k = u * BASE + v
            rk = v * BASE + u

            pc = 0.0
            pgap = 0.0
            if n_pairs > 0:
                pos = np.searchsorted(pair_keys, k)
                if pos < n_pairs and pair_keys[pos] == k:
                    pc = pair_counts[pos]
                    last = pair_lasts[pos]
                    if last >= 0:
                        gap = t - last
                        if gap < 0:
                            gap = 0
                        pgap = 1.0 / (1.0 + np.log1p(gap / 86400.0))

            rc = 0.0
            rgap = 0.0
            if n_pairs > 0:
                posr = np.searchsorted(pair_keys, rk)
                if posr < n_pairs and pair_keys[posr] == rk:
                    rc = pair_counts[posr]
                    lastr = pair_lasts[posr]
                    if lastr >= 0:
                        gapr = t - lastr
                        if gapr < 0:
                            gapr = 0
                        rgap = 1.0 / (1.0 + np.log1p(gapr / 86400.0))

            if v <= max_node:
                dst_cnt_v = dst_cnt[v]
                src2_cnt_v = src_cnt[v]
                dl = dst_last[v]
                sl2 = src_last[v]
                out_len_v = recent_out_len[v]
                in_len_v = recent_in_len[v]
                v_out_set_len = recent_out_set_len[v]
                v_in_set_len = recent_in_set_len[v]
            else:
                dst_cnt_v = 0.0
                src2_cnt_v = 0.0
                dl = -1
                sl2 = -1
                out_len_v = 0
                in_len_v = 0
                v_out_set_len = 0
                v_in_set_len = 0

            if dl < 0:
                dst_last_rec = 0.0
            else:
                gapd = t - dl
                if gapd < 0:
                    gapd = 0
                dst_last_rec = 1.0 / (1.0 + np.log1p(gapd / 86400.0))

            if sl2 < 0:
                src2_last_rec = 0.0
            else:
                gaps = t - sl2
                if gaps < 0:
                    gaps = 0
                src2_last_rec = 1.0 / (1.0 + np.log1p(gaps / 86400.0))

            # position of v within u's recent_out deque; later (larger) index wins on repeats,
            # matching the original dict-comprehension overwrite order.
            p = -1
            if u <= max_node:
                for idx in range(u_recent_len):
                    if recent_out_arr[u, idx] == v:
                        p = idx
            if p >= 0:
                recent_decay = np.exp(-p / 8.0)
                has_pos = 1.0
            else:
                recent_decay = 0.0
                has_pos = 0.0

            in_hit = 0.0
            if v <= max_node:
                for idx in range(v_in_set_len):
                    if recent_in_set_arr[v, idx] == u:
                        in_hit = 1.0
                        break

            twohop = 0
            out_ov = 0
            in_ov = 0
            if u <= max_node and v <= max_node:
                for a in range(v_in_set_len):
                    val = recent_in_set_arr[v, a]
                    for b in range(u_out_len):
                        if recent_out_set_arr[u, b] == val:
                            twohop += 1
                            break
                for a in range(v_out_set_len):
                    val = recent_out_set_arr[v, a]
                    for b in range(u_out_len):
                        if recent_out_set_arr[u, b] == val:
                            out_ov += 1
                            break
                for a in range(v_in_set_len):
                    val = recent_in_set_arr[v, a]
                    for b in range(u_in_len):
                        if recent_in_set_arr[u, b] == val:
                            in_ov += 1
                            break

            twohop_j = twohop / np.sqrt((u_out_len + 1.0) * (v_in_set_len + 1.0))

            if v < n_freq:
                f_v = freq[v]
            else:
                f_v = 0.0

            sf = 0.0
            if use_src_freq and n_sf > 0:
                poss = np.searchsorted(sf_keys, k)
                if poss < n_sf and sf_keys[poss] == k:
                    sf = sf_vals[poss]

            same = 1.0 if u == v else 0.0

            h = (3.0 * np.log1p(pc) + 2.0 * pgap + 1.4 * recent_decay + 1.1 * in_hit
                 + 1.4 * np.log1p(twohop) + 0.8 * np.log1p(out_ov) + 0.7 * np.log1p(in_ov)
                 + 0.9 * np.log1p(rc) + 0.45 * np.log1p(dst_cnt_v) + 0.35 * f_v + 0.75 * sf)

            out[i, j, 0] = np.log1p(pc)
            out[i, j, 1] = pgap
            out[i, j, 2] = src_log
            out[i, j, 3] = np.log1p(dst_cnt_v)
            out[i, j, 4] = src_gap
            out[i, j, 5] = dst_last_rec
            out[i, j, 6] = np.log1p(rc)
            out[i, j, 7] = rgap
            out[i, j, 8] = has_pos
            out[i, j, 9] = recent_decay
            out[i, j, 10] = in_hit
            out[i, j, 11] = f_v
            out[i, j, 12] = same
            out[i, j, 13] = np.log1p(src2_cnt_v)
            out[i, j, 14] = src2_last_rec
            out[i, j, 15] = np.log1p(out_len_v)
            out[i, j, 16] = np.log1p(in_len_v)
            out[i, j, 17] = np.log1p(twohop)
            out[i, j, 18] = np.log1p(out_ov)
            out[i, j, 19] = np.log1p(in_ov)
            out[i, j, 20] = twohop_j
            if use_src_freq:
                out[i, j, 21] = sf
                out[i, j, 22] = 1.0 if sf > 0 else 0.0
                out[i, j, 23] = h
            else:
                out[i, j, 21] = h


def raw_features_numba(stats, srcs, times, cands, freq, src_freq, use_src_freq, return_raw=False):
    # Drop-in replacement for raw_features(): identical output (validated to ~1e-6),
    # 75x-195x faster. Uses stats.get_arrays() so the padded-array conversion is
    # amortized across every call that shares the same Stats object.
    # return_raw=True skips the query-norm expansion (for raw_features_numba_v2).
    n = len(srcs)
    dim = SRC_FREQ_DIM if use_src_freq else RAW_DIM
    out = np.empty((n, 100, dim), np.float32)
    arrays = stats.get_arrays()

    if use_src_freq and src_freq:
        sf_keys = np.fromiter(src_freq.keys(), dtype=np.int64, count=len(src_freq))
        sf_vals = np.fromiter(src_freq.values(), dtype=np.float32, count=len(src_freq))
        order = np.argsort(sf_keys)
        sf_keys, sf_vals = sf_keys[order], sf_vals[order]
    else:
        sf_keys = np.empty(0, np.int64)
        sf_vals = np.empty(0, np.float32)

    _raw_features_kernel(
        srcs.astype(np.int64), times.astype(np.int64), cands.astype(np.int64),
        stats.src_cnt, stats.dst_cnt, stats.src_last, stats.dst_last,
        arrays["pair_keys"], arrays["pair_counts"], arrays["pair_lasts"],
        arrays["recent_out_arr"], arrays["recent_out_len"],
        arrays["recent_in_arr"], arrays["recent_in_len"],
        arrays["recent_out_set_arr"], arrays["recent_out_set_len"],
        arrays["recent_in_set_arr"], arrays["recent_in_set_len"],
        freq.astype(np.float32), sf_keys, sf_vals,
        use_src_freq, np.int64(stats.max_node), out,
    )
    if return_raw:
        return out
    return add_query_norm(out)


@numba.njit(cache=True, parallel=True)
def _periodicity_kernel(
    u_arr, t_arr, cand_arr,
    pair_keys, pair_counts, pair_lasts, pair_firsts,
    recent_in_times_arr, dst_hour_hist, dst_dow_hist, dst_cnt,
    max_node, out,
):
    # 6 v2 features per (query, candidate): none of the original 22 use time
    # *periodicity* — they are all recency-based. dataset1 has a strong diurnal
    # cycle (18x hour-of-day swing) and 0.87 repeat rate, so "on-schedule repeat"
    # and "candidate active at this time of day/week" carry fresh signal.
    n = u_arr.shape[0]
    n_pairs = pair_keys.shape[0]
    for i in numba.prange(n):
        u = u_arr[i]
        t = t_arr[i]
        hour = (t % 86400) // 3600
        dow = (t // 86400) % 7
        for j in range(100):
            v = cand_arr[i, j]
            k = u * BASE + v
            interval_score = 0.0
            has_interval = 0.0
            if n_pairs > 0:
                pos = np.searchsorted(pair_keys, k)
                if pos < n_pairs and pair_keys[pos] == k and pair_counts[pos] >= 2.0:
                    mean_int = (pair_lasts[pos] - pair_firsts[pos]) / (pair_counts[pos] - 1.0)
                    if mean_int > 0:
                        has_interval = 1.0
                        gap = t - pair_lasts[pos]
                        if gap < 0:
                            gap = 0
                        ratio = gap / mean_int
                        # peaks at 1.0 when the pair is exactly "due" again
                        d = ratio - 1.0
                        if d < 0:
                            d = -d
                        interval_score = 1.0 / (1.0 + d)

            hour_aff = 1.0
            dow_aff = 1.0
            recent_1d = 0.0
            recent_7d = 0.0
            if v <= max_node:
                c = dst_cnt[v]
                if c > 0:
                    hour_aff = (dst_hour_hist[v, hour] / c) * 24.0
                    dow_aff = (dst_dow_hist[v, dow] / c) * 7.0
                lo1 = t - 86400
                lo7 = t - 7 * 86400
                for idx in range(64):
                    te = recent_in_times_arr[v, idx]
                    if te < 0:
                        break
                    if te <= t:
                        if te >= lo7:
                            recent_7d += 1.0
                            if te >= lo1:
                                recent_1d += 1.0
                recent_1d /= 64.0
                recent_7d /= 64.0

            out[i, j, 0] = interval_score
            out[i, j, 1] = has_interval
            out[i, j, 2] = np.log1p(hour_aff)
            out[i, j, 3] = np.log1p(dow_aff)
            out[i, j, 4] = recent_1d
            out[i, j, 5] = recent_7d


def raw_features_numba_v2(stats, srcs, times, cands, freq, src_freq, use_src_freq):
    # v2 feature set = the original 22/24 features + 6 periodicity features
    # (pre-query-norm concat, so the z/rank expansion covers all of them).
    # Final dim: (RAW_DIM or SRC_FREQ_DIM + 6) * 3. Models trained on v2 features
    # are NOT loadable with v1 features and vice versa (different mu/sd length).
    raw = raw_features_numba(stats, srcs, times, cands, freq, src_freq, use_src_freq, return_raw=True)
    arrays = stats.get_arrays()
    n = len(srcs)
    p6 = np.empty((n, 100, 6), np.float32)
    _periodicity_kernel(
        srcs.astype(np.int64), times.astype(np.int64), cands.astype(np.int64),
        arrays["pair_keys"], arrays["pair_counts"], arrays["pair_lasts"], arrays["pair_firsts"],
        arrays["recent_in_times_arr"], stats.dst_hour_hist, stats.dst_dow_hist, stats.dst_cnt,
        np.int64(stats.max_node), p6,
    )
    return add_query_norm(np.concatenate([raw, p6], axis=2))


@numba.njit(cache=True, parallel=True)
def _bipartite_cf_kernel(u_arr, cand_arr, out_sorted, out_len, in_sorted, in_len, max_node, out):
    # 5 bipartite collaborative-filtering features per (user u, item v), computed by
    # sorted-array merge-intersection over the existing 64-cap recent-neighbor sets:
    #   item-CF: over u's recent items w, co-buyer overlap |in(w) ∩ in(v)| (sum/max/norm)
    #   user-CF: over v's recent buyers z, co-item overlap |out(z) ∩ out(u)| (sum/max)
    n = u_arr.shape[0]
    for i in numba.prange(n):
        u = u_arr[i]
        u_out_len = out_len[u] if u <= max_node else 0
        for j in range(100):
            v = cand_arr[i, j]
            v_in_len = in_len[v] if v <= max_node else 0

            icf_sum = 0.0
            icf_max = 0.0
            if u_out_len > 0 and v_in_len > 0:
                for a in range(u_out_len):
                    w = out_sorted[u, a]
                    wl = in_len[w] if w <= max_node else 0
                    c = 0
                    p1 = 0
                    p2 = 0
                    while p1 < wl and p2 < v_in_len:
                        x1 = in_sorted[w, p1]
                        x2 = in_sorted[v, p2]
                        if x1 == x2:
                            c += 1
                            p1 += 1
                            p2 += 1
                        elif x1 < x2:
                            p1 += 1
                        else:
                            p2 += 1
                    icf_sum += c
                    if c > icf_max:
                        icf_max = c

            ucf_sum = 0.0
            ucf_max = 0.0
            if u_out_len > 0 and v_in_len > 0:
                for a in range(v_in_len):
                    z = in_sorted[v, a]
                    zl = out_len[z] if z <= max_node else 0
                    c = 0
                    p1 = 0
                    p2 = 0
                    while p1 < zl and p2 < u_out_len:
                        x1 = out_sorted[z, p1]
                        x2 = out_sorted[u, p2]
                        if x1 == x2:
                            c += 1
                            p1 += 1
                            p2 += 1
                        elif x1 < x2:
                            p1 += 1
                        else:
                            p2 += 1
                    ucf_sum += c
                    if c > ucf_max:
                        ucf_max = c

            out[i, j, 0] = np.log1p(icf_sum)
            out[i, j, 1] = np.log1p(icf_max)
            out[i, j, 2] = icf_sum / ((u_out_len + 1.0) * np.sqrt(v_in_len + 1.0))
            out[i, j, 3] = np.log1p(ucf_sum)
            out[i, j, 4] = np.log1p(ucf_max)


def _cf_sorted_sets(stats):
    # Sorted copies of the recent-neighbor set arrays (padding pushed to the end with
    # a huge sentinel) so the CF kernel can use O(len_a+len_b) merge intersection.
    cached = getattr(stats, "_cf_sorted", None)
    if cached is None:
        arrays = stats.get_arrays()

        def _sort(arr, lens):
            a = arr.astype(np.int64, copy=True)
            w = a.shape[1]
            a[np.arange(w)[None, :] >= lens[:, None]] = np.int64(1) << 62
            a.sort(axis=1)
            return a

        cached = (
            _sort(arrays["recent_out_set_arr"], arrays["recent_out_set_len"]),
            arrays["recent_out_set_len"].astype(np.int64),
            _sort(arrays["recent_in_set_arr"], arrays["recent_in_set_len"]),
            arrays["recent_in_set_len"].astype(np.int64),
        )
        stats._cf_sorted = cached
    return cached


def raw_features_numba_v3(stats, srcs, times, cands, freq, src_freq, use_src_freq):
    # v3 = v1's 22/24 features + 5 bipartite CF features. Motivation (measured
    # 2026-07-09): dataset2's user/item ID spaces are disjoint, which makes 10 of
    # the 22 v1 features structurally ZERO there — including all four 2-hop/overlap
    # features — so the scene never had any collaborative-filtering signal. A/B at
    # 30k groups/8 epochs: dataset2 base-model val MRR 0.4934 (v1) -> 0.5697 (v3).
    raw = raw_features_numba(stats, srcs, times, cands, freq, src_freq, use_src_freq, return_raw=True)
    out_sorted, out_len, in_sorted, in_len = _cf_sorted_sets(stats)
    n = len(srcs)
    cf = np.empty((n, 100, 5), np.float32)
    _bipartite_cf_kernel(srcs.astype(np.int64), cands.astype(np.int64),
                         out_sorted, out_len, in_sorted, in_len, np.int64(stats.max_node), cf)
    full = np.concatenate([raw, cf], axis=2)
    # Free intermediates before allocating the 3x-expanded output and expand in
    # chunks: at 180k groups the naive path (raw+full+out+add_query_norm temps
    # all resident) peaks >15GB and gets OOM-killed on the 15.5GB WSL VM.
    del raw, cf
    dim = full.shape[2]
    out = np.empty((n, 100, dim * 3), np.float32)
    step = 20000
    for s in range(0, n, step):
        out[s : s + step] = add_query_norm(full[s : s + step])
    del full
    return out


def feat_fn_for_dim(mu_len):
    # Infer a checkpoint's feature config from its scaler length:
    # v1: 66 (22*3) / 72 (24*3); v3 (+5 CF): 81 (27*3) / 87 (29*3).
    use_cf = mu_len in ((RAW_DIM + 5) * 3, (SRC_FREQ_DIM + 5) * 3)
    use_sf = mu_len in (SRC_FREQ_DIM * 3, (SRC_FREQ_DIM + 5) * 3)
    return (raw_features_numba_v3 if use_cf else raw_features_numba), use_sf


def add_query_norm(raw):
    mean = raw.mean(axis=1, keepdims=True)
    std = raw.std(axis=1, keepdims=True) + 1e-6
    z = (raw - mean) / std
    rank = np.empty_like(raw)
    for k in range(raw.shape[2]):
        order = np.argsort(raw[:, :, k], axis=1)
        r = np.empty_like(order, dtype=np.float32)
        r[np.arange(raw.shape[0])[:, None], order] = np.linspace(0.0, 1.0, raw.shape[1], dtype=np.float32)
        rank[:, :, k] = r
    return np.concatenate([raw, z, rank], axis=2)


def scale_fit(x):
    mu = x.reshape(-1, x.shape[-1]).mean(axis=0).astype(np.float32)
    sd = x.reshape(-1, x.shape[-1]).std(axis=0).astype(np.float32) + 1e-6
    return mu, sd


def scale(x, mu, sd):
    return ((x - mu) / sd).astype(np.float32, copy=False)


def flat_ids(src, cand, ids):
    s = np.repeat(src[ids], 100).astype(np.int32, copy=False)
    d = cand[ids].reshape(-1).astype(np.int32, copy=False)
    return s, d


def history_ids(stats, srcs):
    ids = np.zeros((len(srcs), HIST_LEN), np.int32)
    mask = np.zeros((len(srcs), HIST_LEN), np.float32)
    for i, u0 in enumerate(srcs):
        for j, v in enumerate(stats.recent_out.get(int(u0), ())):
            if j >= HIST_LEN:
                break
            ids[i, j] = int(v)
            mask[i, j] = float(np.exp(-j / 3.0))
    return ids, mask


def history_seq(stats, srcs, times, hist_len=HIST_LEN):
    # Like history_ids, but for NetAttn: a clean boolean padding mask plus the real
    # elapsed time (in log1p-days) since each historical interaction, instead of a
    # fixed position-based decay. Lets the attention layer learn its own notion of
    # recency instead of having exp(-j/3) baked in.
    n = len(srcs)
    ids = np.zeros((n, hist_len), np.int32)
    valid = np.zeros((n, hist_len), np.float32)
    gap = np.zeros((n, hist_len), np.float32)
    for i in range(n):
        u0 = int(srcs[i])
        t0 = int(times[i])
        vs = stats.recent_out.get(u0, ())
        ts = stats.recent_out_times.get(u0, ())
        L = min(hist_len, len(vs))
        for j in range(L):
            ids[i, j] = int(vs[j])
            valid[i, j] = 1.0
            dt = t0 - int(ts[j])
            if dt < 0:
                dt = 0
            gap[i, j] = np.log1p(dt / 86400.0)
    return ids, valid, gap


def flat_history(hist_ids, hist_mask, ids):
    h = np.repeat(hist_ids[ids], 100, axis=0).astype(np.int32, copy=False)
    m = np.repeat(hist_mask[ids], 100, axis=0).astype(np.float32, copy=False)
    return h, m


def train_net(
    x,
    src,
    cand,
    y,
    hist_ids,
    hist_mask,
    valid_x,
    valid_src,
    valid_cand,
    valid_y,
    valid_hist_ids,
    valid_hist_mask,
    epochs,
    batch,
    seed,
    nodes,
    use_hist,
    init_net=None,
    lr=1e-3,
    loss_name="ce",
    emb=64,
):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    net = init_net if init_net is not None else Net(x.shape[-1], nodes, emb=emb, use_hist=use_hist)
    cf_init_path = os.environ.get("CF_INIT", "")
    if cf_init_path and init_net is None and os.path.exists(cf_init_path):
        # Warm-start src/dst embeddings from CF embeddings (teammate's recipe).
        cf = np.load(cf_init_path)
        cu = cf["user"].astype(np.float32)
        ci = cf["item"].astype(np.float32)
        if cu.shape[1] == emb and cu.shape[0] >= nodes:
            net.src_emb.weight.assign(jt.array(cu[:nodes]))
            net.dst_emb.weight.assign(jt.array(ci[:nodes]))
            print(f"warm-start from {cf_init_path}", flush=True)
    opt = nn.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    if init_net is not None:
        best_mrr = eval_mrr(net, valid_x, valid_src, valid_cand, valid_y, valid_hist_ids, valid_hist_mask, batch)
        best_state = {k: np.asarray(v.data).copy() for k, v in net.state_dict().items()}
        print(f"epoch=0 mrr={best_mrr:.5f}", flush=True)
    else:
        best_mrr = -1.0
        best_state = None
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(x))
        losses = []
        for s in range(0, len(order), batch):
            ids = order[s : s + batch]
            xb = jt.array(x[ids].reshape(-1, x.shape[-1]))
            sb, db = flat_ids(src, cand, ids)
            hb, hm = flat_history(hist_ids, hist_mask, ids)
            yb = jt.array(y[ids])
            score = net(xb, jt.array(sb), jt.array(db), jt.array(hb), jt.array(hm)).reshape((len(ids), 100))
            ce_loss = nn.cross_entropy_loss(score, yb)
            if loss_name == "ce":
                loss = ce_loss
            else:
                onehot = np.zeros((len(ids), 100), np.float32)
                onehot[np.arange(len(ids)), y[ids]] = 1.0
                oh = jt.array(onehot)
                pos_score = (score * oh).sum(dim=1, keepdims=True)
                pair = jt.log(1.0 + jt.exp(score - pos_score)) * (1.0 - oh)
                pair_loss = pair.sum() / ((len(ids) * 99.0) + 1e-6)
                if loss_name == "pair":
                    loss = pair_loss
                elif loss_name == "mix":
                    loss = 0.5 * ce_loss + 0.5 * pair_loss
                else:
                    raise ValueError(f"unknown loss: {loss_name}")
            opt.step(loss)
            losses.append(float(np.asarray(loss.data).item()))
        mrr = eval_mrr(net, valid_x, valid_src, valid_cand, valid_y, valid_hist_ids, valid_hist_mask, batch)
        if mrr > best_mrr:
            best_mrr = mrr
            best_state = {k: np.asarray(v.data).copy() for k, v in net.state_dict().items()}
        print(f"epoch={ep + 1} loss={np.mean(losses):.5f} mrr={mrr:.5f}", flush=True)
    if best_state is not None:
        net.load_state_dict({k: jt.array(v) for k, v in best_state.items()})
        print(f"best_mrr={best_mrr:.5f}", flush=True)
    return net


def eval_mrr(net, x, src, cand, y, hist_ids, hist_mask, batch):
    net.eval()
    ranks = []
    for s in range(0, len(x), batch):
        ids = np.arange(s, min(s + batch, len(x)))
        xb = jt.array(x[ids].reshape(-1, x.shape[-1]))
        sb, db = flat_ids(src, cand, ids)
        hb, hm = flat_history(hist_ids, hist_mask, ids)
        score = np.asarray(net(xb, jt.array(sb), jt.array(db), jt.array(hb), jt.array(hm)).reshape((-1, 100)).data)
        if not np.isfinite(score).all():
            score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
        yy = y[s : s + batch]
        pos = score[np.arange(len(yy)), yy]
        ranks.extend(1.0 + (score > pos[:, None]).sum(axis=1))
    return float(np.mean(1.0 / np.asarray(ranks)))


def predict(net, x, src, cand, hist_ids, hist_mask, batch):
    net.eval()
    out = []
    for s in range(0, len(x), batch):
        ids = np.arange(s, min(s + batch, len(x)))
        xb = jt.array(x[ids].reshape(-1, x.shape[-1]))
        sb, db = flat_ids(src, cand, ids)
        hb, hm = flat_history(hist_ids, hist_mask, ids)
        score = np.asarray(net(xb, jt.array(sb), jt.array(db), jt.array(hb), jt.array(hm)).reshape((-1, 100)).data)
        if not np.isfinite(score).all():
            score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
        score -= score.max(axis=1, keepdims=True)
        prob = np.exp(score)
        prob /= prob.sum(axis=1, keepdims=True)
        out.append(prob.astype(np.float32))
    return np.vstack(out)


def make_prop_embeddings(edges, nodes, src_emb, dst_emb):
    # LightGCN-style one-hop propagation using embeddings learned by the Jittor base model.
    prop_src = np.zeros_like(src_emb)
    prop_dst = np.zeros_like(dst_emb)
    src_deg = np.zeros(nodes, np.float32)
    dst_deg = np.zeros(nodes, np.float32)
    for u, v, _ in edges[np.argsort(edges[:, 2], kind="mergesort")]:
        u = int(u)
        v = int(v)
        prop_src[u] += dst_emb[v]
        prop_dst[v] += src_emb[u]
        src_deg[u] += 1.0
        dst_deg[v] += 1.0
    prop_src /= np.maximum(src_deg[:, None], 1.0)
    prop_dst /= np.maximum(dst_deg[:, None], 1.0)
    return prop_src, prop_dst


def query_norm_score(score):
    # MRR only cares about ordering inside each 100-candidate query, so scores are normalized per query.
    score = score.astype(np.float32, copy=False)
    return (score - score.mean(axis=1, keepdims=True)) / (score.std(axis=1, keepdims=True) + 1e-6)


def base_score_extra_features(base_score):
    # For a "stacked" FastRanker (trained with the base model's own score as 2 extra
    # per-candidate features): query-normalized score + within-query rank percentile.
    # Lets the ranker learn when to trust vs. override the base scorer, instead of
    # always blending at one fixed RANK_BLEND ratio.
    q = query_norm_score(base_score)
    order = np.argsort(base_score, axis=1)
    r = np.empty_like(order, dtype=np.float32)
    r[np.arange(base_score.shape[0])[:, None], order] = np.linspace(
        0.0, 1.0, base_score.shape[1], dtype=np.float32)
    return np.stack([q, r], axis=2).astype(np.float32)


def prop_scores(src, cand, src_emb, dst_emb, prop_src, prop_dst):
    s1 = query_norm_score((prop_src[src, None, :] * dst_emb[cand]).sum(axis=2))
    s2 = query_norm_score((src_emb[src, None, :] * prop_dst[cand]).sum(axis=2))
    s3 = query_norm_score((prop_src[src, None, :] * prop_dst[cand]).sum(axis=2))
    return query_norm_score(s1 + s2 + 0.5 * s3)


def recency_array(last, t):
    gap = np.maximum(0, t - last)
    return np.where(last >= 0, 1.0 / (1.0 + np.log1p(gap / 86400.0)), 0.0).astype(np.float32)


def pair_index(edges):
    # Sorted pair index lets the fast ranker look up repeat count and last timestamp without Python dict loops.
    if len(edges) == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32), np.empty(0, np.int64)
    keys = edges[:, 0].astype(np.int64) * BASE + edges[:, 1].astype(np.int64)
    order = np.lexsort((edges[:, 2], keys))
    keys = keys[order]
    times = edges[:, 2][order]
    uniq, start, cnt = np.unique(keys, return_index=True, return_counts=True)
    last = np.maximum.reduceat(times, start).astype(np.int64)
    return uniq.astype(np.int64), cnt.astype(np.float32), last


def lookup_pairs(keys, cnt, last, query):
    flat = query.reshape(-1)
    if len(keys) == 0:
        return np.zeros(query.shape, np.float32), np.full(query.shape, -1, np.int64)
    pos = np.searchsorted(keys, flat)
    safe = np.minimum(pos, len(keys) - 1)
    ok = (pos < len(keys)) & (keys[safe] == flat)
    out_cnt = np.zeros(len(flat), np.float32)
    out_last = np.full(len(flat), -1, np.int64)
    out_cnt[ok] = cnt[pos[ok]]
    out_last[ok] = last[pos[ok]]
    return out_cnt.reshape(query.shape), out_last.reshape(query.shape)


def sorted_lookup(keys, vals, query):
    flat = query.reshape(-1)
    out = np.zeros(len(flat), np.float32)
    if len(keys):
        pos = np.searchsorted(keys, flat)
        safe = np.minimum(pos, len(keys) - 1)
        ok = (pos < len(keys)) & (keys[safe] == flat)
        out[ok] = vals[pos[ok]]
    return out.reshape(query.shape)


def rank_by_query(raw):
    rank = np.empty_like(raw, dtype=np.float32)
    for k in range(raw.shape[2]):
        order = np.argsort(raw[:, :, k], axis=1)
        r = np.empty_like(order, dtype=np.float32)
        r[np.arange(raw.shape[0])[:, None], order] = np.linspace(0.0, 1.0, raw.shape[1], dtype=np.float32)
        rank[:, :, k] = r
    return rank


def make_fast_context(edges, max_node, src_emb, dst_emb, src_freq):
    # Shared context for fast-rank features: pair index, propagated embeddings, and src-candidate priors.
    keys, cnt, last = pair_index(edges)
    prop_src, prop_dst = make_prop_embeddings(edges, int(max_node) + 1, src_emb, dst_emb)
    if src_freq:
        sf_keys = np.fromiter(src_freq.keys(), dtype=np.int64, count=len(src_freq))
        sf_vals = np.fromiter(src_freq.values(), dtype=np.float32, count=len(src_freq))
        order = np.argsort(sf_keys)
        sf_keys = sf_keys[order]
        sf_vals = sf_vals[order]
    else:
        sf_keys = np.empty(0, np.int64)
        sf_vals = np.empty(0, np.float32)
    return {
        "keys": keys,
        "cnt": cnt,
        "last": last,
        "src_emb": src_emb,
        "dst_emb": dst_emb,
        "prop_src": prop_src,
        "prop_dst": prop_dst,
        "sf_keys": sf_keys,
        "sf_vals": sf_vals,
    }


def fast_rank_features(ctx, stats, src, tim, cand, freq):
    # Compact dataset2 feature family. It drops slow set intersections and focuses on repeat/recency,
    # node activity, candidate-pool priors, and three propagation views.
    q = src[:, None].astype(np.int64) * BASE + cand.astype(np.int64)
    rq = cand.astype(np.int64) * BASE + src[:, None].astype(np.int64)
    pc, pl = lookup_pairs(ctx["keys"], ctx["cnt"], ctx["last"], q)
    rc, rl = lookup_pairs(ctx["keys"], ctx["cnt"], ctx["last"], rq)
    s1 = query_norm_score((ctx["prop_src"][src, None, :] * ctx["dst_emb"][cand]).sum(axis=2))
    s2 = query_norm_score((ctx["src_emb"][src, None, :] * ctx["prop_dst"][cand]).sum(axis=2))
    s3 = query_norm_score((ctx["prop_src"][src, None, :] * ctx["prop_dst"][cand]).sum(axis=2))
    prop = query_norm_score(s1 + s2 + 0.5 * s3)
    sf = sorted_lookup(ctx["sf_keys"], ctx["sf_vals"], q)
    mats = [
        np.log1p(pc),
        recency_array(pl, tim[:, None]),
        np.log1p(rc),
        recency_array(rl, tim[:, None]),
        np.log1p(stats.src_cnt[src])[:, None].repeat(100, axis=1),
        np.log1p(stats.dst_cnt[cand]),
        recency_array(stats.src_last[src], tim)[:, None].repeat(100, axis=1),
        recency_array(stats.dst_last[cand], tim[:, None]),
        freq[cand],
        sf,
        prop,
        s1,
        s2,
        s3,
    ]
    raw = np.stack(mats, axis=2).astype(np.float32)
    z = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-6)
    return np.concatenate([raw, z.astype(np.float32), rank_by_query(raw)], axis=2)


def save_fast_model(path, net):
    jt.save({"model": net.state_dict(), "dim": net.dim}, path)


def load_fast_model(path):
    obj = jt.load(path)
    net = FastRanker(int(obj["dim"]))
    net.load_state_dict(obj["model"])
    return net


def predict_fast(net, x, batch):
    net.eval()
    out = []
    for s in range(0, len(x), batch):
        part = x[s : s + batch]
        score = np.asarray(net(jt.array(part.reshape(-1, part.shape[-1]))).reshape((-1, 100)).data)
        if not np.isfinite(score).all():
            score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
        score -= score.max(axis=1, keepdims=True)
        prob = np.exp(score)
        prob /= prob.sum(axis=1, keepdims=True)
        out.append(prob.astype(np.float32))
    return np.vstack(out)


def fast_scores(net, x, batch):
    net.eval()
    out = []
    for s in range(0, len(x), batch):
        part = x[s : s + batch]
        score = np.asarray(net(jt.array(part.reshape(-1, part.shape[-1]))).reshape((-1, 100)).data)
        if not np.isfinite(score).all():
            score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
        out.append(score.astype(np.float32))
    return np.vstack(out)


def scores_to_prob(score):
    score = score.astype(np.float32, copy=False)
    score -= score.max(axis=1, keepdims=True)
    prob = np.exp(score)
    prob /= prob.sum(axis=1, keepdims=True)
    return prob.astype(np.float32)


def candidate_freq_bonus(cand, freq, scene):
    gamma = FREQ_BONUS.get(scene, 0.0)
    if not gamma:
        return 0.0
    return gamma * query_norm_score(freq[cand])


def predict_with_prop(net, x, src, cand, hist_ids, hist_mask, batch, prop=None, alpha=0.0):
    net.eval()
    out = []
    for s in range(0, len(x), batch):
        ids = np.arange(s, min(s + batch, len(x)))
        xb = jt.array(x[ids].reshape(-1, x.shape[-1]))
        sb, db = flat_ids(src, cand, ids)
        hb, hm = flat_history(hist_ids, hist_mask, ids)
        score = np.asarray(net(xb, jt.array(sb), jt.array(db), jt.array(hb), jt.array(hm)).reshape((-1, 100)).data)
        if prop is not None and alpha:
            src_emb, dst_emb, prop_src, prop_dst = prop
            score = score + alpha * prop_scores(src[ids], cand[ids], src_emb, dst_emb, prop_src, prop_dst)
        if not np.isfinite(score).all():
            score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
        score -= score.max(axis=1, keepdims=True)
        prob = np.exp(score)
        prob /= prob.sum(axis=1, keepdims=True)
        out.append(prob.astype(np.float32))
    return np.vstack(out)


def model_scores_with_prop(net, x, src, cand, hist_ids, hist_mask, batch, prop=None, alpha=0.0):
    net.eval()
    out = []
    for s in range(0, len(x), batch):
        ids = np.arange(s, min(s + batch, len(x)))
        xb = jt.array(x[ids].reshape(-1, x.shape[-1]))
        sb, db = flat_ids(src, cand, ids)
        hb, hm = flat_history(hist_ids, hist_mask, ids)
        score = np.asarray(net(xb, jt.array(sb), jt.array(db), jt.array(hb), jt.array(hm)).reshape((-1, 100)).data)
        if prop is not None and alpha:
            src_emb, dst_emb, prop_src, prop_dst = prop
            score = score + alpha * prop_scores(src[ids], cand[ids], src_emb, dst_emb, prop_src, prop_dst)
        if not np.isfinite(score).all():
            score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
        out.append(score.astype(np.float32))
    return np.vstack(out)


def save_model(path, net, mu, sd, nodes):
    jt.save({"model": net.state_dict(), "mu": mu, "sd": sd, "nodes": nodes, "use_hist": net.use_hist}, path)


def load_model(path):
    obj = jt.load(path)
    # Infer embedding dim from the checkpoint so larger-emb (e.g. 128) models load correctly.
    emb_dim = int(np.asarray(obj["model"]["src_emb.weight"]).shape[1]) if "src_emb.weight" in obj["model"] else 64
    net = Net(len(obj["mu"]), int(obj["nodes"]), emb=emb_dim, use_hist=bool(obj.get("use_hist", False)))
    net.load_state_dict(obj["model"])
    return net, obj["mu"], obj["sd"]


def fit_scene(
    scene,
    groups,
    epochs,
    batch,
    seed,
    use_hist=False,
    use_srcfreq2=False,
    finetune=False,
    lr=1e-3,
    loss_name="ce",
    valid_groups=0,
    use_cf=False,
    emb=64,
):
    set_seed(seed)
    print(f"{scene}: read", flush=True)
    train, test = read_scene(scene)
    max_node = node_max(train, test)
    pool, freq, src_freq = test_pool_freq(test, max_node)
    hist, tr_pos, va_pos = split_edges(scene, train)
    train_stats = Stats(hist, max_node)
    valid_stats = Stats(np.vstack([hist, tr_pos]), max_node)

    print(f"{scene}: sample", flush=True)
    src, tim, cand, y = sample_groups(tr_pos, pool, groups, seed)
    if valid_groups <= 0:
        valid_groups = max(10000, groups // 3)
    vsrc, vtim, vcand, vy = sample_groups(va_pos, pool, valid_groups, seed + 17)

    print(f"{scene}: features", flush=True)
    use_src_freq = scene == "dataset1" or (scene == "dataset2" and use_srcfreq2)
    feat_fn = raw_features_numba_v3 if use_cf else raw_features_numba
    x = feat_fn(train_stats, src, tim, cand, freq, src_freq, use_src_freq)
    vx = feat_fn(valid_stats, vsrc, vtim, vcand, freq, src_freq, use_src_freq)
    hist_ids, hist_mask = history_ids(train_stats, src)
    valid_hist_ids, valid_hist_mask = history_ids(valid_stats, vsrc)
    init_net = None
    if finetune:
        init_net, mu, sd = load_model(f"m{scene[-1]}.pkl")
        if len(mu) != x.shape[-1] or init_net.use_hist != bool(use_hist):
            raise ValueError("finetune checkpoint feature config does not match this experiment")
    else:
        mu, sd = scale_fit(x)
    x = scale(x, mu, sd)
    vx = scale(vx, mu, sd)

    print(f"{scene}: train groups={len(x)} valid={len(vx)} dim={x.shape[-1]} hist={use_hist}", flush=True)
    net = train_net(
        x,
        src,
        cand,
        y,
        hist_ids,
        hist_mask,
        vx,
        vsrc,
        vcand,
        vy,
        valid_hist_ids,
        valid_hist_mask,
        epochs,
        batch,
        seed,
        max_node + 1,
        use_hist,
        init_net,
        lr,
        loss_name,
        emb,
    )
    save_model(f"m{scene[-1]}.pkl", net, mu, sd, max_node + 1)
    return float(eval_mrr(net, vx, vsrc, vcand, vy, valid_hist_ids, valid_hist_mask, batch))


def eval_ranker_mrr(net, x, y, batch):
    net.eval()
    ranks = []
    for s in range(0, len(x), batch):
        part = x[s : s + batch]
        score = np.asarray(net(jt.array(part.reshape(-1, part.shape[-1]))).reshape((-1, 100)).data)
        if not np.isfinite(score).all():
            score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
        yy = y[s : s + batch]
        pos = score[np.arange(len(yy)), yy]
        ranks.extend(1.0 + (score > pos[:, None]).sum(axis=1))
    return float(np.mean(1.0 / np.asarray(ranks)))


def train_fast_scene(scene, groups, valid_groups, epochs, batch, seed):
    # Train the second-stage ranker on official split positives with candidate-pool negatives.
    # Validation history includes train positives but not validation positives, avoiding temporal leakage.
    set_seed(seed)
    print(f"{scene}: fast-rank read", flush=True)
    train, test = read_scene(scene)
    max_node = node_max(train, test)
    pool, freq, src_freq = test_pool_freq(test, max_node)
    hist, tr_pos, va_pos = split_edges(scene, train)
    if valid_groups <= 0:
        valid_groups = max(20000, groups // 4)

    base, _, _ = load_model(f"m{scene[-1]}.pkl")
    state = base.state_dict()
    src_emb = np.asarray(state["src_emb.weight"].data, dtype=np.float32)
    dst_emb = np.asarray(state["dst_emb.weight"].data, dtype=np.float32)

    print(f"{scene}: fast-rank sample", flush=True)
    src, tim, cand, y = sample_groups(tr_pos, pool, groups, seed)
    vsrc, vtim, vcand, vy = sample_groups(va_pos, pool, valid_groups, seed + 29)

    print(f"{scene}: fast-rank features", flush=True)
    train_stats = Stats(hist, max_node)
    train_ctx = make_fast_context(hist, max_node, src_emb, dst_emb, src_freq)
    x = fast_rank_features(train_ctx, train_stats, src, tim, cand, freq)
    valid_edges = np.vstack([hist, tr_pos])
    valid_stats = Stats(valid_edges, max_node)
    valid_ctx = make_fast_context(valid_edges, max_node, src_emb, dst_emb, src_freq)
    vx = fast_rank_features(valid_ctx, valid_stats, vsrc, vtim, vcand, freq)

    print(f"{scene}: fast-rank train groups={len(x)} valid={len(vx)} dim={x.shape[-1]}", flush=True)
    net = FastRanker(x.shape[-1])
    opt = nn.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    best_mrr = -1.0
    best_state = None
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(x))
        losses = []
        for s in range(0, len(order), batch):
            ids = order[s : s + batch]
            score = net(jt.array(x[ids].reshape(-1, x.shape[-1]))).reshape((len(ids), 100))
            loss = nn.cross_entropy_loss(score, jt.array(y[ids]))
            opt.step(loss)
            losses.append(float(np.asarray(loss.data).item()))
        mrr = eval_ranker_mrr(net, vx, vy, batch)
        if mrr > best_mrr:
            best_mrr = mrr
            best_state = {k: np.asarray(v.data).copy() for k, v in net.state_dict().items()}
        print(f"fast-rank epoch={ep + 1} loss={np.mean(losses):.5f} mrr={mrr:.5f}", flush=True)
    if best_state is not None:
        net.load_state_dict({k: jt.array(v) for k, v in best_state.items()})
    save_fast_model(f"r{scene[-1]}.pkl", net)
    print(f"{scene}: fast-rank best_mrr={best_mrr:.6f}", flush=True)
    return best_mrr


def write_fast_scene(zout, scene, batch):
    # Inference for scenes with r*.pkl: combine the stable base scorer and the fast structural ranker.
    train, test = read_scene(scene)
    max_node = node_max(train, test)
    _, freq, src_freq = test_pool_freq(test, max_node)
    stats = Stats(train[["src", "dst", "time"]].to_numpy(np.int64), max_node)
    base, mu, sd = load_model(f"m{scene[-1]}.pkl")
    state = base.state_dict()
    src_emb = np.asarray(state["src_emb.weight"].data, dtype=np.float32)
    dst_emb = np.asarray(state["dst_emb.weight"].data, dtype=np.float32)
    edges = train[["src", "dst", "time"]].to_numpy(np.int64)
    ctx = make_fast_context(edges, max_node, src_emb, dst_emb, src_freq)
    prop = None
    alpha = PROP_ALPHA.get(scene, 0.0)
    if alpha:
        prop = (src_emb, dst_emb, ctx["prop_src"], ctx["prop_dst"])
    blend = RANK_BLEND.get(scene, 1.0)
    feat_fn, use_src_freq = feat_fn_for_dim(len(mu))
    ranker = load_fast_model(f"r{scene[-1]}.pkl")
    with zout.open(f"{scene}.csv", "w") as f:
        for s in range(0, len(test), batch):
            part = test.iloc[s : s + batch]
            src = part.src.to_numpy(np.int64, copy=False)
            tim = part.time.to_numpy(np.int64, copy=False)
            cand = part.iloc[:, 2:].to_numpy(np.int64, copy=False)
            fx = fast_rank_features(ctx, stats, src, tim, cand, freq)
            # A "stacked" ranker (trained with base_score_extra_features appended, see
            # train_ranker_v3_stack.py) has more input dims than the plain feature set
            # produces; detect that and compute the base score unconditionally in that
            # case (it's now a required ranker input, not just an optional blend partner).
            stacked = fx.shape[-1] != ranker.dim
            base_score = None
            if blend < 1.0 or stacked:
                x = scale(feat_fn(stats, src, tim, cand, freq, src_freq, use_src_freq), mu, sd)
                hist_ids, hist_mask = history_ids(stats, src)
                base_score = model_scores_with_prop(
                    base,
                    x,
                    src,
                    cand,
                    hist_ids,
                    hist_mask,
                    max(128, batch // 4),
                    prop,
                    alpha,
                )
            if stacked:
                fx = np.concatenate([fx, base_score_extra_features(base_score)], axis=2)
            rank_score = fast_scores(ranker, fx, max(128, batch))
            if blend < 1.0:
                # Blend logits after query normalization so neither scorer wins just by having a larger scale.
                score = (1.0 - blend) * query_norm_score(base_score) + blend * query_norm_score(rank_score)
            else:
                score = rank_score
            score = score + candidate_freq_bonus(cand, freq, scene)
            prob = scores_to_prob(score)
            text = "".join(",".join(f"{p:.8f}" for p in row) + "\n" for row in prob)
            f.write(text.encode())
            print(f"{scene}: fast-rank wrote {min(s + batch, len(test))}/{len(test)}", flush=True)


def write_scene(zout, scene, batch, use_prop=True, use_rank=True):
    # If a second-stage ranker exists for this scene, it owns inference; otherwise use the base path.
    if use_rank and os.path.exists(f"r{scene[-1]}.pkl"):
        write_fast_scene(zout, scene, batch)
        return
    train, test = read_scene(scene)
    max_node = node_max(train, test)
    _, freq, src_freq = test_pool_freq(test, max_node)
    edges_arr = train[["src", "dst", "time"]].to_numpy(np.int64)
    stats = Stats(edges_arr, max_node)
    cf = load_cf_emb(scene)
    alpha = PROP_ALPHA.get(scene, 0.0) if use_prop else 0.0
    # Ensemble members (m{d}_ens*.pkl) or the single m{d}.pkl; each contributes a
    # query-normalized score, averaged (teammate's real-validated 1.4331 recipe).
    members = []
    for mp in _ensemble_paths(scene):
        net, mu, sd = load_model(mp)
        feat_fn, use_src_freq = feat_fn_for_dim(len(mu))
        prop = None
        if alpha:
            state = net.state_dict()
            src_emb = np.asarray(state["src_emb.weight"].data, dtype=np.float32)
            dst_emb = np.asarray(state["dst_emb.weight"].data, dtype=np.float32)
            prop_src, prop_dst = make_prop_embeddings(edges_arr, int(max_node) + 1, src_emb, dst_emb)
            prop = (src_emb, dst_emb, prop_src, prop_dst)
        members.append((net, mu, sd, feat_fn, use_src_freq, prop))
    with zout.open(f"{scene}.csv", "w") as f:
        for s in range(0, len(test), batch):
            part = test.iloc[s : s + batch]
            src = part.src.to_numpy(np.int64, copy=False)
            tim = part.time.to_numpy(np.int64, copy=False)
            cand = part.iloc[:, 2:].to_numpy(np.int64, copy=False)
            hist_ids, hist_mask = history_ids(stats, src)
            acc = None
            for net, mu, sd, feat_fn, use_src_freq, prop in members:
                x = scale(feat_fn(stats, src, tim, cand, freq, src_freq, use_src_freq), mu, sd)
                sc = model_scores_with_prop(net, x, src, cand, hist_ids, hist_mask, max(128, batch // 4), prop, alpha)
                sc = query_norm_score(sc)
                acc = sc if acc is None else acc + sc
            score = acc / len(members)
            score = score + candidate_cf_bonus(src, cand, cf, scene)
            score = score + candidate_freq_bonus(cand, freq, scene)
            prob = scores_to_prob(score)
            text = "".join(",".join(f"{p:.8f}" for p in row) + "\n" for row in prob)
            f.write(text.encode())
            print(f"{scene}: wrote {min(s + batch, len(test))}/{len(test)} (ensemble={len(members)})", flush=True)


def submit(batch, use_prop=True, use_rank=True):
    with zipfile.ZipFile("result.zip", "w", zipfile.ZIP_DEFLATED) as zout:
        write_scene(zout, "dataset1", batch, use_prop, use_rank)
        write_scene(zout, "dataset2", batch, use_prop, use_rank)


def check():
    print("jittor", jt.__version__, "cuda", bool(jt.has_cuda and jt.flags.use_cuda))
    with zipfile.ZipFile(DATA) as z:
        for name in z.namelist():
            print(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["all", "train", "rank", "submit", "check"], nargs="?", default="all")
    p.add_argument("--groups1", type=int, default=40000)
    p.add_argument("--groups2", type=int, default=70000)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260704)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--hist", action="store_true")
    p.add_argument("--srcfreq2", action="store_true")
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=["ce", "pair", "mix"], default="ce")
    p.add_argument("--valid1", type=int, default=0)
    p.add_argument("--valid2", type=int, default=0)
    p.add_argument("--rank-scene", choices=["dataset1", "dataset2", "both"], default="dataset2")
    p.add_argument("--rank-groups", type=int, default=120000)
    p.add_argument("--rank-valid", type=int, default=50000)
    p.add_argument("--rank-epochs", type=int, default=20)
    p.add_argument("--no-prop", action="store_true")
    p.add_argument("--no-rank", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.groups1 = 512
        args.groups2 = 512
        args.valid1 = 512
        args.valid2 = 512
        args.epochs = 1
        args.batch = 128

    if args.mode == "check":
        check()
        return
    if args.mode in ("all", "train"):
        s1 = fit_scene(
            "dataset1",
            args.groups1,
            args.epochs,
            args.batch,
            args.seed + 1,
            args.hist,
            args.srcfreq2,
            args.finetune,
            args.lr,
            args.loss,
            args.valid1,
        )
        s2 = fit_scene(
            "dataset2",
            args.groups2,
            args.epochs,
            args.batch,
            args.seed + 2,
            args.hist,
            args.srcfreq2,
            args.finetune,
            args.lr,
            args.loss,
            args.valid2,
        )
        print(f"valid dataset1={s1:.6f} dataset2={s2:.6f} sum={s1 + s2:.6f}", flush=True)
    if args.mode == "rank":
        scenes = ["dataset1", "dataset2"] if args.rank_scene == "both" else [args.rank_scene]
        for scene in scenes:
            train_fast_scene(
                scene,
                args.rank_groups,
                args.rank_valid,
                args.rank_epochs,
                args.batch,
                args.seed + (11 if scene == "dataset1" else 12),
            )
    if args.mode in ("all", "submit"):
        submit(args.batch, not args.no_prop, not args.no_rank)


if __name__ == "__main__":
    main()
