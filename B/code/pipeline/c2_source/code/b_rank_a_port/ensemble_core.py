"""Shared scoring and fitted fusion utilities for B-rank D3/D4."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import run


SEGMENT_ORDER = (
    "base_train",
    "base_valid",
    "meta_train",
    "validation",
    "confirmation",
)


def qnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-6
    )


def mrr(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        scores = scores[mask]
        labels = labels[mask]
        if not len(labels):
            return float("nan")
    positive = scores[np.arange(len(labels)), labels]
    columns = np.arange(scores.shape[1])[None, :]
    rank = 1 + (scores > positive[:, None]).sum(axis=1)
    rank += ((scores == positive[:, None]) & (columns < labels[:, None])).sum(axis=1)
    return float(np.mean(1.0 / rank))


def scene_data(data: Path, scene: str):
    run.DATA = str(data.resolve())
    train, test = run.read_scene(scene)
    max_node = run.node_max(train, test)
    use_src_freq = scene == "dataset3"
    pool, freq, src_freq = run.test_pool_freq(test, max_node, use_src_freq)
    columns = ["src", "dst", "time"]
    history = train[train["split"] == 0][columns].to_numpy(np.int64)
    target = train[train["split"] == 1][columns].to_numpy(np.int64)
    target = target[np.argsort(target[:, 2], kind="mergesort")]
    segments = run.split1_segments(target)
    return train, test, max_node, use_src_freq, pool, freq, src_freq, history, segments


def segment_history(history: np.ndarray, segments: dict[str, np.ndarray], name: str) -> np.ndarray:
    index = SEGMENT_ORDER.index(name)
    parts = [history] + [segments[key] for key in SEGMENT_ORDER[:index]]
    return np.vstack(parts)


def sample_segment(
    segments: dict[str, np.ndarray],
    pool: np.ndarray,
    name: str,
    groups: int,
    seed: int,
):
    return run.sample_groups(segments[name], pool, groups, seed)


def component_scores(
    *,
    scene: str,
    model_dirs: list[Path],
    history: np.ndarray,
    max_node: int,
    use_src_freq: bool,
    freq: np.ndarray,
    src_freq: dict[int, float],
    src: np.ndarray,
    time: np.ndarray,
    candidates: np.ndarray,
    labels: np.ndarray,
    batch: int,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray | float]]:
    stats = run.Stats(history, max_node)
    raw = run.raw_features_numba(
        stats,
        src,
        time,
        candidates,
        freq,
        src_freq,
        use_src_freq,
        return_raw=True,
    )
    heuristic_index = 23 if use_src_freq else 21
    names = ["pair_repeat", "recent_pair", "candidate_pool", "hand_score"]
    values = [
        qnorm(3.0 * raw[:, :, 0] + 2.0 * raw[:, :, 1]),
        qnorm(raw[:, :, 8] + raw[:, :, 9]),
        qnorm(raw[:, :, 11]),
        qnorm(raw[:, :, heuristic_index]),
    ]
    source_counts = stats.src_cnt[src]
    hot_threshold = float(np.quantile(source_counts, 0.8))
    strata = {
        "source_hot": source_counts >= hot_threshold,
        "pair_seen": raw[np.arange(len(labels)), labels, 0] > 0.0,
        "source_hot_threshold": hot_threshold,
    }
    del raw

    feature_cache: dict[tuple[str, bool], np.ndarray] = {}
    history_ids, history_mask = run.history_ids(stats, src)
    for model_dir in model_dirs:
        suffix = scene[-1]
        base, mu, sd = run.load_model(str(model_dir / f"m{suffix}.pkl"))
        ranker = run.load_fast_model(str(model_dir / f"r{suffix}.pkl"))
        feature_function, member_src_freq = run.feat_fn_for_dim(len(mu))
        key = (feature_function.__name__, member_src_freq)
        if key not in feature_cache:
            feature_cache[key] = feature_function(
                stats, src, time, candidates, freq, src_freq, member_src_freq
            )
        feature = run.scale(feature_cache[key], mu, sd)
        state = base.state_dict()
        src_embedding = np.asarray(state["src_emb.weight"].data, dtype=np.float32)
        dst_embedding = np.asarray(state["dst_emb.weight"].data, dtype=np.float32)
        context = run.make_fast_context(
            history, max_node, src_embedding, dst_embedding, src_freq, stats
        )
        base_score = run.model_scores_with_prop(
            base,
            feature,
            src,
            candidates,
            history_ids,
            history_mask,
            max(128, batch // 4),
            None,
            0.0,
        )
        propagation_score = run.prop_scores(
            src,
            candidates,
            src_embedding,
            dst_embedding,
            context["prop_src"],
            context["prop_dst"],
        )
        fast_feature = run.fast_rank_features(
            context, stats, src, time, candidates, freq
        )
        if fast_feature.shape[-1] != ranker.dim:
            fast_feature = np.concatenate(
                [fast_feature, run.base_score_extra_features(base_score)], axis=2
            )
        rank_score = run.fast_scores(ranker, fast_feature, max(128, batch))
        label = model_dir.name
        names.extend((f"{label}:base", f"{label}:rank", f"{label}:prop"))
        values.extend((qnorm(base_score), qnorm(rank_score), qnorm(propagation_score)))
        del context, fast_feature, feature, base_score, rank_score, propagation_score
    return names, np.stack(values, axis=0), strata


def tune_convex(components: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    individual = np.asarray([mrr(score, labels) for score in components])
    weights = np.zeros(len(components), dtype=np.float64)
    weights[int(individual.argmax())] = 1.0
    score = np.tensordot(weights, components, axes=(0, 0))
    best = mrr(score, labels)
    for step_values in (
        np.arange(0.05, 0.51, 0.05),
        np.arange(0.02, 0.21, 0.02),
        np.arange(0.01, 0.11, 0.01),
    ):
        changed = True
        while changed:
            changed = False
            for index in range(len(components)):
                for alpha in step_values:
                    candidate_score = (1.0 - alpha) * score + alpha * components[index]
                    value = mrr(candidate_score, labels)
                    if value > best + 1e-10:
                        weights *= 1.0 - alpha
                        weights[index] += alpha
                        score = candidate_score
                        best = value
                        changed = True
    weights /= weights.sum()
    return weights, best


def mixed_score(weights: np.ndarray, components: np.ndarray) -> np.ndarray:
    return np.tensordot(np.asarray(weights, dtype=np.float64), components, axes=(0, 0))
