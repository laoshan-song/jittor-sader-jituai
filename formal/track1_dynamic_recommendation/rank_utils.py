"""Ranking utilities shared by Track 1 submission generators."""

from __future__ import annotations


def rank_probabilities(scores: list[float]) -> list[float]:
    """Return a valid distribution with one distinct value per score rank."""
    count = len(scores)
    total = count * (count + 1) / 2.0
    probabilities_by_index = [0.0] * count
    for rank, index in enumerate(sorted(range(count), key=lambda value: scores[value], reverse=True)):
        probabilities_by_index[index] = (count - rank) / total
    return probabilities_by_index
