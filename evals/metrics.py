from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(ranked: list[str], relevant: Iterable[str], k: int = 5) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0 if not ranked[:k] else 0.0
    return len(set(ranked[:k]) & relevant_set) / len(relevant_set)


def mrr_at_k(ranked: list[str], relevant: Iterable[str], k: int = 5) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0 if not ranked[:k] else 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevance: dict[str, int], k: int = 5) -> float:
    if not relevance:
        return 1.0 if not ranked[:k] else 0.0

    def dcg(values: list[int]) -> float:
        return sum(value / math.log2(index + 2) for index, value in enumerate(values))

    actual = [relevance.get(item, 0) for item in ranked[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg(actual) / ideal_score if ideal_score else 1.0


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile_value * len(ordered)) - 1))
    return ordered[index]
