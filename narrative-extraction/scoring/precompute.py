"""Precompute and cache coherence matrices for reuse across experiments."""

import json
import logging
import os
from datetime import datetime
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def _parse_date(doc: dict) -> datetime:
    return datetime.fromisoformat(doc["date"])


def precompute_matrix(
    docs: list[dict],
    scorer_fn: Callable[[dict, dict], float],
    window_days: int | None = None,
    cache_path: str = "cache/coherence.npz",
) -> None:
    """Compute pairwise coherence matrix and save to disk.

    Args:
        docs: List of documents (must be sorted by date).
        scorer_fn: Coherence scoring function (doc_a, doc_b) -> float.
        window_days: If set, only score pairs within this temporal window.
        cache_path: Path to save the .npz cache file.
    """
    n = len(docs)
    matrix = np.zeros((n, n), dtype=np.float32)
    ids = [d["id"] for d in docs]

    total = 0
    computed = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total += 1
            if window_days is not None:
                delta = abs((_parse_date(docs[i]) - _parse_date(docs[j])).days)
                if delta > window_days:
                    continue
            matrix[i, j] = scorer_fn(docs[i], docs[j])
            computed += 1
        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d docs scored (%d pairs computed)", i + 1, n, computed)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(cache_path, matrix=matrix, ids=np.array(ids))
    logger.info("Cache saved to %s (%d pairs, %.1f MB)",
                cache_path, computed, os.path.getsize(cache_path) / 1e6)


def load_cached_scorer(cache_path: str) -> Callable[[dict, dict], float]:
    """Load a cached coherence matrix and return a scorer function.

    Args:
        cache_path: Path to the .npz cache file.

    Returns:
        A function (doc_a, doc_b) -> float that looks up scores from cache.
    """
    data = np.load(cache_path, allow_pickle=False)
    matrix = data["matrix"]
    ids = list(data["ids"])
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}

    def cached_scorer(doc_a: dict, doc_b: dict) -> float:
        i = id_to_idx.get(doc_a["id"])
        j = id_to_idx.get(doc_b["id"])
        if i is None or j is None:
            return 0.0
        return float(matrix[i, j])

    return cached_scorer
