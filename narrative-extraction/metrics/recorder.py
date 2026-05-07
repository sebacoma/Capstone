"""Metrics recording utilities for narrative extraction experiments."""

import time
import tracemalloc
from typing import Any


class MetricsRecorder:
    """Context manager that records wall time and peak memory usage."""

    def __init__(self) -> None:
        self.results: dict[str, Any] = {}

    def __enter__(self) -> "MetricsRecorder":
        tracemalloc.start()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.results["wall_time_seconds"] = time.perf_counter() - self._start_time
        _, peak = tracemalloc.get_traced_memory()
        self.results["peak_memory_mb"] = peak / (1024 * 1024)
        tracemalloc.stop()


def avg_edge_coherence(narrative: list[dict], scorer_fn: callable) -> float:
    """Compute average coherence between consecutive documents in a narrative.

    Args:
        narrative: Ordered list of documents.
        scorer_fn: Function (doc_a, doc_b) -> float.

    Returns:
        Average pairwise coherence score.
    """
    if len(narrative) < 2:
        return 0.0
    scores = [
        scorer_fn(narrative[i], narrative[i + 1])
        for i in range(len(narrative) - 1)
    ]
    return sum(scores) / len(scores)
