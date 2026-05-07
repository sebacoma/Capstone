"""Narrative Trails baseline (German et al., 2025) — max-capacity path extraction."""

import heapq
import logging
from collections import defaultdict
from datetime import datetime
from typing import Callable

import networkx as nx

from baselines.result import NarrativeResult

logger = logging.getLogger(__name__)


def _parse_date(doc: dict) -> datetime:
    """Parse ISO date from document."""
    return datetime.fromisoformat(doc["date"])


def _days_between(doc_a: dict, doc_b: dict) -> int:
    """Compute absolute days between two documents."""
    return abs((_parse_date(doc_a) - _parse_date(doc_b)).days)


def _build_graph(
    docs: list[dict],
    scorer_fn: Callable[[dict, dict], float],
    window_days: int | None = None,
) -> nx.DiGraph:
    """Build directed coherence graph, optionally filtered by temporal window."""
    num_docs = len(docs)
    G = nx.DiGraph()
    G.add_nodes_from(range(num_docs))

    computed = 0
    skipped = 0
    for i in range(num_docs):
        for j in range(num_docs):
            if i == j:
                continue
            if window_days is not None and _days_between(docs[i], docs[j]) > window_days:
                skipped += 1
                continue
            weight = scorer_fn(docs[i], docs[j])
            G.add_edge(i, j, weight=weight)
            computed += 1

    logger.info("Graph built: %d edges computed, %d skipped (window=%s days)",
                computed, skipped, window_days)
    return G


def _maximin_bounded(G: nx.DiGraph, source: int, n: int) -> list[int]:
    """Find the n-node path from source that maximizes the minimum edge weight.

    Uses a bounded-depth MaxiMin Dijkstra variant (German et al., 2025):
    priority queue keyed by (-bottleneck, node, depth), with pruning via
    best_cap[node][depth] to avoid exploring suboptimal states.

    Args:
        G: Directed graph with 'weight' edge attribute.
        source: Starting node index.
        n: Desired path length (number of nodes).

    Returns:
        List of node indices forming the best bottleneck path, or [].
    """
    if n == 1:
        return [source]

    # State: (-min_capacity, node, depth, path_as_tuple)
    heap: list[tuple[float, int, int, tuple[int, ...]]] = [
        (-float("inf"), source, 0, (source,))
    ]
    # best_cap[node][depth] = best bottleneck found to reach node at exactly depth hops
    best_cap: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(lambda: -float("inf")))
    best_cap[source][0] = float("inf")

    best_result: tuple[int, ...] | None = None
    best_bottleneck = -float("inf")

    while heap:
        neg_cap, node, depth, path = heapq.heappop(heap)
        min_cap = -neg_cap

        # Reached target depth
        if depth == n - 1:
            if min_cap > best_bottleneck:
                best_bottleneck = min_cap
                best_result = path
            continue

        # Prune: a better path to this (node, depth) was already found
        if min_cap < best_cap[node][depth]:
            continue

        path_set = set(path)
        for neighbor in G.successors(node):
            if neighbor in path_set:
                continue
            edge_cap = G[node][neighbor]["weight"]
            new_min = min(min_cap, edge_cap)
            new_depth = depth + 1
            if new_min > best_cap[neighbor][new_depth]:
                best_cap[neighbor][new_depth] = new_min
                heapq.heappush(heap, (-new_min, neighbor, new_depth, path + (neighbor,)))

    return list(best_result) if best_result else []


def _greedy_path(G: nx.DiGraph, source: int, n: int) -> list[int]:
    """Greedy fallback: at each step pick the unvisited neighbor with highest weight."""
    path = [source]
    visited = {source}

    for _ in range(n - 1):
        current = path[-1]
        best_neighbor = -1
        best_weight = -1.0

        for neighbor in G.successors(current):
            if neighbor not in visited:
                w = G[current][neighbor]["weight"]
                if w > best_weight:
                    best_weight = w
                    best_neighbor = neighbor

        if best_neighbor == -1:
            logger.warning("No unvisited neighbors at step %d, path truncated", len(path))
            break

        path.append(best_neighbor)
        visited.add(best_neighbor)

    return path


def extract_narrative(
    documents: list[dict],
    scorer_fn: Callable[[dict, dict], float],
    n: int = 6,
    window_days: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
) -> NarrativeResult:
    """Extract a narrative trail of n documents using max-capacity path optimization.

    Implements the MaxiMin Dijkstra algorithm from German et al. (2025), finding
    the path of exactly n nodes that maximizes the minimum edge weight (bottleneck).
    Falls back to greedy if MaxiMin yields no result.

    Args:
        documents: List of documents with 'id', 'title', 'text', 'date'.
        scorer_fn: Coherence scoring function (doc_a, doc_b) -> float.
        n: Number of documents in the extracted narrative.
        window_days: If set, only consider edges within this temporal window.
        start_index: Index of the source node in the date-sorted document list (default 0).
        end_index: Reserved for future use (not yet implemented for trails).

    Returns:
        NarrativeResult with the narrative path and consecutive edges.
    """
    if len(documents) < n:
        logger.warning("Not enough documents (%d) for narrative of size %d", len(documents), n)
        return NarrativeResult(documents=[], edges=[])

    # Sort by date
    docs = sorted(documents, key=_parse_date)
    num_docs = len(docs)

    # Validate start_index
    src = start_index
    if src < 0 or src >= num_docs:
        logger.warning("start_index %d out of range (0-%d), using 0", src, num_docs - 1)
        src = 0

    logger.info("Building coherence graph for %d documents (start=%d)...", len(docs), src)
    G = _build_graph(docs, scorer_fn, window_days)

    # Try MaxiMin first
    path = _maximin_bounded(G, source=src, n=n)
    if len(path) == n:
        logger.info("MaxiMin path found (bottleneck=%.4f)",
                     min(G[path[i]][path[i+1]]["weight"] for i in range(len(path)-1)))
    else:
        logger.warning("MaxiMin did not find a complete path, falling back to greedy")
        path = _greedy_path(G, source=src, n=n)

    path_docs = [docs[i] for i in path]
    trail_edges = [
        (docs[path[k]], docs[path[k + 1]], G[path[k]][path[k + 1]]["weight"])
        for k in range(len(path) - 1)
    ]
    return NarrativeResult(documents=path_docs, edges=trail_edges)
