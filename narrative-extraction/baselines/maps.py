"""Narrative Maps baseline (Keith et al., 2021) — ILP-based narrative extraction."""

import logging
from datetime import datetime
from typing import Callable

import pulp

from baselines.result import NarrativeResult

logger = logging.getLogger(__name__)


def _parse_date(doc: dict) -> datetime:
    """Parse ISO date from document."""
    return datetime.fromisoformat(doc["date"])


def _days_between(doc_a: dict, doc_b: dict) -> int:
    """Compute absolute days between two documents."""
    return abs((_parse_date(doc_a) - _parse_date(doc_b)).days)


def extract_narrative(
    documents: list[dict],
    scorer_fn: Callable[[dict, dict], float],
    n: int = 6,
    window_days: int | None = None,
    time_limit: int = 120,
    start_index: int = 0,
    end_index: int | None = None,
) -> NarrativeResult:
    """Extract a narrative map of n documents using Integer Linear Programming.

    Formulates an ILP that selects a path of n documents maximizing total
    coherence, subject to flow constraints (valid path, no cycles).
    Returns the path plus all edges between selected nodes (the map structure).

    Args:
        documents: List of documents with 'id', 'title', 'text', 'date'.
        scorer_fn: Coherence scoring function (doc_a, doc_b) -> float.
        n: Number of documents in the extracted narrative.
        window_days: If set, only consider edges within this temporal window.
        time_limit: Maximum seconds for ILP solver (default 120).
        start_index: Index of the source node in the date-sorted document list (default 0).
        end_index: If set, index of the required end node in the date-sorted list.

    Returns:
        NarrativeResult with the narrative path and all inter-node edges.
    """
    if len(documents) < n:
        logger.warning("Not enough documents (%d) for narrative of size %d", len(documents), n)
        return NarrativeResult(documents=[], edges=[])

    # Sort by date, source node = earliest
    docs = sorted(documents, key=_parse_date)
    num_docs = len(docs)

    # Precompute coherence for valid edges only
    logger.info("Computing coherence matrix for %d documents (window=%s days)...",
                num_docs, window_days)
    coherence: dict[tuple[int, int], float] = {}
    edges: list[tuple[int, int]] = []
    computed = 0
    skipped = 0
    for i in range(num_docs):
        for j in range(num_docs):
            if i == j:
                continue
            if window_days is not None and _days_between(docs[i], docs[j]) > window_days:
                skipped += 1
                continue
            coherence[i, j] = scorer_fn(docs[i], docs[j])
            edges.append((i, j))
            computed += 1

    logger.info("Coherence computed: %d edges, %d skipped", computed, skipped)

    # ILP formulation
    prob = pulp.LpProblem("NarrativeMaps", pulp.LpMaximize)

    # Binary edge variables: x[i,j] = 1 if edge i->j is in the narrative
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary) for i, j in edges}

    # Binary node variables: y[i] = 1 if node i is in the narrative
    y = [pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary) for i in range(num_docs)]

    # Ordering variables for subtour elimination (MTZ formulation)
    u = [pulp.LpVariable(f"u_{i}", lowBound=0, upBound=n - 1, cat=pulp.LpContinuous)
         for i in range(num_docs)]

    # Objective: maximize total coherence along selected edges
    prob += pulp.lpSum(coherence[i, j] * x[i, j] for i, j in edges)

    # Constraint 1: exactly n nodes selected
    prob += pulp.lpSum(y) == n

    # Validate start/end indices
    src = start_index
    if src < 0 or src >= num_docs:
        logger.warning("start_index %d out of range (0-%d), using 0", src, num_docs - 1)
        src = 0

    dst = end_index
    if dst is not None and (dst < 0 or dst >= num_docs or dst == src):
        logger.warning("end_index %d invalid, ignoring", dst)
        dst = None

    # Constraint 2: source node must be in the narrative
    prob += y[src] == 1

    # Constraint 2b: end node must be in the narrative (if specified)
    if dst is not None:
        prob += y[dst] == 1

    # Constraint 3: each selected node (except source) has exactly one predecessor
    for j in range(num_docs):
        if j == src:
            continue
        incoming = [x[i, j] for i in range(num_docs) if (i, j) in x]
        if incoming:
            prob += pulp.lpSum(incoming) == y[j]
        else:
            # No incoming edges possible → node cannot be selected
            prob += y[j] == 0

    # Constraint 4: each selected node has at most one successor
    for i in range(num_docs):
        outgoing = [x[i, j] for j in range(num_docs) if (i, j) in x]
        if outgoing:
            prob += pulp.lpSum(outgoing) <= y[i]

    # Constraint 5: total edges = n - 1 (path of n nodes)
    prob += pulp.lpSum(x[i, j] for i, j in edges) == n - 1

    # Constraint 6: source node has exactly one outgoing edge
    source_out = [x[src, j] for j in range(num_docs) if (src, j) in x]
    if source_out:
        prob += pulp.lpSum(source_out) == 1

    # Constraint 6b: end node has no outgoing edges (sink)
    if dst is not None:
        end_out = [x[dst, j] for j in range(num_docs) if (dst, j) in x]
        if end_out:
            prob += pulp.lpSum(end_out) == 0

    # Constraint 7: MTZ subtour elimination
    prob += u[src] == 0
    for i, j in edges:
        if j != src:
            prob += u[j] >= u[i] + 1 - num_docs * (1 - x[i, j])

    # Solve with time limit
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    if prob.status not in (pulp.constants.LpStatusOptimal, 1):
        logger.warning("ILP solver did not find a solution (status=%s)", prob.status)
        return NarrativeResult(documents=[], edges=[])

    # Reconstruct path from source
    path_edges: dict[int, int] = {}
    for i, j in edges:
        val = pulp.value(x[i, j])
        if val is not None and val > 0.5:
            path_edges[i] = j

    # Walk the path from source
    path = [src]
    current = src
    for _ in range(n - 1):
        if current not in path_edges:
            break
        current = path_edges[current]
        path.append(current)

    # Collect ALL edges between selected nodes (the map structure)
    selected_set = set(path)
    all_edges = []
    for (i, j), score in coherence.items():
        if i in selected_set and j in selected_set:
            all_edges.append((docs[i], docs[j], score))

    path_docs = [docs[i] for i in path]
    return NarrativeResult(documents=path_docs, edges=all_edges)
