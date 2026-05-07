"""Result types for narrative extraction baselines."""

from dataclasses import dataclass, field


@dataclass
class NarrativeResult:
    """Result of a narrative extraction algorithm.

    Attributes:
        documents: Ordered list of documents forming the narrative path.
        edges: List of (source_doc, target_doc, coherence_score) tuples
               representing all edges between selected nodes (for maps)
               or just the path edges (for trails).
    """
    documents: list[dict]
    edges: list[tuple[dict, dict, float]] = field(default_factory=list)
