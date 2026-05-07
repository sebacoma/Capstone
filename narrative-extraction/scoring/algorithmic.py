"""Algorithmic coherence scorer using sentence-transformers embeddings."""

from sentence_transformers import SentenceTransformer
import numpy as np

_model: SentenceTransformer | None = None
_embedding_cache: dict[str, np.ndarray] = {}


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_embedding(doc: dict) -> np.ndarray:
    """Get or compute embedding for a document, caching by id."""
    doc_id = doc["id"]
    if doc_id not in _embedding_cache:
        model = _get_model()
        _embedding_cache[doc_id] = model.encode(doc["text"], normalize_embeddings=True)
    return _embedding_cache[doc_id]


def score_coherence(doc_a: dict, doc_b: dict) -> float:
    """Compute cosine similarity between two documents' embeddings.

    Args:
        doc_a: First document with 'id' and 'text' fields.
        doc_b: Second document with 'id' and 'text' fields.

    Returns:
        Cosine similarity score in [0, 1].
    """
    emb_a = _get_embedding(doc_a)
    emb_b = _get_embedding(doc_b)
    similarity = float(np.dot(emb_a, emb_b))
    return max(0.0, min(1.0, similarity))
