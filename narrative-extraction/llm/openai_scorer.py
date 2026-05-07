"""Proprietary LLM coherence scorer using OpenAI API."""

import os
import re

from openai import OpenAI


class OpenAIConfigError(Exception):
    """Raised when OpenAI API key is not configured."""
    pass


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy-initialize OpenAI client."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIConfigError(
                "OPENAI_API_KEY environment variable not set. "
                "Export it with: export OPENAI_API_KEY='sk-...'"
            )
        _client = OpenAI(api_key=api_key)
    return _client


def score_coherence(doc_a: dict, doc_b: dict, model: str = "gpt-4o-mini") -> float:
    """Score narrative coherence between two documents using OpenAI API.

    Args:
        doc_a: First document with 'title' and 'text' fields.
        doc_b: Second document with 'title' and 'text' fields.
        model: OpenAI model name (default: gpt-4o-mini for cost efficiency).

    Returns:
        Coherence score in [0.0, 1.0].

    Raises:
        OpenAIConfigError: If API key is not set.
    """
    client = _get_client()

    prompt = (
        "You are evaluating narrative coherence between two news documents.\n\n"
        f"DOCUMENT A:\nTitle: {doc_a['title']}\nText: {doc_a['text']}\n\n"
        f"DOCUMENT B:\nTitle: {doc_b['title']}\nText: {doc_b['text']}\n\n"
        "On a scale from 0.0 to 1.0, how well does Document B continue the narrative "
        "of Document A? Consider temporal progression, causal connections, and thematic "
        "coherence.\n\n"
        "Respond with ONLY a single decimal number between 0.0 and 1.0. Nothing else."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )

    text = response.choices[0].message.content.strip()

    # Try direct float parse
    try:
        score = float(text)
        return max(0.0, min(1.0, score))
    except ValueError:
        pass

    # Regex fallback
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))

    return 0.5
