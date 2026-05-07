"""LLM-based coherence scorer using Ollama."""

import re
import ollama


class OllamaConnectionError(Exception):
    """Raised when Ollama server is not reachable."""
    pass


def score_coherence(doc_a: dict, doc_b: dict, model: str = "llama3.2") -> float:
    """Score narrative coherence between two documents using an LLM.

    Args:
        doc_a: First document with 'title' and 'text' fields.
        doc_b: Second document with 'title' and 'text' fields.
        model: Ollama model name.

    Returns:
        Coherence score in [0.0, 1.0].

    Raises:
        OllamaConnectionError: If Ollama is not running.
    """
    prompt = (
        "You are evaluating narrative coherence between two news documents.\n\n"
        f"DOCUMENT A:\nTitle: {doc_a['title']}\nText: {doc_a['text']}\n\n"
        f"DOCUMENT B:\nTitle: {doc_b['title']}\nText: {doc_b['text']}\n\n"
        "On a scale from 0.0 to 1.0, how well does Document B continue the narrative "
        "of Document A? Consider temporal progression, causal connections, and thematic "
        "coherence.\n\n"
        "Respond with ONLY a single decimal number between 0.0 and 1.0. Nothing else."
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "connect" in error_msg or "refused" in error_msg or "unavailable" in error_msg:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama. Make sure Ollama is running "
                f"('ollama serve') and the model '{model}' is pulled "
                f"('ollama pull {model}')."
            ) from e
        raise

    text = response["message"]["content"].strip()

    # Try direct float parse first
    try:
        score = float(text)
        return max(0.0, min(1.0, score))
    except ValueError:
        pass

    # Regex fallback: find first float-like pattern
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))

    # If all parsing fails, return neutral score
    return 0.5
