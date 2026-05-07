"""LLM-based narrative quality evaluation using Ollama (Keith 2025 protocol)."""

import json
import re
from datetime import datetime, timezone

import ollama


def _format_narrative(narrative: list[dict]) -> str:
    """Format a narrative as numbered text for LLM consumption."""
    return "\n\n".join(
        f"[{i+1}] ({doc['date']}) {doc['title']}\n{doc['text']}"
        for i, doc in enumerate(narrative)
    )


def _parse_json_response(text: str) -> dict:
    """Robustly parse JSON from an LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON object from surrounding text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"parse_error": text[:300]}


def evaluate_narrative(
    narrative: list[dict],
    judge_model: str = "llama3.2",
) -> dict:
    """Pointwise evaluation: score a single narrative (Keith 2025 protocol).

    Args:
        narrative: Ordered list of documents forming the narrative.
        judge_model: Ollama model to use as judge.

    Returns:
        Dict with coherence_score (0-10), justification, and metadata.
    """
    narrative_text = _format_narrative(narrative)

    prompt = (
        "You are evaluating the quality of a narrative extracted from a news corpus.\n"
        "The narrative is a sequence of documents that should tell a coherent story "
        "with logical and temporal progression.\n\n"
        f"NARRATIVE ({len(narrative)} documents):\n{narrative_text}\n\n"
        "Evaluate this narrative and respond in valid JSON with exactly these fields:\n"
        '- "coherence_score": integer from 0 to 10 (0 = no coherence, 10 = perfect narrative)\n'
        '- "justification": a brief explanation of your score\n\n'
        "Respond ONLY with the JSON object, no other text."
    )

    response = ollama.chat(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )

    result = _parse_json_response(response["message"]["content"])
    if "coherence_score" not in result:
        result["coherence_score"] = -1
        result["justification"] = result.pop("parse_error", "Unknown parse failure")

    result["evaluation_type"] = "pointwise"
    result["judge_model"] = judge_model
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["num_documents"] = len(narrative)
    return result


def compare_narratives(
    narrative_a: list[dict],
    narrative_b: list[dict],
    method_a: str = "method_a",
    method_b: str = "method_b",
    judge_model: str = "llama3.2",
) -> dict:
    """Pairwise evaluation: compare two narratives (Keith 2025 protocol).

    Args:
        narrative_a: First narrative (ordered list of documents).
        narrative_b: Second narrative (ordered list of documents).
        method_a: Label for narrative A.
        method_b: Label for narrative B.
        judge_model: Ollama model to use as judge.

    Returns:
        Dict with preferred ("A", "B", or "tie"), justification, and metadata.
    """
    text_a = _format_narrative(narrative_a)
    text_b = _format_narrative(narrative_b)

    prompt = (
        "You are comparing two narratives extracted from the same news corpus.\n"
        "Each narrative is a sequence of documents that should tell a coherent story "
        "with logical and temporal progression.\n\n"
        f"NARRATIVE A ({len(narrative_a)} documents):\n{text_a}\n\n"
        f"NARRATIVE B ({len(narrative_b)} documents):\n{text_b}\n\n"
        "Which narrative is more coherent? Consider:\n"
        "1. Temporal progression (do events follow a logical timeline?)\n"
        "2. Causal connections (does each document logically follow from the previous?)\n"
        "3. Thematic unity (do all documents contribute to the same story?)\n\n"
        "Respond in valid JSON with exactly these fields:\n"
        '- "preferred": "A", "B", or "tie"\n'
        '- "justification": a brief explanation of your choice\n\n'
        "Respond ONLY with the JSON object, no other text."
    )

    response = ollama.chat(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )

    result = _parse_json_response(response["message"]["content"])
    if "preferred" not in result:
        result["preferred"] = "error"
        result["justification"] = result.pop("parse_error", "Unknown parse failure")

    result["evaluation_type"] = "pairwise"
    result["method_a"] = method_a
    result["method_b"] = method_b
    result["judge_model"] = judge_model
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result
