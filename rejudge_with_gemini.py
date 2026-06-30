#!/usr/bin/env python3
"""
Re-evalúa los 194 resultados existentes usando Gemini 2.5 Flash como tercer juez.

Carga results_camino2_claude_judge.json (que ya tiene LLaMA y Claude scores),
agrega judge_score_gemini, y guarda como results_camino2_three_judges.json.

Uso:
    export GEMINI_API_KEY='...'
    python rejudge_with_gemini.py
    python rejudge_with_gemini.py --test   # solo 5 experimentos
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from google import genai
from google.genai import types
from tqdm import tqdm

INPUT_JSON = "results_camino2_claude_judge.json"
OUTPUT_JSON = "results_camino2_three_judges.json"
GEMINI_MODEL = "gemini-2.5-flash"

# Gemini 2.5 Flash pricing (USD por token)
# Input: $0.075/1M tokens  |  Output: $0.30/1M tokens
GEMINI_INPUT_COST_PER_TOKEN = 0.075e-6
GEMINI_OUTPUT_COST_PER_TOKEN = 0.30e-6


def _format_narrative(narrative: list[dict]) -> str:
    return "\n\n".join(
        f"[{i+1}] ({doc['date']}) {doc['title']}\n{doc.get('doc_first_paragraph', doc.get('text', ''))}"
        for i, doc in enumerate(narrative)
    )


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    # Quitar markdown fences si los hay
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    score_match = re.search(r'"coherence_score"\s*:\s*(\d+)', text)
    if score_match:
        score = int(score_match.group(1))
        just_match = re.search(r'"justification"\s*:\s*"([^"]*)', text)
        justification = just_match.group(1) if just_match else "Parsed from truncated response"
        return {"coherence_score": score, "justification": justification}
    return {"parse_error": text[:300]}


def evaluate_narrative_gemini(
    narrative: list[dict],
    client: genai.Client,
    max_retries: int = 3,
) -> dict:
    """Evaluar narrativa con Gemini 2.5 Flash como juez. Mismo prompt que Claude."""
    narrative_text = _format_narrative(narrative)
    prompt = (
        "You are evaluating the quality of a narrative extracted from a news corpus.\n"
        "The narrative is a sequence of documents that should tell a coherent story "
        "with logical and temporal progression.\n\n"
        f"NARRATIVE ({len(narrative)} documents):\n{narrative_text}\n\n"
        "Evaluate this narrative and respond in valid JSON with exactly these fields:\n"
        '"coherence_score": integer from 0 to 10 (0 = no coherence, 10 = perfect narrative)\n'
        '"justification": a brief explanation of your score (max 2 sentences)\n\n'
        "Respond ONLY with the JSON object, no other text."
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
            text_out = response.text

            # Calcular costo desde usage_metadata si está disponible
            input_tokens = 0
            output_tokens = 0
            api_cost = 0.0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                input_tokens = getattr(um, "prompt_token_count", 0) or 0
                output_tokens = getattr(um, "candidates_token_count", 0) or 0
                api_cost = (
                    input_tokens * GEMINI_INPUT_COST_PER_TOKEN
                    + output_tokens * GEMINI_OUTPUT_COST_PER_TOKEN
                )

            result = _parse_json_response(text_out)
            if "coherence_score" not in result:
                result["coherence_score"] = -1
                result["justification"] = result.pop("parse_error", "Unknown parse failure")

            result["judge_model"] = GEMINI_MODEL
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            result["num_documents"] = len(narrative)
            result["input_tokens"] = input_tokens
            result["output_tokens"] = output_tokens
            result["api_cost_usd"] = api_cost
            return result

        except Exception as e:
            wait = 2 ** attempt  # 1, 2, 4 segundos
            err_str = str(e)
            print(f"\n  Error intento {attempt+1}/{max_retries}: {err_str[:120]} — esperando {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)

    return {
        "coherence_score": -1,
        "justification": f"Gemini API failed after {max_retries} retries",
        "judge_model": GEMINI_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_documents": len(narrative),
        "input_tokens": 0,
        "output_tokens": 0,
        "api_cost_usd": 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Solo evaluar los primeros 5 experimentos")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Exporta GEMINI_API_KEY primero")
        print("  export GEMINI_API_KEY='...'")
        sys.exit(1)

    print("=" * 60)
    print(f"RE-EVALUACIÓN CON GEMINI 2.5 FLASH{'  [MODO TEST: 5 exp]' if args.test else ''}")
    print("=" * 60)

    # Cargar input
    print(f"\nCargando {INPUT_JSON}...")
    with open(INPUT_JSON, encoding="utf-8") as f:
        all_results = json.load(f)
    print(f"  {len(all_results)} experimentos cargados")

    # Filtrar los que tienen narrativa válida
    valid_results = [r for r in all_results if r.get("narrative")]
    print(f"  {len(valid_results)} con narrativa válida")

    if args.test:
        valid_results = valid_results[:5]
        print(f"  [TEST] Evaluando solo {len(valid_results)} experimentos")

    # Cargar resultados previos para skip (resume)
    previous_results = {}
    if os.path.exists(OUTPUT_JSON) and not args.test:
        with open(OUTPUT_JSON, encoding="utf-8") as f:
            prev_data = json.load(f)
        for r in prev_data:
            key = (r.get("method"), r.get("n"), r.get("start_index", 0), r.get("_source_path"))
            sc = r.get("judge_score_gemini")
            if sc is not None and sc >= 0:
                previous_results[key] = r
        print(f"  {len(previous_results)} ya con score Gemini válido (se saltan)")

    client = genai.Client(api_key=api_key)

    total_cost = 0.0
    errors = 0
    skipped = 0
    rejudged = []
    start_time = time.time()

    print(f"\nEvaluando con Gemini 2.5 Flash...\n")

    for result in tqdm(valid_results, desc="Gemini judge"):
        method = result.get("method", "unknown")
        prev_key = (method, result.get("n"), result.get("start_index", 0), result.get("_source_path"))

        if prev_key in previous_results:
            rejudged.append(previous_results[prev_key])
            skipped += 1
            continue

        narrative = result["narrative"]
        gemini_judge = evaluate_narrative_gemini(narrative, client)

        cost = gemini_judge.get("api_cost_usd", 0.0)
        total_cost += cost

        if gemini_judge.get("coherence_score", -1) < 0:
            errors += 1

        # Construir resultado con los tres jueces
        enriched = dict(result)  # mantiene method, model, n, window_days, start_index, narrative, metrics, judge_llama, judge_score_llama, judge_claude, judge_score_claude, etc.
        enriched["judge_gemini"] = gemini_judge
        enriched["judge_score_gemini"] = gemini_judge.get("coherence_score")
        enriched["judge_gemini_justification"] = gemini_judge.get("justification", "")
        enriched["judge_gemini_cost_usd"] = cost
        enriched["timestamp_rejudge_gemini"] = gemini_judge.get("timestamp")

        rejudged.append(enriched)

        # Rate limiting suave
        time.sleep(0.5)

    elapsed = time.time() - start_time

    if args.test:
        print("\n[TEST] Resultados de los 5 experimentos:")
        for r in rejudged:
            print(f"  method={r['method']} n={r['n']} "
                  f"llama={r.get('judge_score_llama')} "
                  f"claude={r.get('judge_score_claude')} "
                  f"gemini={r.get('judge_score_gemini')} "
                  f"cost=${r.get('judge_gemini_cost_usd', 0):.5f}")
        print("\nTest OK. Si los scores se ven bien, corre sin --test para el batch completo.")
        return

    # Guardar JSON
    print(f"\nGuardando {len(rejudged)} resultados en {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rejudged, f, indent=2, ensure_ascii=False)
    print(f"  -> {OUTPUT_JSON}")

    # Resumen
    valid_gemini = [r["judge_score_gemini"] for r in rejudged
                    if r.get("judge_score_gemini") is not None and r["judge_score_gemini"] >= 0]
    valid_claude = [r["judge_score_claude"] for r in rejudged
                    if r.get("judge_score_claude") is not None and r["judge_score_claude"] >= 0]
    valid_llama = [r["judge_score_llama"] for r in rejudged
                   if r.get("judge_score_llama") is not None and r["judge_score_llama"] >= 0]

    import numpy as np

    print(f"\n{'=' * 60}")
    print("RESUMEN")
    print(f"{'=' * 60}")
    print(f"  Total evaluados:        {len(rejudged)}")
    print(f"  Saltados (ya tenían):   {skipped}")
    print(f"  Nuevos evaluados:       {len(rejudged) - skipped}")
    print(f"  Errores (score=-1):     {errors}")
    print(f"  Costo total Gemini:     ${total_cost:.4f} USD")
    print(f"  Tiempo de ejecución:    {elapsed/60:.1f} min")
    if valid_gemini:
        print(f"\n  Gemini scores: mean={np.mean(valid_gemini):.2f}, "
              f"std={np.std(valid_gemini):.2f}, "
              f"min={min(valid_gemini)}, max={max(valid_gemini)}")
    if valid_claude:
        print(f"  Claude scores: mean={np.mean(valid_claude):.2f}, "
              f"std={np.std(valid_claude):.2f}, "
              f"min={min(valid_claude)}, max={max(valid_claude)}")
    if valid_llama:
        print(f"  LLaMA scores:  mean={np.mean(valid_llama):.2f}, "
              f"std={np.std(valid_llama):.2f}, "
              f"min={min(valid_llama)}, max={max(valid_llama)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
