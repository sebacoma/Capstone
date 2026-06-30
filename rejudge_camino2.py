#!/usr/bin/env python3
"""
Re-evalúa TODOS los 194 resultados existentes (camino 1 + camino 2)
usando Claude Sonnet 4.6 como juez.

Mantiene el judge_score original de LLaMA 3.2 y agrega judge_score_claude.
Enriquece cada doc con doc_first_paragraph del corpus original.

Uso:
    export ANTHROPIC_API_KEY_PERSONAL='sk-ant-...'
    python rejudge_camino2.py
"""

import json
import os
import sys
import time

from tqdm import tqdm

# Reusar funciones del pipeline principal
from run_experiments_local import (
    evaluate_narrative_claude,
    load_documents,
    CORPUS_PATH,
    RESULTS_DIR,
)

OUTPUT_JSON = "results_camino2_claude_judge.json"
OUTPUT_CSV = "results_camino2_claude_judge.csv"


def load_corpus_map(corpus_path: str) -> dict:
    """Cargar corpus y crear mapa id -> doc completo."""
    docs = load_documents(corpus_path)
    return {d["id"]: d for d in docs}


def get_first_paragraph(text: str, max_chars: int = 500) -> str:
    """Extraer primer párrafo del texto, truncado a max_chars."""
    if not text:
        return ""
    paragraphs = text.split("\n")
    first = ""
    for p in paragraphs:
        p = p.strip()
        if p:
            first = p
            break
    if not first:
        first = text
    return first[:max_chars]


def load_all_results(results_dir: str) -> list[dict]:
    """Cargar todos los JSON de resultados."""
    results = []
    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                data["_source_path"] = fpath
                results.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  SKIP {fpath}: {e}")
    return results


def enrich_narrative(narrative: list[dict], corpus_map: dict) -> list[dict]:
    """Enriquecer docs de la narrativa con first_paragraph del corpus."""
    enriched = []
    for doc in narrative:
        doc_id = doc["id"]
        corpus_doc = corpus_map.get(doc_id, {})
        enriched_doc = dict(doc)
        enriched_doc["doc_first_paragraph"] = get_first_paragraph(
            corpus_doc.get("text", "")
        )
        enriched.append(enriched_doc)
    return enriched


def rebuild_narrative_docs(narrative_meta: list[dict], corpus_map: dict) -> list[dict]:
    """Reconstruir docs completos desde los metadatos guardados en el resultado."""
    docs = []
    for meta in narrative_meta:
        doc_id = meta["id"]
        corpus_doc = corpus_map.get(doc_id)
        if corpus_doc:
            docs.append(corpus_doc)
        else:
            # Si no está en el corpus, crear un doc mínimo
            docs.append({
                "id": doc_id,
                "date": meta.get("date", ""),
                "title": meta.get("title", ""),
                "text": meta.get("title", ""),
            })
    return docs


def main():
    if not os.environ.get("ANTHROPIC_API_KEY_PERSONAL"):
        print("ERROR: Exporta ANTHROPIC_API_KEY_PERSONAL primero")
        print("  export ANTHROPIC_API_KEY_PERSONAL='sk-ant-...'")
        sys.exit(1)

    print("=" * 60)
    print("RE-EVALUACIÓN CON CLAUDE SONNET 4.6")
    print("=" * 60)

    # Cargar corpus
    print("\nCargando corpus...")
    corpus_map = load_corpus_map(CORPUS_PATH)
    print(f"  {len(corpus_map)} documentos en corpus")

    # Cargar resultados
    print("\nCargando resultados existentes...")
    all_results = load_all_results(RESULTS_DIR)
    print(f"  {len(all_results)} resultados encontrados")

    # Filtrar exitosos (que tienen narrativa no vacía)
    valid_results = [r for r in all_results if r.get("narrative")]
    print(f"  {len(valid_results)} con narrativa válida")

    # Cargar resultados previos del rejudge (si existen) para skip
    previous_results = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            prev_data = json.load(f)
        for r in prev_data:
            key = (r.get("method"), r.get("n"), r.get("start_index", 0), r.get("_source_path"))
            sc = r.get("judge_score_claude")
            if sc is not None and sc >= 0:
                previous_results[key] = r
        print(f"  {len(previous_results)} resultados previos con score válido (se saltan)")

    # Re-evaluar
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    rejudged = []
    errors = 0
    skipped = 0

    print(f"\nRe-evaluando {len(valid_results)} narrativas con Claude...\n")

    for result in tqdm(valid_results, desc="Re-judging"):
        narrative_meta = result["narrative"]
        method = result.get("method", "unknown")

        # Skip si ya tiene score válido del run anterior
        prev_key = (method, result.get("n"), result.get("start_index", 0), result.get("_source_path"))
        if prev_key in previous_results:
            rejudged.append(previous_results[prev_key])
            skipped += 1
            continue

        # Reconstruir docs completos para el juez
        full_docs = rebuild_narrative_docs(narrative_meta, corpus_map)

        # Evaluar con Claude
        try:
            claude_judge = evaluate_narrative_claude(full_docs)
        except Exception as e:
            print(f"\n  ERROR en {method}: {e}")
            claude_judge = {
                "coherence_score": -1,
                "justification": f"Error: {e}",
                "input_tokens": 0,
                "output_tokens": 0,
                "api_cost_usd": 0.0,
            }
            errors += 1

        # Acumular costos
        cost = claude_judge.get("api_cost_usd", 0.0)
        total_cost += cost
        total_input_tokens += claude_judge.get("input_tokens", 0)
        total_output_tokens += claude_judge.get("output_tokens", 0)

        # Enriquecer narrativa con first_paragraph
        enriched_narrative = enrich_narrative(narrative_meta, corpus_map)

        # Construir resultado enriquecido
        rejudged_result = {
            "method": method,
            "model": result.get("model"),
            "n": result.get("n"),
            "window_days": result.get("window_days"),
            "start_index": result.get("start_index", 0),
            "narrative": enriched_narrative,
            "metrics": result.get("metrics", {}),
            # Judge original (LLaMA 3.2)
            "judge_llama": result.get("judge", {}),
            "judge_score_llama": result.get("judge", {}).get("coherence_score"),
            # Judge nuevo (Claude)
            "judge_claude": claude_judge,
            "judge_score_claude": claude_judge.get("coherence_score"),
            "judge_claude_justification": claude_judge.get("justification", ""),
            "judge_claude_cost_usd": cost,
            "timestamp_original": result.get("timestamp"),
            "timestamp_rejudge": claude_judge.get("timestamp"),
            "_source_path": result.get("_source_path"),
        }
        rejudged.append(rejudged_result)

        # Rate limiting: pausa entre llamadas para evitar 429
        time.sleep(1.5)

    # Guardar JSON
    print(f"\nGuardando {len(rejudged)} resultados...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rejudged, f, indent=2, ensure_ascii=False)
    print(f"  -> {OUTPUT_JSON}")

    # Guardar CSV
    try:
        import pandas as pd
        rows = []
        for r in rejudged:
            rows.append({
                "method": r["method"],
                "model": r["model"],
                "n": r["n"],
                "start_index": r["start_index"],
                "judge_score_llama": r["judge_score_llama"],
                "judge_score_claude": r["judge_score_claude"],
                "judge_claude_justification": r["judge_claude_justification"],
                "avg_edge_coherence": r["metrics"].get("avg_edge_coherence"),
                "wall_time_seconds": r["metrics"].get("wall_time_seconds"),
                "api_cost_usd_extraction": r["metrics"].get("api_cost_usd", 0.0),
                "api_cost_usd_claude_judge": r["judge_claude_cost_usd"],
                "narrative_ids": ";".join(d["id"] for d in r["narrative"]),
            })
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"  -> {OUTPUT_CSV}")
    except ImportError:
        print("  AVISO: pandas no disponible, CSV no generado")

    # Resumen
    print(f"\n{'=' * 60}")
    print("RESUMEN DE RE-EVALUACIÓN")
    print(f"{'=' * 60}")
    print(f"  Total re-evaluados: {len(rejudged)}")
    print(f"  Saltados (ya tenían score): {skipped}")
    print(f"  Nuevos evaluados: {len(rejudged) - skipped}")
    print(f"  Errores: {errors}")
    print(f"  Input tokens totales: {total_input_tokens:,}")
    print(f"  Output tokens totales: {total_output_tokens:,}")
    print(f"  Costo total Claude API: ${total_cost:.4f} USD")

    # Stats rápidas
    valid_claude = [r["judge_score_claude"] for r in rejudged
                    if r["judge_score_claude"] is not None and r["judge_score_claude"] >= 0]
    valid_llama = [r["judge_score_llama"] for r in rejudged
                   if r["judge_score_llama"] is not None and r["judge_score_llama"] >= 0]

    if valid_claude:
        import numpy as np
        print(f"\n  Claude scores: mean={np.mean(valid_claude):.2f}, "
              f"std={np.std(valid_claude):.2f}, "
              f"min={min(valid_claude)}, max={max(valid_claude)}")
    if valid_llama:
        import numpy as np
        print(f"  LLaMA scores:  mean={np.mean(valid_llama):.2f}, "
              f"std={np.std(valid_llama):.2f}, "
              f"min={min(valid_llama)}, max={max(valid_llama)}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
