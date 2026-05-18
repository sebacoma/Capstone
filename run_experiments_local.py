#!/usr/bin/env python3
"""
Ejecuta los experimentos de narrative extraction localmente (Mac + Ollama).
Equivalente al notebook de Colab pero como script standalone.

Uso:
    python run_experiments_local.py                  # todos los pendientes
    python run_experiments_local.py --only-llm       # solo LLM extractors
    python run_experiments_local.py --only-algorithmic  # solo algoritmicos
"""

import argparse
import heapq
import itertools
import json
import logging
import os
import pickle
import random
import re
import signal
import time
import tracemalloc
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from math import ceil, sqrt
from typing import Any, Callable

import numpy as np
import ollama
import pulp
import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
CORPUS_PATH = "narrative-extraction/data/corpus.jsonl"
RESULTS_DIR = "results_local/results_v2"
CACHE_DIR = "results_local/coherence_cache"
LLM_CACHE_DIR = "results_local/llm_extractor_cache"

METHODS = [
    "maps_algorithmic",
    "trails_algorithmic",
    "llm_extractor_llama32",
    "llm_extractor_mistral",
    "llm_extractor_phi3",
    "llm_extractor_gemma3",
    "llm_extractor_gpt4omini",
]

NARRATIVE_SIZES = [6, 12, 18]
REPS_ALGORITHMIC = 1
REPS_LLM = 5
WINDOW_DAYS = 30
START_INDICES = [0, 50, 100, 150]
JUDGE_MODEL = "llama3.2"
SKIP_JUDGE = False
MAX_DOCS = 200
MAX_TEXT_CHARS = 800

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LLM_CACHE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("pipeline")

# ============================================================
# DATA LOADING
# ============================================================
def load_documents(path: str) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    logger.info("Loaded %d documents from %s", len(docs), path)
    return docs

# ============================================================
# NARRATIVE RESULT
# ============================================================
@dataclass
class NarrativeResult:
    documents: list[dict]
    edges: list[tuple[dict, dict, float]] = field(default_factory=list)
    error_flag: bool = False
    api_cost_usd: float = 0.0

# ============================================================
# SCORING — Sentence Transformers
# ============================================================
_st_model: SentenceTransformer | None = None
_embedding_cache: dict[str, np.ndarray] = {}

def _get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model

def _get_embedding(doc: dict) -> np.ndarray:
    doc_id = doc["id"]
    if doc_id not in _embedding_cache:
        model = _get_st_model()
        _embedding_cache[doc_id] = model.encode(doc["text"], normalize_embeddings=True)
    return _embedding_cache[doc_id]

def alg_score_coherence(doc_a: dict, doc_b: dict) -> float:
    emb_a = _get_embedding(doc_a)
    emb_b = _get_embedding(doc_b)
    similarity = float(np.dot(emb_a, emb_b))
    return max(0.0, min(1.0, similarity))

# ============================================================
# LLM EXTRACTOR
# ============================================================
class OllamaConnectionError(Exception):
    pass

class OpenAIConfigError(Exception):
    pass

_openai_client: OpenAI | None = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIConfigError(
                "OPENAI_API_KEY no configurada. Exportala con:\n"
                "  export OPENAI_API_KEY='sk-...'\n"
                "O usa --only-llm para saltar GPT-4o-mini."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def _format_docs_for_prompt(documents: list[dict]) -> str:
    lines = []
    for doc in documents:
        text_truncated = doc["text"][:MAX_TEXT_CHARS]
        lines.append(
            f"ID: {doc['id']}\n"
            f"Fecha: {doc['date']}\n"
            f"Titulo: {doc['title']}\n"
            f"Texto: {text_truncated}\n"
        )
    return "\n---\n".join(lines)

def _build_extractor_prompt(documents: list[dict], n: int) -> str:
    dump = _format_docs_for_prompt(documents)
    return (
        f"Eres un experto en analisis narrativo de noticias. Te entrego un conjunto de "
        f"{len(documents)} articulos de noticias chilenos, cada uno con un identificador, "
        f"fecha, titulo y un extracto del texto. Tu tarea es seleccionar exactamente "
        f"{n} articulos que, ordenados temporalmente, conformen la narrativa mas "
        f"coherente posible: una secuencia de eventos relacionados causal o "
        f"tematicamente que cuente una historia con sentido.\n\n"
        f"ARTICULOS DISPONIBLES:\n{dump}\n\n"
        f"INSTRUCCIONES DE FORMATO:\n"
        f"- Responde SOLO con un objeto JSON, sin texto antes ni despues.\n"
        f"- No incluyas explicaciones, introducciones ni comentarios fuera del JSON.\n"
        f"- El JSON debe tener esta estructura exacta:\n\n"
        '{{"selected_ids": ["art_XXXX", "art_XXXX", ...], '
        '"justification": "una oracion"}}\n\n'
        f"REGLAS:\n"
        f"- selected_ids debe contener EXACTAMENTE {n} IDs (ni mas, ni menos).\n"
        f"- Los IDs deben ser validos (copiados exactamente de los articulos).\n"
        f"- Ordenalos cronologicamente por fecha."
    )

def _parse_extractor_response(text: str, valid_ids: set, n: int) -> dict | None:
    text = text.strip()
    ids = None
    data = {}

    # Estrategia 1: JSON
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            ids = data.get("selected_ids", [])
        except json.JSONDecodeError:
            logger.warning("JSON malformado: %s", match.group()[:200])

    # Estrategia 2: regex fallback
    if ids is None or not isinstance(ids, list) or len(ids) == 0:
        found_ids = re.findall(r"art_\d{4}", text)
        seen = set()
        ids = []
        for fid in found_ids:
            if fid not in seen:
                seen.add(fid)
                ids.append(fid)
        if ids:
            logger.info("IDs extraidos via regex fallback: %d encontrados", len(ids))
            data = {"selected_ids": ids, "justification": "extraido via regex fallback"}
        else:
            logger.warning("No se encontro JSON ni IDs en la respuesta")
            return None

    ids = [i for i in ids if i in valid_ids]
    if len(ids) < n:
        logger.warning("Solo %d IDs validos de %d requeridos", len(ids), n)
        return None
    if len(ids) > n:
        logger.info("Truncando %d IDs a %d", len(ids), n)
        ids = ids[:n]

    data["selected_ids"] = ids
    return data

def _call_llm(prompt: str, model: str, temperature: float,
              backend: str = "ollama", timeout_seconds: int = 180) -> tuple[str, float]:
    """Llamar al LLM. Devuelve (respuesta, costo_usd)."""
    if backend == "openai":
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text_out = response.choices[0].message.content
        # Estimar costo (gpt-4o-mini: $0.15/1M input, $0.60/1M output)
        tokens_in = len(prompt) / 4
        tokens_out = len(text_out) / 4 if text_out else 0
        cost = tokens_in * 0.15e-6 + tokens_out * 0.60e-6
        return text_out, cost

    # backend == "ollama"
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Ollama no respondio en {timeout_seconds}s")
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": 2048},
        )
    except TimeoutError:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "connect" in error_msg or "refused" in error_msg:
            raise OllamaConnectionError(f"Cannot connect to Ollama: {e}") from e
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return response["message"]["content"], 0.0

def _get_llm_cache_path(model: str, n: int, start_index: int, seed: int | None) -> str:
    model_safe = model.replace("-", "_").replace(".", "_")
    seed_str = f"_seed{seed}" if seed is not None else ""
    return os.path.join(LLM_CACHE_DIR, f"{model_safe}_n{n}_start{start_index}{seed_str}.json")

def run_llm_extractor(
    documents: list[dict], n: int, model: str, backend: str = "ollama",
    seed: int | None = None, start_index: int = 0,
) -> NarrativeResult:
    cache_path = _get_llm_cache_path(model, n, start_index, seed)
    if os.path.exists(cache_path):
        logger.info("Cache hit: %s", cache_path)
        with open(cache_path) as f:
            cached = json.load(f)
        doc_map = {d["id"]: d for d in documents}
        selected_docs = [doc_map[did] for did in cached["selected_ids"] if did in doc_map]
        if len(selected_docs) == n:
            edges = []
            for i in range(len(selected_docs) - 1):
                score = alg_score_coherence(selected_docs[i], selected_docs[i + 1])
                edges.append((selected_docs[i], selected_docs[i + 1], score))
            return NarrativeResult(
                documents=selected_docs, edges=edges,
                api_cost_usd=cached.get("api_cost_usd", 0.0),
            )
        logger.warning("Cache invalido, recalculando...")

    docs_for_prompt = list(documents)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(docs_for_prompt)

    valid_ids = {d["id"] for d in documents}
    prompt = _build_extractor_prompt(docs_for_prompt, n)
    temperatures = [0.0, 0.3, 0.5]
    total_cost = 0.0
    parsed = None

    for attempt, temp in enumerate(temperatures):
        logger.info("LLM extractor intento %d/%d (temp=%.1f, model=%s, backend=%s)",
                     attempt + 1, len(temperatures), temp, model, backend)
        try:
            response_text, cost = _call_llm(prompt, model, temp, backend=backend)
            total_cost += cost
            parsed = _parse_extractor_response(response_text, valid_ids, n)
            if parsed is not None:
                logger.info("Exitoso en intento %d", attempt + 1)
                break
            logger.warning("Respuesta invalida en intento %d: %s",
                          attempt + 1, response_text[:200])
        except Exception as e:
            logger.error("Error en intento %d: %s", attempt + 1, e)
            if isinstance(e, (OllamaConnectionError, OpenAIConfigError)):
                raise

    if parsed is None:
        logger.error("LLM extractor fallo despues de %d intentos", len(temperatures))
        return NarrativeResult(documents=[], edges=[], error_flag=True,
                              api_cost_usd=total_cost)

    doc_map = {d["id"]: d for d in documents}
    selected_docs = [doc_map[did] for did in parsed["selected_ids"]]
    selected_docs.sort(key=lambda d: d["date"])

    edges = []
    for i in range(len(selected_docs) - 1):
        score = alg_score_coherence(selected_docs[i], selected_docs[i + 1])
        edges.append((selected_docs[i], selected_docs[i + 1], score))

    cache_data = {
        "model": model, "n": n, "start_index": start_index, "seed": seed,
        "selected_ids": [d["id"] for d in selected_docs],
        "justification": parsed.get("justification", ""),
        "api_cost_usd": total_cost,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    return NarrativeResult(documents=selected_docs, edges=edges,
                          api_cost_usd=total_cost)

# ============================================================
# METRICS
# ============================================================
class MetricsRecorder:
    def __init__(self):
        self.results: dict[str, Any] = {}
    def __enter__(self):
        tracemalloc.start()
        self._start_time = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.results["wall_time_seconds"] = time.perf_counter() - self._start_time
        _, peak = tracemalloc.get_traced_memory()
        self.results["peak_memory_mb"] = peak / (1024 * 1024)
        tracemalloc.stop()

def avg_edge_coherence(narrative: list[dict], scorer_fn) -> float:
    if len(narrative) < 2:
        return 0.0
    scores = [scorer_fn(narrative[i], narrative[i + 1]) for i in range(len(narrative) - 1)]
    return sum(scores) / len(scores)

# ============================================================
# LLM JUDGE
# ============================================================
def _format_narrative(narrative: list[dict]) -> str:
    return "\n\n".join(
        f"[{i+1}] ({doc['date']}) {doc['title']}\n{doc['text']}"
        for i, doc in enumerate(narrative)
    )

def _parse_json_response(text: str) -> dict:
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
        justification = just_match.group(1) if just_match else "Parsed from truncated JSON"
        return {"coherence_score": score, "justification": justification}
    return {"parse_error": text[:300]}

def evaluate_narrative(narrative: list[dict], judge_model: str = "llama3.2") -> dict:
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
    response = ollama.chat(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": 512},
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

# ============================================================
# NARRATIVE MAPS (LP Relaxation)
# ============================================================
def _parse_date(doc: dict) -> datetime:
    return datetime.fromisoformat(doc["date"])

def _days_between(doc_a: dict, doc_b: dict) -> int:
    return abs((_parse_date(doc_a) - _parse_date(doc_b)).days)

def extract_narrative_maps(
    documents, scorer_fn, n=6, window_days=None,
    time_limit=120, start_index=0, end_index=None,
) -> NarrativeResult:
    if len(documents) < n:
        return NarrativeResult(documents=[], edges=[])

    docs = sorted(documents, key=_parse_date)
    num_docs = len(docs)

    window_i_j = {i: [] for i in range(num_docs)}
    window_j_i = {j: [] for j in range(num_docs)}
    for i in range(num_docs - 1):
        for j in range(i + 1, num_docs):
            if window_days is not None and _days_between(docs[i], docs[j]) > window_days:
                continue
            window_i_j[i].append(j)
            window_j_i[j].append(i)

    coherence = {}
    edges = []
    total_pairs = sum(len(window_i_j[i]) for i in range(num_docs))
    done = 0
    for i in range(num_docs):
        for j in window_i_j[i]:
            coherence[i, j] = scorer_fn(docs[i], docs[j])
            edges.append((i, j))
            done += 1
            if done % 500 == 0:
                print(f"    Coherence: {done}/{total_pairs} pares")

    prob = pulp.LpProblem("NarrativeMaps", pulp.LpMaximize)
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", 0, 1, pulp.LpContinuous) for i, j in edges}
    y = [pulp.LpVariable(f"y_{i}", 0, 1, pulp.LpContinuous) for i in range(num_docs)]
    minedge = pulp.LpVariable("minedge", 0, 1, pulp.LpContinuous)

    prob += minedge
    prob += pulp.lpSum(y) == n

    src = start_index if 0 <= start_index < num_docs else 0
    dst = end_index
    if dst is not None and (dst < 0 or dst >= num_docs or dst == src):
        dst = None
    has_end = dst is not None

    prob += y[src] == 1
    if has_end:
        prob += y[dst] == 1

    for j in range(num_docs):
        if j == src:
            continue
        incoming = [x[i, j] for i in window_j_i[j] if (i, j) in x]
        if incoming:
            prob += pulp.lpSum(incoming) == y[j]
        else:
            prob += y[j] == 0

    for i in range(num_docs):
        outgoing = [x[i, j] for j in window_i_j[i] if (i, j) in x]
        if outgoing:
            if has_end and i == dst:
                prob += pulp.lpSum(outgoing) == 0
            else:
                prob += pulp.lpSum(outgoing) <= y[i]

    incoming_src = [x[i, src] for i in window_j_i[src] if (i, src) in x]
    if incoming_src:
        prob += pulp.lpSum(incoming_src) == 0

    for i, j in edges:
        prob += minedge <= 1 - x[i, j] + coherence[i, j]

    prob.solve(pulp.PULP_CBC_CMD(msg=0, mip=False, timeLimit=time_limit))
    if prob.status not in (pulp.constants.LpStatusOptimal, 1):
        return NarrativeResult(documents=[], edges=[])

    threshold = 0.1 / n
    varsdict = {}
    for i in range(num_docs):
        val = pulp.value(y[i])
        varsdict[f"y_{i}"] = max(0.0, min(1.0, val)) if val is not None else 0.0
    for i, j in edges:
        val = pulp.value(x[i, j])
        varsdict[f"x_{i}_{j}"] = max(0.0, min(1.0, val)) if val is not None else 0.0

    path_edges_map = {}
    for i in range(num_docs):
        if varsdict[f"y_{i}"] <= threshold:
            continue
        best_j, best_val = None, threshold
        for j in window_i_j[i]:
            if (i, j) in x:
                val = varsdict[f"x_{i}_{j}"]
                if val > best_val:
                    best_val = val
                    best_j = j
        if best_j is not None:
            path_edges_map[i] = best_j

    path = [src]
    current = src
    visited = {src}
    for _ in range(n - 1):
        if current not in path_edges_map:
            break
        nxt = path_edges_map[current]
        if nxt in visited:
            break
        visited.add(nxt)
        path.append(nxt)
        current = nxt

    selected_set = set(path)
    all_edges = [(docs[i], docs[j], score) for (i, j), score in coherence.items()
                 if i in selected_set and j in selected_set]
    return NarrativeResult(documents=[docs[i] for i in path], edges=all_edges)

# ============================================================
# NARRATIVE TRAILS (MaxiMin Dijkstra)
# ============================================================
def _build_graph(docs, scorer_fn, window_days=None):
    num_docs = len(docs)
    G = nx.DiGraph()
    G.add_nodes_from(range(num_docs))
    edge_pairs = []
    for i in range(num_docs):
        for j in range(num_docs):
            if i == j:
                continue
            if window_days is not None and _days_between(docs[i], docs[j]) > window_days:
                continue
            edge_pairs.append((i, j))
    total = len(edge_pairs)
    for idx, (i, j) in enumerate(edge_pairs):
        weight = scorer_fn(docs[i], docs[j])
        G.add_edge(i, j, weight=weight)
        if (idx + 1) % 500 == 0:
            print(f"    Graph: {idx+1}/{total} edges")
    return G

def _maximin_bounded(G, source, n):
    if n == 1:
        return [source]
    heap = [(-float("inf"), source, 0, (source,))]
    best_cap = defaultdict(lambda: defaultdict(lambda: -float("inf")))
    best_cap[source][0] = float("inf")
    best_result = None
    best_bottleneck = -float("inf")
    while heap:
        neg_cap, node, depth, path = heapq.heappop(heap)
        min_cap = -neg_cap
        if depth == n - 1:
            if min_cap > best_bottleneck:
                best_bottleneck = min_cap
                best_result = path
            continue
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

def _greedy_path(G, source, n):
    path = [source]
    visited = {source}
    for _ in range(n - 1):
        current = path[-1]
        best_neighbor, best_weight = -1, -1.0
        for neighbor in G.successors(current):
            if neighbor not in visited:
                w = G[current][neighbor]["weight"]
                if w > best_weight:
                    best_weight = w
                    best_neighbor = neighbor
        if best_neighbor == -1:
            break
        path.append(best_neighbor)
        visited.add(best_neighbor)
    return path

def extract_narrative_trails(
    documents, scorer_fn, n=6, window_days=None,
    start_index=0, end_index=None,
) -> NarrativeResult:
    if len(documents) < n:
        return NarrativeResult(documents=[], edges=[])
    docs = sorted(documents, key=_parse_date)
    src = start_index if 0 <= start_index < len(docs) else 0
    G = _build_graph(docs, scorer_fn, window_days)
    path = _maximin_bounded(G, source=src, n=n)
    if len(path) != n:
        logger.warning("MaxiMin incompleto, fallback a greedy")
        path = _greedy_path(G, source=src, n=n)
    path_docs = [docs[i] for i in path]
    trail_edges = [
        (docs[path[k]], docs[path[k + 1]], G[path[k]][path[k + 1]]["weight"])
        for k in range(len(path) - 1)
    ]
    return NarrativeResult(documents=path_docs, edges=trail_edges)

# ============================================================
# EXPERIMENT RUNNER
# ============================================================
LLM_EXTRACTOR_CONFIG = {
    "llm_extractor_llama32":   {"model": "llama3.2",    "backend": "ollama"},
    "llm_extractor_mistral":   {"model": "mistral",     "backend": "ollama"},
    "llm_extractor_phi3":      {"model": "phi3",        "backend": "ollama"},
    "llm_extractor_gemma3":    {"model": "gemma3:4b",   "backend": "ollama"},
    "llm_extractor_gpt4omini": {"model": "gpt-4o-mini", "backend": "openai"},
}
ALGORITHMIC_METHODS = ["maps_algorithmic", "trails_algorithmic"]
EXTRACTOR_METHODS = list(LLM_EXTRACTOR_CONFIG.keys())

def subset_docs_for_experiment(all_docs, start_index, n, window_days, method):
    sorted_docs = sorted(all_docs, key=_parse_date)
    start_date = _parse_date(sorted_docs[start_index])
    max_reach = timedelta(days=(n - 1) * window_days)
    is_trails = method.startswith("trails")
    date_lo = start_date - max_reach if is_trails else start_date
    date_hi = start_date + max_reach
    subset = []
    new_start = 0
    for i, doc in enumerate(sorted_docs):
        d = _parse_date(doc)
        if date_lo <= d <= date_hi:
            if i == start_index:
                new_start = len(subset)
            subset.append(doc)
    return (subset, new_start) if subset else (sorted_docs, start_index)

def run_single_experiment(
    documents, method, n=6, window_days=30,
    judge_model="llama3.2", skip_judge=False,
    start_index=0, cached_scorer_fn=None, seed=None,
) -> dict | None:
    with MetricsRecorder() as recorder:
        if method in LLM_EXTRACTOR_CONFIG:
            cfg = LLM_EXTRACTOR_CONFIG[method]
            narrative = run_llm_extractor(
                documents=documents, n=n, model=cfg["model"],
                backend=cfg["backend"], seed=seed, start_index=start_index,
            )
            if narrative.error_flag:
                logger.error("LLM extractor fallo para %s", method)
                return None
        elif method == "maps_algorithmic":
            scorer_fn = cached_scorer_fn or alg_score_coherence
            narrative = extract_narrative_maps(
                documents, scorer_fn, n=n, window_days=window_days,
                start_index=start_index,
            )
        elif method == "trails_algorithmic":
            scorer_fn = cached_scorer_fn or alg_score_coherence
            narrative = extract_narrative_trails(
                documents, scorer_fn, n=n, window_days=window_days,
                start_index=start_index,
            )
        else:
            logger.error("Metodo desconocido: %s", method)
            return None

    if not narrative.documents:
        return None

    print(f"\n{'='*60}")
    print(f"Narrative ({method}, n={n}, start={start_index})")
    print(f"{'='*60}")
    for i, doc in enumerate(narrative.documents):
        print(f"  {i+1}. [{doc['date']}] {doc['id']}: {doc['title'][:70]}")
    print(f"{'='*60}\n")

    edge_coh = avg_edge_coherence(narrative.documents, alg_score_coherence)

    judge_result = {}
    if not skip_judge:
        try:
            judge_result = evaluate_narrative(narrative.documents, judge_model=judge_model)
            logger.info("Judge score: %s/10", judge_result.get("coherence_score"))
        except Exception as e:
            logger.warning("Judge failed: %s", e)
            judge_result = {"error": str(e)}

    model_name = LLM_EXTRACTOR_CONFIG[method]["model"] if method in LLM_EXTRACTOR_CONFIG else None

    result = {
        "method": method, "model": model_name, "n": n,
        "window_days": window_days, "start_index": start_index,
        "narrative": [{"id": d["id"], "date": d["date"], "title": d["title"]}
                      for d in narrative.documents],
        "metrics": {
            **recorder.results,
            "avg_edge_coherence": edge_coh,
            "api_cost_usd": narrative.api_cost_usd,
        },
        "judge": judge_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if method == "maps_algorithmic" and narrative.edges:
        result["graph"] = {
            "nodes": [{"id": d["id"], "date": d["date"], "title": d["title"]}
                      for d in narrative.documents],
            "edges": [{"source": s["id"], "target": t["id"],
                       "coherence_score": round(sc, 6)}
                      for s, t, sc in narrative.edges],
        }
    else:
        result["graph"] = {
            "nodes": [{"id": d["id"], "date": d["date"], "title": d["title"]}
                      for d in narrative.documents],
            "edges": [{"source": narrative.documents[i]["id"],
                       "target": narrative.documents[i+1]["id"],
                       "coherence_score": round(narrative.edges[i][2], 6)}
                      for i in range(len(narrative.edges))] if narrative.edges else [],
        }

    return result

# ============================================================
# MAIN
# ============================================================
def get_output_path(exp: dict) -> str:
    start = exp.get("start", 0)
    if start == 0:
        return os.path.join(RESULTS_DIR, exp["method"], f"n{exp['n']}", f"rep{exp['rep']}.json")
    return os.path.join(RESULTS_DIR, exp["method"], f"n{exp['n']}", f"start{start}", f"rep{exp['rep']}.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-llm", action="store_true", help="Solo LLM extractors")
    parser.add_argument("--only-algorithmic", action="store_true", help="Solo algoritmicos")
    parser.add_argument("--skip-judge", action="store_true", help="Saltar LLM judge")
    args = parser.parse_args()

    if args.skip_judge:
        global SKIP_JUDGE
        SKIP_JUDGE = True

    # Load data
    print("Cargando corpus...")
    all_documents_full = load_documents(CORPUS_PATH)
    if MAX_DOCS and len(all_documents_full) > MAX_DOCS:
        all_documents_full.sort(key=lambda d: d["date"])
        step = len(all_documents_full) / MAX_DOCS
        indices = [int(i * step) for i in range(MAX_DOCS)]
        all_documents = [all_documents_full[i] for i in indices]
        print(f"Sampleados {len(all_documents)} docs de {len(all_documents_full)}")
    else:
        all_documents = all_documents_full

    # Load ST model
    print("Cargando sentence-transformers...")
    _ = _get_st_model()

    # Pre-compute algorithmic cache
    print("Pre-computando cache de coherencia...")
    sorted_all = sorted(all_documents, key=_parse_date)
    alg_cache = {}
    alg_cache_path = os.path.join(CACHE_DIR, "algorithmic.pkl")
    if os.path.exists(alg_cache_path):
        with open(alg_cache_path, "rb") as f:
            alg_cache = pickle.load(f)
        print(f"  Cache cargado: {len(alg_cache)} pares")
    else:
        n_docs = len(sorted_all)
        computed = 0
        for i in range(n_docs):
            for j in range(n_docs):
                if i == j:
                    continue
                di, dj = _parse_date(sorted_all[i]), _parse_date(sorted_all[j])
                if abs((di - dj).days) > WINDOW_DAYS:
                    continue
                alg_cache[(sorted_all[i]["id"], sorted_all[j]["id"])] = alg_score_coherence(sorted_all[i], sorted_all[j])
                computed += 1
                if computed % 5000 == 0:
                    print(f"    {computed} pares...")
        with open(alg_cache_path, "wb") as f:
            pickle.dump(alg_cache, f)
        print(f"  Cache computado: {computed} pares")

    def alg_cached_scorer(doc_a, doc_b):
        key = (doc_a["id"], doc_b["id"])
        return alg_cache.get(key, alg_score_coherence(doc_a, doc_b))

    # Build experiment list
    methods = METHODS
    if args.only_llm:
        methods = EXTRACTOR_METHODS
    elif args.only_algorithmic:
        methods = ALGORITHMIC_METHODS

    experiments = []
    for method in methods:
        reps = REPS_ALGORITHMIC if method in ALGORITHMIC_METHODS else REPS_LLM
        for n_size, start, rep in itertools.product(
            NARRATIVE_SIZES, START_INDICES, range(1, reps + 1)
        ):
            experiments.append({"method": method, "n": n_size, "start": start, "rep": rep})

    pending = [exp for exp in experiments if not os.path.exists(get_output_path(exp))]
    print(f"\nTotal: {len(experiments)} | Completados: {len(experiments)-len(pending)} | Pendientes: {len(pending)}")

    if not pending:
        print("Nada que hacer!")
        return

    # Orden: n chico primero (mas probable que funcionen), algoritmicos primero
    method_priority = {
        "maps_algorithmic": 0, "trails_algorithmic": 1,
        "llm_extractor_gpt4omini": 2, "llm_extractor_mistral": 3,
        "llm_extractor_llama32": 4, "llm_extractor_phi3": 5,
        "llm_extractor_gemma3": 6,
    }
    pending.sort(key=lambda e: (method_priority.get(e["method"], 9), e["n"], e.get("start", 0), e["rep"]))
    start_time = time.time()

    for idx, exp in enumerate(pending):
        out_path = get_output_path(exp)
        start_idx = exp.get("start", 0)
        subset_docs, new_start = subset_docs_for_experiment(
            all_documents, start_idx, exp["n"], WINDOW_DAYS, exp["method"])

        elapsed = time.time() - start_time
        eta = f"ETA: {(elapsed/idx)*(len(pending)-idx)/60:.1f}min" if idx > 0 else "ETA: ..."
        print(f"\n[{idx+1}/{len(pending)}] {exp['method']} n={exp['n']} "
              f"start={start_idx} rep={exp['rep']} ({len(subset_docs)} docs) {eta}")

        cached_scorer = alg_cached_scorer if exp["method"] in ALGORITHMIC_METHODS else None
        try:
            result = run_single_experiment(
                documents=subset_docs, method=exp["method"], n=exp["n"],
                window_days=WINDOW_DAYS, judge_model=JUDGE_MODEL,
                skip_judge=SKIP_JUDGE, start_index=new_start,
                cached_scorer_fn=cached_scorer, seed=exp["rep"],
            )
            if result is not None:
                result["start_index"] = start_idx
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"  -> Guardado en {out_path}")
            else:
                print(f"  -> FALLO")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            import traceback
            traceback.print_exc()

    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Batch completado: {len(pending)} experimentos en {elapsed_total/60:.1f} minutos")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
