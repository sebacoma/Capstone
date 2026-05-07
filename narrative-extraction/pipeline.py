"""Main pipeline for narrative extraction experiments."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from functools import partial

from dotenv import load_dotenv

load_dotenv()

from baselines import maps, trails
from evaluation.judge import evaluate_narrative
from llm.openai_scorer import score_coherence as openai_score_coherence
from llm.scorer import score_coherence as llm_score_coherence
from metrics.recorder import MetricsRecorder, avg_edge_coherence
from scoring.algorithmic import score_coherence as alg_score_coherence

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METHODS = [
    "maps_algorithmic", "maps_llm", "maps_proprietary",
    "trails_algorithmic", "trails_llm", "trails_proprietary",
]


def load_documents(path: str) -> list[dict]:
    """Load documents from a JSONL file."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    logger.info("Loaded %d documents from %s", len(docs), path)
    return docs


def run(args: argparse.Namespace) -> dict | None:
    """Execute the narrative extraction pipeline. Returns result dict."""
    docs = load_documents(args.input)

    # Select scorer
    method_parts = args.method.split("_", 1)
    algo_name, scorer_name = method_parts[0], method_parts[1]

    if scorer_name == "llm":
        scorer_fn = partial(llm_score_coherence, model=args.model)
    elif scorer_name == "proprietary":
        scorer_fn = partial(openai_score_coherence, model=args.openai_model)
    else:
        scorer_fn = alg_score_coherence

    # Select algorithm
    if algo_name == "maps":
        extract_fn = maps.extract_narrative
    else:
        extract_fn = trails.extract_narrative

    # Build kwargs for extraction
    extract_kwargs: dict = {"n": args.n}
    if args.window is not None:
        extract_kwargs["window_days"] = args.window
    if hasattr(args, "start") and args.start is not None:
        extract_kwargs["start_index"] = args.start
    if hasattr(args, "end") and args.end is not None:
        extract_kwargs["end_index"] = args.end

    # Run with metrics
    with MetricsRecorder() as recorder:
        narrative = extract_fn(docs, scorer_fn, **extract_kwargs)

    if not narrative.documents:
        logger.error("No narrative extracted. Check logs for details.")
        return None

    # Print narrative
    print(f"\n{'='*60}")
    print(f"Narrative ({args.method}, n={args.n})")
    print(f"{'='*60}")
    for i, doc in enumerate(narrative.documents):
        print(f"  {i+1}. [{doc['date']}] {doc['id']}: {doc['title']}")
    print(f"{'='*60}\n")

    # Compute edge coherence
    edge_coh = avg_edge_coherence(narrative.documents, scorer_fn)
    logger.info("Average edge coherence: %.4f", edge_coh)

    # LLM judge evaluation
    judge_result: dict = {}
    if not args.skip_judge:
        judge_model = args.judge_model or args.model
        logger.info("Running LLM judge evaluation (model=%s)...", judge_model)
        try:
            judge_result = evaluate_narrative(narrative.documents, judge_model=judge_model)
            logger.info("Judge coherence score: %s/10", judge_result.get("coherence_score"))
            logger.info("Justification: %s", judge_result.get("justification"))
        except Exception as e:
            logger.warning("Judge evaluation failed: %s", e)
            judge_result = {"error": str(e)}
    else:
        logger.info("Judge evaluation skipped (--skip-judge)")

    # Determine model name for result metadata
    if scorer_name == "proprietary":
        model_name = args.openai_model
    elif scorer_name == "llm":
        model_name = args.model
    else:
        model_name = None

    # Build result
    start_idx = getattr(args, "start", None)
    end_idx = getattr(args, "end", None)
    result = {
        "method": args.method,
        "model": model_name,
        "n": args.n,
        "window_days": args.window,
        "start_index": start_idx,
        "end_index": end_idx,
        "narrative": [
            {"id": d["id"], "date": d["date"], "title": d["title"]}
            for d in narrative.documents
        ],
        "metrics": {
            **recorder.results,
            "avg_edge_coherence": edge_coh,
        },
        "judge": judge_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Add graph structure for maps methods
    if algo_name == "maps" and narrative.edges:
        result["graph"] = {
            "nodes": [
                {"id": d["id"], "date": d["date"], "title": d["title"]}
                for d in narrative.documents
            ],
            "edges": [
                {
                    "source": src["id"],
                    "target": tgt["id"],
                    "coherence_score": round(score, 6),
                }
                for src, tgt, score in narrative.edges
            ],
        }

    # Save
    os.makedirs(os.path.dirname(args.output) or "results", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", args.output)

    return result


def main() -> None:
    """CLI entry point."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Narrative Extraction Pipeline")
    parser.add_argument("--input", default="data/sample.jsonl",
                        help="Path to JSONL input file")
    parser.add_argument("--method", choices=METHODS, required=True,
                        help="Extraction method")
    parser.add_argument("--n", type=int, default=6,
                        help="Narrative length (number of documents)")
    parser.add_argument("--model", default="llama3.2",
                        help="Ollama model for LLM methods")
    parser.add_argument("--openai-model", default="gpt-4o-mini",
                        help="OpenAI model for proprietary methods (default: gpt-4o-mini)")
    parser.add_argument("--judge-model", default=None,
                        help="Ollama model for judge (default: same as --model)")
    parser.add_argument("--window", type=int, default=None,
                        help="Temporal window in days for edge filtering")
    parser.add_argument("--start", type=int, default=None,
                        help="Start node index in date-sorted docs (default: 0 = earliest)")
    parser.add_argument("--end", type=int, default=None,
                        help="End node index in date-sorted docs (default: None = free)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge evaluation")
    parser.add_argument("--output", default=f"results/run_{timestamp}.json",
                        help="Output path")
    args = parser.parse_args()

    result = run(args)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
