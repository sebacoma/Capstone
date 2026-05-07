"""Batch experiment runner for all method x model x n combinations."""

import argparse
import itertools
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import load_documents, run as pipeline_run

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ALGORITHMIC_METHODS = ["maps_algorithmic", "trails_algorithmic"]
LLM_METHODS = ["maps_llm", "trails_llm"]
PROPRIETARY_METHODS = ["maps_proprietary", "trails_proprietary"]
ALL_METHODS = ALGORITHMIC_METHODS + LLM_METHODS + PROPRIETARY_METHODS
DEFAULT_LOCAL_MODELS = ["llama3.2", "mistral", "phi3", "gemma2"]
DEFAULT_OPENAI_MODELS = ["gpt-4o-mini"]
DEFAULT_SIZES = [6, 12, 18]


def build_experiment_list(
    methods: list[str],
    local_models: list[str],
    openai_models: list[str],
    sizes: list[int],
    reps_algorithmic: int,
    reps_llm: int,
    reps_proprietary: int,
    start_indices: list[int] | None = None,
) -> list[dict]:
    """Build a list of all experiment configurations."""
    experiments = []
    starts = start_indices if start_indices else [0]

    for method in methods:
        if method in ALGORITHMIC_METHODS:
            model_list = [None]
            reps = reps_algorithmic
        elif method in LLM_METHODS:
            model_list = local_models
            reps = reps_llm
        elif method in PROPRIETARY_METHODS:
            model_list = openai_models
            reps = reps_proprietary
        else:
            continue

        for model, n, start, rep in itertools.product(model_list, sizes, starts, range(1, reps + 1)):
            experiments.append({
                "method": method,
                "model": model,
                "n": n,
                "start": start,
                "rep": rep,
            })

    return experiments


def get_output_path(results_dir: str, exp: dict) -> str:
    """Compute the output path for an experiment."""
    model_dir = exp["model"] or "algorithmic"
    start = exp.get("start", 0)
    if start == 0:
        # Backward compatible: no start prefix for default start=0
        return os.path.join(
            results_dir, exp["method"], model_dir, f"n{exp['n']}", f"rep{exp['rep']}.json"
        )
    return os.path.join(
        results_dir, exp["method"], model_dir, f"n{exp['n']}", f"start{start}", f"rep{exp['rep']}.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch narrative extraction experiments")
    parser.add_argument("--input", default="data/corpus.jsonl",
                        help="Path to JSONL input file")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, help="Methods to run")
    parser.add_argument("--models", nargs="+", default=DEFAULT_LOCAL_MODELS,
                        help="Ollama models for LLM methods")
    parser.add_argument("--openai-models", nargs="+", default=DEFAULT_OPENAI_MODELS,
                        help="OpenAI models for proprietary methods")
    parser.add_argument("--openai-model", default="gpt-4o-mini",
                        help="Default OpenAI model (used in pipeline args)")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
                        help="Narrative sizes to test")
    parser.add_argument("--reps-algorithmic", type=int, default=1,
                        help="Repetitions for algorithmic methods")
    parser.add_argument("--reps-llm", type=int, default=5,
                        help="Repetitions for LLM methods")
    parser.add_argument("--reps-proprietary", type=int, default=5,
                        help="Repetitions for proprietary methods")
    parser.add_argument("--window", type=int, default=30,
                        help="Temporal window in days")
    parser.add_argument("--judge-model", default="llama3.2",
                        help="Fixed model for LLM judge evaluation")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge evaluation")
    parser.add_argument("--start-indices", nargs="+", type=int, default=None,
                        help="Start node indices to test (e.g., 0 50 100 150)")
    parser.add_argument("--results-dir", default="results",
                        help="Base directory for results")
    parser.add_argument("--dry-run", action="store_true",
                        help="List experiments without running")
    args = parser.parse_args()

    experiments = build_experiment_list(
        methods=args.methods,
        local_models=args.models,
        openai_models=args.openai_models,
        sizes=args.sizes,
        reps_algorithmic=args.reps_algorithmic,
        reps_llm=args.reps_llm,
        reps_proprietary=args.reps_proprietary,
        start_indices=args.start_indices,
    )

    # Check which are already done
    pending = []
    done = 0
    for exp in experiments:
        out_path = get_output_path(args.results_dir, exp)
        if os.path.exists(out_path):
            done += 1
        else:
            pending.append(exp)

    total = len(experiments)
    logger.info("Experiments: %d total, %d done, %d pending", total, done, len(pending))

    if args.dry_run:
        print(f"\n{'Method':<25} {'Model':<15} {'n':>3} {'Start':>6} {'Rep':>4}  {'Status':<8}")
        print("-" * 75)
        for exp in experiments:
            out_path = get_output_path(args.results_dir, exp)
            status = "DONE" if os.path.exists(out_path) else "PENDING"
            model_str = exp["model"] or "-"
            print(f"{exp['method']:<25} {model_str:<15} {exp['n']:>3} {exp.get('start', 0):>6} {exp['rep']:>4}  {status:<8}")
        print(f"\nTotal: {total} | Done: {done} | Pending: {len(pending)}")
        return

    if not pending:
        logger.info("All experiments already completed!")
        return

    # Run pending experiments
    start_time = time.time()
    for idx, exp in enumerate(pending):
        out_path = get_output_path(args.results_dir, exp)
        model_str = exp["model"] or "algorithmic"

        # ETA
        elapsed = time.time() - start_time
        if idx > 0:
            avg_per_exp = elapsed / idx
            remaining = avg_per_exp * (len(pending) - idx)
            eta_str = f"ETA: {remaining / 60:.1f}min"
        else:
            eta_str = "ETA: calculating..."

        start_idx = exp.get("start", 0)
        logger.info("[%d/%d] %s model=%s n=%d start=%d rep=%d  %s",
                     idx + 1, len(pending), exp["method"], model_str, exp["n"],
                     start_idx, exp["rep"], eta_str)

        # Build args namespace matching pipeline.run() expectations
        is_proprietary = exp["method"] in PROPRIETARY_METHODS
        run_args = argparse.Namespace(
            input=args.input,
            method=exp["method"],
            model=exp["model"] or "llama3.2",
            openai_model=exp["model"] if is_proprietary else args.openai_model,
            judge_model=args.judge_model,
            n=exp["n"],
            start=start_idx if start_idx != 0 else None,
            end=None,
            window=args.window,
            skip_judge=args.skip_judge,
            output=out_path,
        )

        try:
            result = pipeline_run(run_args)
            if result is None:
                logger.error("Experiment failed: %s", exp)
        except Exception as e:
            logger.error("Experiment crashed: %s — %s", exp, e)

    elapsed_total = time.time() - start_time
    logger.info("Batch completed: %d experiments in %.1f minutes",
                len(pending), elapsed_total / 60)

    # Save manifest
    manifest_path = os.path.join(args.results_dir, "manifest.json")
    manifest = {
        "completed_at": datetime.now().isoformat(),
        "input": args.input,
        "window": args.window,
        "total_experiments": total,
        "experiments": [
            {**exp, "output": get_output_path(args.results_dir, exp)}
            for exp in experiments
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", manifest_path)


if __name__ == "__main__":
    main()
