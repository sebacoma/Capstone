"""Statistical analysis of narrative extraction experiment results."""

import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all result JSON files into a DataFrame."""
    records = []
    for root, _dirs, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith(".json") or fname == "manifest.json":
                continue
            path = os.path.join(root, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            record = {
                "method": data.get("method"),
                "model": data.get("model"),
                "n": data.get("n"),
                "avg_edge_coherence": data.get("metrics", {}).get("avg_edge_coherence"),
                "wall_time_seconds": data.get("metrics", {}).get("wall_time_seconds"),
                "peak_memory_mb": data.get("metrics", {}).get("peak_memory_mb"),
                "judge_score": data.get("judge", {}).get("coherence_score"),
                "file": path,
            }
            records.append(record)

    df = pd.DataFrame(records)
    logger.info("Loaded %d result files from %s", len(df), results_dir)
    return df


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics grouped by method, model, and n."""
    metrics = ["avg_edge_coherence", "wall_time_seconds", "peak_memory_mb", "judge_score"]

    # Group key depends on whether model is present
    group_cols = ["method", "model", "n"]
    grouped = df.groupby(group_cols, dropna=False)

    stats = []
    for name, group in grouped:
        row = {"method": name[0], "model": name[1], "n": name[2], "count": len(group)}
        for metric in metrics:
            values = group[metric].dropna()
            if len(values) > 0:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std()
                row[f"{metric}_median"] = values.median()
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_median"] = None
        stats.append(row)

    return pd.DataFrame(stats)


def pairwise_tests(df: pd.DataFrame, metric: str = "avg_edge_coherence") -> pd.DataFrame:
    """Run pairwise t-tests with Bonferroni correction between methods."""
    methods = df["method"].dropna().unique()
    n_comparisons = len(list(combinations(methods, 2)))
    if n_comparisons == 0:
        return pd.DataFrame()

    results = []
    for m1, m2 in combinations(sorted(methods), 2):
        g1 = df[df["method"] == m1][metric].dropna().values
        g2 = df[df["method"] == m2][metric].dropna().values

        if len(g1) < 2 or len(g2) < 2:
            continue

        t_stat, p_value = ttest_ind(g1, g2, equal_var=False)
        d = cohens_d(g1, g2)
        p_bonferroni = min(p_value * n_comparisons, 1.0)

        results.append({
            "method_a": m1,
            "method_b": m2,
            f"mean_a ({metric})": np.mean(g1),
            f"mean_b ({metric})": np.mean(g2),
            "t_statistic": t_stat,
            "p_value": p_value,
            "p_bonferroni": p_bonferroni,
            "cohens_d": d,
            "significant_0.05": p_bonferroni < 0.05,
            "n_a": len(g1),
            "n_b": len(g2),
        })

    return pd.DataFrame(results)


def print_markdown_table(df: pd.DataFrame, title: str = "") -> None:
    """Print a DataFrame as a markdown table."""
    if title:
        print(f"\n## {title}\n")
    if df.empty:
        print("No data available.\n")
        return

    # Format floats
    formatted = df.copy()
    for col in formatted.select_dtypes(include=[np.floating]).columns:
        formatted[col] = formatted[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

    print(formatted.to_markdown(index=False))
    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze narrative extraction results")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing result JSON files")
    parser.add_argument("--output-dir", default="analysis",
                        help="Directory for output CSV files")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        logger.error("No results found in %s", args.results_dir)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Descriptive statistics
    summary = descriptive_stats(df)
    summary_path = os.path.join(args.output_dir, "summary_table.csv")
    summary.to_csv(summary_path, index=False)
    print_markdown_table(summary, "Descriptive Statistics")
    logger.info("Summary saved to %s", summary_path)

    # Pairwise tests on edge coherence
    tests_coh = pairwise_tests(df, "avg_edge_coherence")
    if not tests_coh.empty:
        tests_path = os.path.join(args.output_dir, "significance_coherence.csv")
        tests_coh.to_csv(tests_path, index=False)
        print_markdown_table(tests_coh, "Pairwise t-tests (avg_edge_coherence)")
        logger.info("Coherence tests saved to %s", tests_path)

    # Pairwise tests on judge score
    tests_judge = pairwise_tests(df, "judge_score")
    if not tests_judge.empty:
        tests_path = os.path.join(args.output_dir, "significance_judge.csv")
        tests_judge.to_csv(tests_path, index=False)
        print_markdown_table(tests_judge, "Pairwise t-tests (judge_score)")
        logger.info("Judge tests saved to %s", tests_path)


if __name__ == "__main__":
    main()
