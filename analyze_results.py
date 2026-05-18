#!/usr/bin/env python3
"""
Análisis estadístico de los experimentos de narrative extraction.
Genera tablas, test-t de Welch, Cohen's d y visualizaciones.

Uso:
    python analyze_results.py                    # análisis completo
    python analyze_results.py --results-dir X    # directorio custom
"""

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# CARGA DE RESULTADOS
# ============================================================

def load_all_results(results_dir: str) -> pd.DataFrame:
    """Cargar todos los JSON de resultados en un DataFrame."""
    rows = []
    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Extraer rep del nombre de archivo
            rep = int(fname.replace("rep", "").replace(".json", ""))

            row = {
                "method": data.get("method", "unknown"),
                "model": data.get("model"),
                "n": data.get("n"),
                "start_index": data.get("start_index", 0),
                "rep": rep,
                "window_days": data.get("window_days"),
                # Métricas
                "judge_score": data.get("judge", {}).get("coherence_score"),
                "avg_edge_coherence": data.get("metrics", {}).get("avg_edge_coherence"),
                "wall_time_seconds": data.get("metrics", {}).get("wall_time_seconds"),
                "peak_memory_mb": data.get("metrics", {}).get("peak_memory_mb"),
                "api_cost_usd": data.get("metrics", {}).get("api_cost_usd", 0.0),
                # Meta
                "n_docs_narrative": len(data.get("narrative", [])),
                "timestamp": data.get("timestamp"),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"ERROR: No se encontraron resultados en {results_dir}")
        sys.exit(1)

    # Limpiar judge_score inválidos
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    df.loc[df["judge_score"] < 0, "judge_score"] = np.nan

    # Clasificar tipo de método
    df["method_type"] = df["method"].apply(lambda m:
        "algorithmic" if "algorithmic" in m else
        "proprietary" if "gpt" in m else
        "local_llm"
    )

    print(f"Cargados {len(df)} resultados desde {results_dir}")
    print(f"  Métodos: {sorted(df['method'].unique())}")
    print(f"  Tamaños: {sorted(df['n'].unique())}")
    print(f"  Judge scores válidos: {df['judge_score'].notna().sum()}/{len(df)}")
    return df


# ============================================================
# ESTADÍSTICAS DESCRIPTIVAS
# ============================================================

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de estadísticas descriptivas por método y n."""
    grouped = df.groupby(["method", "n"]).agg(
        judge_mean=("judge_score", "mean"),
        judge_std=("judge_score", "std"),
        judge_median=("judge_score", "median"),
        edge_coh_mean=("avg_edge_coherence", "mean"),
        edge_coh_std=("avg_edge_coherence", "std"),
        wall_time_mean=("wall_time_seconds", "mean"),
        memory_mean=("peak_memory_mb", "mean"),
        api_cost_total=("api_cost_usd", "sum"),
        count=("judge_score", "count"),
        valid_judges=("judge_score", lambda x: x.notna().sum()),
    ).round(4)
    return grouped


def summary_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen agregada por método (todos los n juntos)."""
    grouped = df.groupby("method").agg(
        judge_mean=("judge_score", "mean"),
        judge_std=("judge_score", "std"),
        edge_coh_mean=("avg_edge_coherence", "mean"),
        wall_time_mean=("wall_time_seconds", "mean"),
        wall_time_total=("wall_time_seconds", "sum"),
        memory_mean=("peak_memory_mb", "mean"),
        api_cost_total=("api_cost_usd", "sum"),
        experiments=("judge_score", "count"),
        failures=("judge_score", lambda x: x.isna().sum()),
    ).round(4)
    return grouped


# ============================================================
# TEST-T DE WELCH
# ============================================================

def welch_t_test(group_a: pd.Series, group_b: pd.Series) -> dict:
    """Test-t de Welch entre dos grupos. Retorna t, p, y Cohen's d."""
    a = group_a.dropna().values
    b = group_b.dropna().values

    if len(a) < 2 or len(b) < 2:
        return {"t_stat": np.nan, "p_value": np.nan, "cohens_d": np.nan,
                "n_a": len(a), "n_b": len(b), "significant": False}

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt(((len(a) - 1) * np.std(a, ddof=1)**2 +
                           (len(b) - 1) * np.std(b, ddof=1)**2) /
                          (len(a) + len(b) - 2))
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "cohens_d": round(cohens_d, 4),
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": round(np.mean(a), 4),
        "mean_b": round(np.mean(b), 4),
        "significant": p_value < 0.05,
    }


def cohens_d_interpretation(d: float) -> str:
    """Interpretar tamaño del efecto según convenciones de Cohen."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def pairwise_comparisons(df: pd.DataFrame, metric: str = "judge_score") -> pd.DataFrame:
    """Comparaciones pairwise entre todos los métodos con test-t de Welch."""
    methods = sorted(df["method"].unique())
    rows = []

    for method_a, method_b in combinations(methods, 2):
        scores_a = df[df["method"] == method_a][metric]
        scores_b = df[df["method"] == method_b][metric]
        result = welch_t_test(scores_a, scores_b)
        result["method_a"] = method_a
        result["method_b"] = method_b
        result["metric"] = metric
        result["effect_size"] = cohens_d_interpretation(result["cohens_d"])
        rows.append(result)

    return pd.DataFrame(rows)


def pairwise_by_n(df: pd.DataFrame, metric: str = "judge_score") -> pd.DataFrame:
    """Comparaciones pairwise estratificadas por tamaño de narrativa."""
    methods = sorted(df["method"].unique())
    n_sizes = sorted(df["n"].unique())
    rows = []

    for n in n_sizes:
        df_n = df[df["n"] == n]
        for method_a, method_b in combinations(methods, 2):
            scores_a = df_n[df_n["method"] == method_a][metric]
            scores_b = df_n[df_n["method"] == method_b][metric]
            result = welch_t_test(scores_a, scores_b)
            result["n"] = n
            result["method_a"] = method_a
            result["method_b"] = method_b
            result["effect_size"] = cohens_d_interpretation(result["cohens_d"])
            rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# VISUALIZACIONES
# ============================================================

def generate_plots(df: pd.DataFrame, output_dir: str):
    """Generar todas las visualizaciones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(output_dir, exist_ok=True)

    # Colores consistentes por método
    method_colors = {
        "maps_algorithmic": "#2196F3",
        "trails_algorithmic": "#4CAF50",
        "llm_extractor_llama32": "#FF9800",
        "llm_extractor_mistral": "#E91E63",
        
        "llm_extractor_gemma3": "#00BCD4",
        "llm_extractor_gpt4omini": "#F44336",
    }

    short_names = {
        "maps_algorithmic": "Maps",
        "trails_algorithmic": "Trails",
        "llm_extractor_llama32": "LLaMA 3.2",
        "llm_extractor_mistral": "Mistral",
        
        "llm_extractor_gemma3": "Gemma 3",
        "llm_extractor_gpt4omini": "GPT-4o-mini",
    }

    methods_order = [m for m in [
        "maps_algorithmic", "trails_algorithmic",
        "llm_extractor_llama32", "llm_extractor_mistral",
        "llm_extractor_gemma3",
        "llm_extractor_gpt4omini",
    ] if m in df["method"].unique()]

    # --- 1. Boxplot: Judge Score por método ---
    fig, ax = plt.subplots(figsize=(12, 6))
    data_to_plot = [df[df["method"] == m]["judge_score"].dropna().values for m in methods_order]
    bp = ax.boxplot(data_to_plot, labels=[short_names.get(m, m) for m in methods_order],
                    patch_artist=True, widths=0.6)
    for patch, method in zip(bp["boxes"], methods_order):
        patch.set_facecolor(method_colors.get(method, "#999999"))
        patch.set_alpha(0.7)
    ax.set_ylabel("Judge Score (0-10)")
    ax.set_title("Coherencia Narrativa por Método (LLM-as-a-Judge)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 10.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "judge_boxplot.png"), dpi=150)
    plt.close()

    # --- 2. Boxplot: Judge Score por método y n ---
    n_sizes = sorted(df["n"].unique())
    fig, axes = plt.subplots(1, len(n_sizes), figsize=(6 * len(n_sizes), 6), sharey=True)
    if len(n_sizes) == 1:
        axes = [axes]
    for ax, n in zip(axes, n_sizes):
        df_n = df[df["n"] == n]
        data = [df_n[df_n["method"] == m]["judge_score"].dropna().values for m in methods_order]
        bp = ax.boxplot(data, labels=[short_names.get(m, m) for m in methods_order],
                        patch_artist=True, widths=0.6)
        for patch, method in zip(bp["boxes"], methods_order):
            patch.set_facecolor(method_colors.get(method, "#999999"))
            patch.set_alpha(0.7)
        ax.set_title(f"n = {n}")
        ax.set_ylim(0, 10.5)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Judge Score (0-10)")
    fig.suptitle("Coherencia por Método y Tamaño de Narrativa", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "judge_by_n.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- 3. Heatmap: Cohen's d ---
    comparisons = pairwise_comparisons(df, "judge_score")
    if not comparisons.empty:
        methods_in = sorted(set(comparisons["method_a"]) | set(comparisons["method_b"]))
        n_methods = len(methods_in)
        d_matrix = np.zeros((n_methods, n_methods))
        p_matrix = np.ones((n_methods, n_methods))
        for _, row in comparisons.iterrows():
            i = methods_in.index(row["method_a"])
            j = methods_in.index(row["method_b"])
            d_matrix[i, j] = row["cohens_d"]
            d_matrix[j, i] = -row["cohens_d"]
            p_matrix[i, j] = row["p_value"]
            p_matrix[j, i] = row["p_value"]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(d_matrix, cmap="RdBu_r", vmin=-2, vmax=2)
        labels = [short_names.get(m, m) for m in methods_in]
        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(n_methods):
            for j in range(n_methods):
                if i == j:
                    continue
                sig = "*" if p_matrix[i, j] < 0.05 else ""
                ax.text(j, i, f"{d_matrix[i,j]:.2f}{sig}",
                        ha="center", va="center", fontsize=9)
        plt.colorbar(im, label="Cohen's d (A vs B)")
        ax.set_title("Tamaño del Efecto (Cohen's d) — Judge Score\n* = p < 0.05")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "effect_size_heatmap.png"), dpi=150)
        plt.close()

    # --- 4. Barplot: Tiempo promedio por método ---
    time_data = df.groupby("method")["wall_time_seconds"].mean().reindex(methods_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        [short_names.get(m, m) for m in methods_order],
        time_data.values,
        color=[method_colors.get(m, "#999") for m in methods_order],
        alpha=0.7,
    )
    ax.set_ylabel("Tiempo promedio (segundos)")
    ax.set_title("Eficiencia: Tiempo de Ejecución por Método")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, time_data.values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wall_time_comparison.png"), dpi=150)
    plt.close()

    # --- 5. Scatter: Calidad vs Tiempo (trade-off) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for method in methods_order:
        df_m = df[df["method"] == method]
        mean_score = df_m["judge_score"].mean()
        mean_time = df_m["wall_time_seconds"].mean()
        if np.isnan(mean_score) or np.isnan(mean_time):
            continue
        ax.scatter(mean_time, mean_score, s=200,
                   color=method_colors.get(method, "#999"), zorder=5,
                   edgecolors="black", linewidth=0.5)
        ax.annotate(short_names.get(method, method),
                    (mean_time, mean_score),
                    textcoords="offset points", xytext=(10, 5), fontsize=10)
    ax.set_xlabel("Tiempo promedio (segundos)")
    ax.set_ylabel("Judge Score promedio (0-10)")
    ax.set_title("Trade-off: Calidad Narrativa vs Eficiencia")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_vs_time.png"), dpi=150)
    plt.close()

    # --- 6. Barplot: Costo API acumulado ---
    cost_data = df.groupby("method")["api_cost_usd"].sum().reindex(methods_order)
    if cost_data.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            [short_names.get(m, m) for m in methods_order],
            cost_data.values,
            color=[method_colors.get(m, "#999") for m in methods_order],
            alpha=0.7,
        )
        ax.set_ylabel("Costo API acumulado (USD)")
        ax.set_title("Costo Computacional por Método")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, cost_data.values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"${val:.4f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "api_cost.png"), dpi=150)
        plt.close()

    print(f"  Plots guardados en {output_dir}/")


# ============================================================
# EXPORTAR TABLAS
# ============================================================

def export_tables(df: pd.DataFrame, comparisons: pd.DataFrame,
                  comparisons_by_n: pd.DataFrame, output_dir: str):
    """Exportar tablas a CSV y markdown."""
    os.makedirs(output_dir, exist_ok=True)

    # Descriptivas
    desc = descriptive_stats(df)
    desc.to_csv(os.path.join(output_dir, "descriptive_stats.csv"))

    summary = summary_by_method(df)
    summary.to_csv(os.path.join(output_dir, "summary_by_method.csv"))

    # Comparaciones
    comparisons.to_csv(os.path.join(output_dir, "pairwise_welch_t.csv"), index=False)
    comparisons_by_n.to_csv(os.path.join(output_dir, "pairwise_welch_t_by_n.csv"), index=False)

    # Markdown summary para el informe
    md_path = os.path.join(output_dir, "results_summary.md")
    with open(md_path, "w") as f:
        f.write("# Resultados del Análisis Estadístico\n\n")

        f.write("## Resumen por Método\n\n")
        f.write(summary.to_markdown() + "\n\n")

        f.write("## Estadísticas Descriptivas por Método y n\n\n")
        f.write(desc.to_markdown() + "\n\n")

        f.write("## Comparaciones Pairwise (Test-t de Welch)\n\n")
        cols = ["method_a", "method_b", "mean_a", "mean_b", "t_stat",
                "p_value", "cohens_d", "effect_size", "significant"]
        f.write(comparisons[cols].to_markdown(index=False) + "\n\n")

        # Significativos
        sig = comparisons[comparisons["significant"]]
        if not sig.empty:
            f.write("### Diferencias Significativas (p < 0.05)\n\n")
            f.write(sig[cols].to_markdown(index=False) + "\n\n")

        # Resumen de hallazgos
        f.write("## Interpretación de Cohen's d\n\n")
        f.write("| Rango |d| | Interpretación |\n")
        f.write("|---|---|\n")
        f.write("| < 0.2 | Negligible |\n")
        f.write("| 0.2 - 0.5 | Small |\n")
        f.write("| 0.5 - 0.8 | Medium |\n")
        f.write("| > 0.8 | Large |\n\n")

    print(f"  Tablas exportadas en {output_dir}/")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Análisis estadístico de resultados")
    parser.add_argument("--results-dir", default="results_local/results_v2",
                        help="Directorio con resultados JSON")
    parser.add_argument("--output-dir", default="results_local/analysis",
                        help="Directorio de salida para tablas y plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="Saltar generación de plots")
    args = parser.parse_args()

    plots_dir = os.path.join(args.output_dir, "plots")
    tables_dir = os.path.join(args.output_dir, "tables")

    # Cargar datos
    print("=" * 60)
    print("ANÁLISIS ESTADÍSTICO — Narrative Extraction")
    print("=" * 60)
    df = load_all_results(args.results_dir)

    # Estadísticas descriptivas
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DESCRIPTIVAS")
    print("=" * 60)
    summary = summary_by_method(df)
    print(summary.to_string())

    # Comparaciones pairwise
    print("\n" + "=" * 60)
    print("COMPARACIONES PAIRWISE (Test-t de Welch + Cohen's d)")
    print("=" * 60)
    comparisons = pairwise_comparisons(df, "judge_score")
    if not comparisons.empty:
        display_cols = ["method_a", "method_b", "mean_a", "mean_b",
                        "t_stat", "p_value", "cohens_d", "effect_size", "significant"]
        print(comparisons[display_cols].to_string(index=False))

        sig = comparisons[comparisons["significant"]]
        print(f"\n  Diferencias significativas (p < 0.05): {len(sig)}/{len(comparisons)}")

    # Comparaciones por n
    print("\n" + "=" * 60)
    print("COMPARACIONES POR TAMAÑO DE NARRATIVA")
    print("=" * 60)
    comparisons_by_n = pairwise_by_n(df, "judge_score")
    if not comparisons_by_n.empty:
        for n in sorted(df["n"].unique()):
            comp_n = comparisons_by_n[comparisons_by_n["n"] == n]
            sig_n = comp_n[comp_n["significant"]]
            print(f"\n  n={n}: {len(sig_n)}/{len(comp_n)} diferencias significativas")

    # Exportar
    print("\n" + "=" * 60)
    print("EXPORTANDO RESULTADOS")
    print("=" * 60)
    export_tables(df, comparisons, comparisons_by_n, tables_dir)

    if not args.no_plots:
        try:
            generate_plots(df, plots_dir)
        except ImportError as e:
            print(f"  AVISO: matplotlib no disponible, saltando plots: {e}")

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    best_method = summary["judge_mean"].idxmax()
    best_score = summary.loc[best_method, "judge_mean"]
    print(f"  Mejor método (judge_score): {best_method} ({best_score:.2f}/10)")

    local_llm_methods = df[df["method_type"] == "local_llm"]["method"].unique()
    if len(local_llm_methods) > 0:
        local_scores = df[df["method_type"] == "local_llm"].groupby("method")["judge_score"].mean()
        best_local = local_scores.idxmax()
        print(f"  Mejor LLM local: {best_local} ({local_scores[best_local]:.2f}/10)")

    alg_score = df[df["method_type"] == "algorithmic"]["judge_score"].mean()
    local_score = df[df["method_type"] == "local_llm"]["judge_score"].mean()
    prop_score = df[df["method_type"] == "proprietary"]["judge_score"].mean()
    print(f"\n  Promedios por tipo:")
    print(f"    Algorítmicos:  {alg_score:.2f}/10")
    print(f"    LLMs locales:  {local_score:.2f}/10")
    if not np.isnan(prop_score):
        print(f"    Propietario:   {prop_score:.2f}/10")

    print(f"\n  Archivos generados en: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
