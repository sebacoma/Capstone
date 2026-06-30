#!/usr/bin/env python3
"""
Genera todas las figuras con los datos re-evaluados por Claude Sonnet 4.6.

Figuras:
  a) boxplot_judge_claude.png
  b) heatmap_cohens_d.png
  c) scatter_quality_vs_time.png
  d) jaccard_heatmap.png
  e) judge_comparison.png
  f) tasa_exito_camino2_real.png

Uso:
    python generate_figures.py
"""

import json
import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from run_experiments_local import calculate_jaccard_similarity

INPUT_JSON = "results_camino2_claude_judge.json"
ANALYSIS_DIR = "analysis_v2"
OUTPUT_DIR = "analysis_v2/figures"

# Estilo consistente
FIGSIZE = (13, 7)
DPI = 150
COLORS = {
    "maps_algorithmic": "#4472C4",
    "trails_algorithmic": "#ED7D31",
    "llm_extractor_llama32": "#A5A5A5",
    "llm_extractor_mistral": "#FFC000",
    "llm_extractor_gemma3": "#5B9BD5",
    "llm_extractor_gpt4omini": "#70AD47",
}
SHORT_NAMES = {
    "maps_algorithmic": "Maps",
    "trails_algorithmic": "Trails",
    "llm_extractor_llama32": "LLaMA 3.2",
    "llm_extractor_mistral": "Mistral",
    "llm_extractor_gemma3": "Gemma 3",
    "llm_extractor_gpt4omini": "GPT-4o-mini",
}
METHODS_ORDER = [
    "maps_algorithmic", "trails_algorithmic",
    "llm_extractor_gpt4omini", "llm_extractor_llama32",
    "llm_extractor_mistral", "llm_extractor_gemma3",
]


def load_data() -> pd.DataFrame:
    with open(INPUT_JSON) as f:
        data = json.load(f)
    rows = []
    for r in data:
        narrative_ids = [d["id"] for d in r.get("narrative", [])]
        rows.append({
            "method": r["method"],
            "n": r["n"],
            "start_index": r.get("start_index", 0),
            "judge_score_llama": r.get("judge_score_llama"),
            "judge_score_claude": r.get("judge_score_claude"),
            "avg_edge_coherence": r.get("metrics", {}).get("avg_edge_coherence"),
            "wall_time_seconds": r.get("metrics", {}).get("wall_time_seconds"),
            "narrative_ids": narrative_ids,
        })
    df = pd.DataFrame(rows)
    for col in ["judge_score_llama", "judge_score_claude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
    return df


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })


# ============================================================
# a) Boxplot Judge Claude
# ============================================================
def plot_boxplot_claude(df):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    methods = [m for m in METHODS_ORDER if m in df["method"].unique()]
    data = [df[df["method"] == m]["judge_score_claude"].dropna().values for m in methods]
    labels = [SHORT_NAMES.get(m, m) for m in methods]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(COLORS.get(method, "#999"))
        patch.set_alpha(0.7)

    # Añadir puntos individuales
    for i, (d, method) in enumerate(zip(data, methods)):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.3, s=15, color=COLORS.get(method, "#999"), zorder=3)

    ax.set_ylabel("Coherence Score (0-10)", fontsize=12)
    ax.set_title("Coherencia Narrativa por Método — Juez: Claude Sonnet 4.6", fontsize=14)
    ax.set_ylim(-0.5, 10.5)

    # Añadir medias como texto
    for i, d in enumerate(data):
        if len(d) > 0:
            ax.text(i + 1, -0.3, f"μ={np.mean(d):.1f}", ha="center", fontsize=9, color="#555")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_judge_claude.png"), dpi=DPI)
    plt.close()
    print("  -> boxplot_judge_claude.png")


# ============================================================
# b) Heatmap Cohen's d
# ============================================================
def plot_heatmap_cohens_d(df):
    methods = [m for m in METHODS_ORDER if m in df["method"].unique()]
    n = len(methods)
    d_matrix = np.zeros((n, n))
    p_matrix = np.ones((n, n))

    for i, j in combinations(range(n), 2):
        a = df[df["method"] == methods[i]]["judge_score_claude"].dropna().values
        b = df[df["method"] == methods[j]]["judge_score_claude"].dropna().values
        if len(a) >= 2 and len(b) >= 2:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            pooled = np.sqrt(((len(a)-1)*np.std(a,ddof=1)**2 +
                              (len(b)-1)*np.std(b,ddof=1)**2) / (len(a)+len(b)-2))
            d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0
            d_matrix[i, j] = d
            d_matrix[j, i] = -d
            p_matrix[i, j] = p
            p_matrix[j, i] = p

    n_comp = n * (n - 1) // 2

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(d_matrix, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    labels = [SHORT_NAMES.get(m, m) for m in methods]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=10, color="#999")
                continue
            p_bonf = min(p_matrix[i, j] * n_comp, 1.0)
            sig = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else ""
            color = "white" if abs(d_matrix[i, j]) > 1.5 else "black"
            ax.text(j, i, f"{d_matrix[i,j]:.2f}{sig}",
                    ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, label="Cohen's d (fila vs columna)")
    ax.set_title("Tamaño del Efecto (Cohen's d) — Judge Score Claude\n* p<0.05  ** p<0.01  *** p<0.001 (Bonferroni)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_cohens_d.png"), dpi=DPI)
    plt.close()
    print("  -> heatmap_cohens_d.png")


# ============================================================
# c) Scatter Quality vs Time
# ============================================================
def plot_scatter_quality_time(df):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    methods = [m for m in METHODS_ORDER if m in df["method"].unique()]

    for method in methods:
        df_m = df[df["method"] == method]
        mean_score = df_m["judge_score_claude"].mean()
        mean_time = df_m["wall_time_seconds"].mean()
        if np.isnan(mean_score) or np.isnan(mean_time):
            continue
        ax.scatter(mean_time, mean_score, s=250,
                   color=COLORS.get(method, "#999"), zorder=5,
                   edgecolors="black", linewidth=1)
        ax.annotate(SHORT_NAMES.get(method, method),
                    (mean_time, mean_score),
                    textcoords="offset points", xytext=(12, 5),
                    fontsize=11, fontweight="bold")

    ax.set_xlabel("Tiempo promedio (segundos)", fontsize=12)
    ax.set_ylabel("Coherence Score promedio (Claude judge)", fontsize=12)
    ax.set_title("Trade-off: Calidad Narrativa vs Eficiencia", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter_quality_vs_time.png"), dpi=DPI)
    plt.close()
    print("  -> scatter_quality_vs_time.png")


# ============================================================
# d) Jaccard Heatmap
# ============================================================
def plot_jaccard_heatmap(df):
    # Leer desde CSV si existe, sino calcular
    jac_path = os.path.join(ANALYSIS_DIR, "jaccard_matrix.csv")
    if os.path.exists(jac_path):
        jac = pd.read_csv(jac_path, index_col=0)
    else:
        # Calcular
        from analyze_new_results import jaccard_matrix
        jac = jaccard_matrix(df)

    methods = [m for m in METHODS_ORDER if m in jac.index]
    jac = jac.loc[methods, methods]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(jac.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    labels = [SHORT_NAMES.get(m, m) for m in methods]
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(methods)):
        for j in range(len(methods)):
            val = jac.values[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=color)

    plt.colorbar(im, label="Jaccard Similarity")
    ax.set_title("Similitud de Jaccard entre Métodos\n(proporción de documentos compartidos)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "jaccard_heatmap.png"), dpi=DPI)
    plt.close()
    print("  -> jaccard_heatmap.png")


# ============================================================
# e) Judge Comparison (LLaMA vs Claude)
# ============================================================
def plot_judge_comparison(df):
    methods = [m for m in METHODS_ORDER if m in df["method"].unique()]

    llama_means = []
    claude_means = []
    llama_stds = []
    claude_stds = []
    valid_methods = []

    for m in methods:
        df_m = df[df["method"] == m]
        lm = df_m["judge_score_llama"].dropna()
        cm = df_m["judge_score_claude"].dropna()
        if len(lm) > 0 and len(cm) > 0:
            llama_means.append(lm.mean())
            claude_means.append(cm.mean())
            llama_stds.append(lm.std())
            claude_stds.append(cm.std())
            valid_methods.append(m)

    x = np.arange(len(valid_methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars1 = ax.bar(x - width/2, llama_means, width, yerr=llama_stds,
                   label="LLaMA 3.2 (3B)", color="#FF9800", alpha=0.7,
                   capsize=4, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, claude_means, width, yerr=claude_stds,
                   label="Claude Sonnet 4.6", color="#4472C4", alpha=0.7,
                   capsize=4, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Coherence Score medio (0-10)", fontsize=12)
    ax.set_title("Comparación de Jueces: LLaMA 3.2 vs Claude Sonnet 4.6\n"
                 "(Hallazgo metodológico: discriminación del juez)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(m, m) for m in valid_methods], rotation=30, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 10.5)

    # Anotar medias
    for bar, val in zip(bars1, llama_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", fontsize=9, color="#666")
    for bar, val in zip(bars2, claude_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", fontsize=9, color="#666")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "judge_comparison.png"), dpi=DPI)
    plt.close()
    print("  -> judge_comparison.png")


# ============================================================
# f) Tasa de Éxito por método y n
# ============================================================
def plot_tasa_exito(df):
    """Bar plot agrupado de tasa de éxito por método y por n."""
    methods = [m for m in METHODS_ORDER if m in df["method"].unique()]
    n_sizes = sorted(df["n"].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(methods))
    total_bars = len(n_sizes)
    width = 0.8 / total_bars

    for i, n_size in enumerate(n_sizes):
        rates = []
        for m in methods:
            df_mn = df[(df["method"] == m) & (df["n"] == n_size)]
            total = len(df_mn)
            success = df_mn["judge_score_claude"].notna().sum()
            rates.append(success / total * 100 if total > 0 else 0)
        offset = (i - total_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=f"n={n_size}",
                      alpha=0.7, edgecolor="black", linewidth=0.5)
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{rate:.0f}%", ha="center", fontsize=8)

    ax.set_ylabel("Tasa de Éxito (%)", fontsize=12)
    ax.set_title("Tasa de Éxito por Método y Tamaño de Narrativa", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(m, m) for m in methods], rotation=30, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 115)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tasa_exito_camino2_real.png"), dpi=DPI)
    plt.close()
    print("  -> tasa_exito_camino2_real.png")


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(INPUT_JSON):
        print(f"ERROR: No se encontró {INPUT_JSON}")
        print("  Ejecuta primero: python rejudge_camino2.py")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()

    print("=" * 60)
    print("GENERACIÓN DE FIGURAS — Claude Judge")
    print("=" * 60)

    df = load_data()
    print(f"  {len(df)} resultados cargados")

    plot_boxplot_claude(df)
    plot_heatmap_cohens_d(df)
    plot_scatter_quality_time(df)
    plot_jaccard_heatmap(df)
    plot_judge_comparison(df)
    plot_tasa_exito(df)

    print(f"\n  Todas las figuras en: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
