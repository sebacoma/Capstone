#!/usr/bin/env python3
"""
Genera figuras nuevas con los tres jueces: LLaMA, Claude y Gemini.

Figuras generadas en analysis_v3/figures/:
  - raincloud_judge_scores.png     (REEMPLAZA boxplot_judge_claude.png)
  - three_judges_comparison.png
  - judge_agreement_heatmap.png
  - heatmap_cohens_d_gemini.png

NO sobreescribe figuras existentes (jaccard_heatmap, tasa_exito, scatter, etc.)

Uso:
    python generate_figures_v3.py
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
from scipy.stats import gaussian_kde

INPUT_JSON = "results_camino2_three_judges.json"
ANALYSIS_DIR = "analysis_v3"
OUTPUT_DIR = "analysis_v3/figures"

DPI = 150

COLORS = {
    "maps_algorithmic":       "#4472C4",
    "trails_algorithmic":     "#ED7D31",
    "llm_extractor_llama32":  "#A5A5A5",
    "llm_extractor_mistral":  "#FFC000",
    "llm_extractor_gemma3":   "#5B9BD5",
    "llm_extractor_gpt4omini":"#70AD47",
}
SHORT_NAMES = {
    "maps_algorithmic":        "Maps",
    "trails_algorithmic":      "Trails",
    "llm_extractor_llama32":   "LLaMA 3.2",
    "llm_extractor_mistral":   "Mistral",
    "llm_extractor_gemma3":    "Gemma 3",
    "llm_extractor_gpt4omini": "GPT-4o-mini",
}
METHODS_ORDER = [
    "maps_algorithmic", "trails_algorithmic",
    "llm_extractor_gpt4omini", "llm_extractor_llama32",
    "llm_extractor_mistral", "llm_extractor_gemma3",
]

JUDGE_COLORS = {
    "llama":  "#FF9800",
    "claude": "#4472C4",
    "gemini": "#34A853",
}
JUDGE_LABELS = {
    "llama":  "LLaMA 3.2",
    "claude": "Claude Sonnet 4.6",
    "gemini": "Gemini 2.5 Flash",
}


def load_data() -> pd.DataFrame:
    with open(INPUT_JSON, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        rows.append({
            "method": r["method"],
            "n": r.get("n"),
            "start_index": r.get("start_index", 0),
            "judge_score_llama":  r.get("judge_score_llama"),
            "judge_score_claude": r.get("judge_score_claude"),
            "judge_score_gemini": r.get("judge_score_gemini"),
            "wall_time_seconds":  r.get("metrics", {}).get("wall_time_seconds"),
        })
    df = pd.DataFrame(rows)
    for col in ["judge_score_llama", "judge_score_claude", "judge_score_gemini"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
    return df


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "font.size":        11,
    })


# ============================================================
# 4.1  RAINCLOUD PLOT — tres paneles (uno por juez)
# ============================================================

def _draw_raincloud_panel(ax, df, score_col, methods, title, color_dict):
    """Dibuja un panel de raincloud plot horizontal."""
    n_methods = len(methods)
    y_positions = np.arange(n_methods)

    for i, method in enumerate(methods):
        scores = df[df["method"] == method][score_col].dropna().values
        if len(scores) == 0:
            continue

        color = color_dict.get(method, "#999")
        y = y_positions[i]

        # --- Nube (KDE) — arriba del eje Y ---
        if len(scores) >= 3 and np.std(scores) > 0:
            try:
                kde = gaussian_kde(scores, bw_method=0.4)
                x_range = np.linspace(0, 10, 200)
                kde_vals = kde(x_range)
                kde_vals = kde_vals / kde_vals.max() * 0.35  # normalizar altura
                ax.fill_between(x_range, y + 0.05, y + 0.05 + kde_vals,
                                alpha=0.5, color=color)
                ax.plot(x_range, y + 0.05 + kde_vals, color=color, linewidth=0.8, alpha=0.8)
            except Exception:
                pass  # skip KDE si los datos tienen varianza demasiado baja

        # --- Boxplot al medio ---
        bp = ax.boxplot(
            scores,
            positions=[y],
            vert=False,
            widths=0.12,
            patch_artist=True,
            manage_ticks=False,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            flierprops=dict(marker="o", color=color, alpha=0.3, markersize=3),
        )

        # --- Lluvia (puntos individuales) — abajo ---
        jitter = np.random.uniform(-0.10, -0.02, size=len(scores))
        ax.scatter(scores, y + jitter,
                   alpha=0.35, s=12, color=color, zorder=3)

    # Etiquetas eje Y
    ax.set_yticks(y_positions)
    ax.set_yticklabels([SHORT_NAMES.get(m, m) for m in methods], fontsize=10)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.6, n_methods - 0.3)
    ax.set_xlabel("Coherence Score (0–10)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axvline(5, color="#999", linestyle="--", linewidth=0.8, alpha=0.5)


def plot_raincloud(df):
    methods = [m for m in METHODS_ORDER if m in df["method"].unique()]
    judges = [
        ("judge_score_llama",  "LLaMA 3.2 (3B)"),
        ("judge_score_claude", "Claude Sonnet 4.6"),
        ("judge_score_gemini", "Gemini 2.5 Flash"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    fig.suptitle("Distribución de scores por juez y método", fontsize=15, fontweight="bold", y=1.01)

    np.random.seed(42)
    for ax, (score_col, judge_name) in zip(axes, judges):
        _draw_raincloud_panel(ax, df, score_col, methods, judge_name, COLORS)

    # Solo el primer panel muestra etiquetas Y
    for ax in axes[1:]:
        ax.set_ylabel("")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "raincloud_judge_scores.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  -> raincloud_judge_scores.png")


# ============================================================
# 4.2  BAR PLOT AGRUPADO — tres jueces
# ============================================================

def plot_three_judges_comparison(df):
    methods = [m for m in METHODS_ORDER if m in df["method"].unique()]

    claude_means, claude_stds = [], []
    gemini_means, gemini_stds = [], []

    for m in methods:
        sub = df[df["method"] == m]
        for score_col, means_list, stds_list in [
            ("judge_score_claude", claude_means, claude_stds),
            ("judge_score_gemini", gemini_means, gemini_stds),
        ]:
            vals = sub[score_col].dropna().values
            means_list.append(np.mean(vals) if len(vals) else np.nan)
            stds_list.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)

    x = np.arange(len(methods))
    width = 0.30

    fig, ax = plt.subplots(figsize=(14, 7))
    kw = dict(alpha=0.8, edgecolor="black", linewidth=0.5, capsize=4)

    bars2 = ax.bar(x - width/2, claude_means, width, yerr=claude_stds,
                   label=JUDGE_LABELS["claude"], color=JUDGE_COLORS["claude"], **kw)
    bars3 = ax.bar(x + width/2, gemini_means, width, yerr=gemini_stds,
                   label=JUDGE_LABELS["gemini"], color=JUDGE_COLORS["gemini"], **kw)

    # Anotar medias
    for bars, means in [(bars2, claude_means), (bars3, gemini_means)]:
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.25,
                        f"{val:.1f}", ha="center", fontsize=8, color="#444")

    ax.set_ylabel("Coherence Score medio (0–10)", fontsize=12)
    ax.set_title("Comparación de jueces por método\n"
                 "(Claude Sonnet 4.6 · Gemini 2.5 Flash)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(m, m) for m in methods], rotation=30, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 12)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "three_judges_comparison.png")
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"  -> three_judges_comparison.png")


# ============================================================
# 4.3  HEATMAP ACUERDO ENTRE JUECES
# ============================================================

def plot_judge_agreement_heatmap(df):
    judges = ["judge_score_llama", "judge_score_claude", "judge_score_gemini"]
    labels = ["LLaMA 3.2", "Claude Sonnet 4.6", "Gemini 2.5 Flash"]
    n = len(judges)
    matrix = np.full((n, n), np.nan)
    p_matrix = np.ones((n, n))

    for i, j1 in enumerate(judges):
        for j_idx, j2 in enumerate(judges):
            both = df[[j1, j2]].dropna()
            if len(both) >= 3:
                res = stats.spearmanr(both[j1].values, both[j2].values)
                rho_val = np.asarray(res.statistic)
                p_val = np.asarray(res.pvalue)
                # spearmanr puede devolver matriz 2x2 cuando recibe 2 arrays
                matrix[i, j_idx] = float(rho_val[0, 1]) if rho_val.ndim == 2 else float(rho_val)
                p_matrix[i, j_idx] = float(p_val[0, 1]) if p_val.ndim == 2 else float(p_val)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    for i in range(n):
        for j_idx in range(n):
            val = matrix[i, j_idx]
            p = p_matrix[i, j_idx]
            if np.isnan(val):
                continue
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j_idx, i, f"ρ={val:.3f}{sig}",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    plt.colorbar(im, label="Spearman ρ", fraction=0.046, pad=0.04)
    ax.set_title("Acuerdo entre jueces — Correlación Spearman\n"
                 "* p<0.05  ** p<0.01  *** p<0.001", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "judge_agreement_heatmap.png")
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"  -> judge_agreement_heatmap.png")


# ============================================================
# 4.4  HEATMAP COHEN'S D — GEMINI
# ============================================================

def plot_heatmap_cohens_d_gemini(df):
    methods = [m for m in METHODS_ORDER if m in df["method"].unique()]
    n = len(methods)
    d_matrix = np.zeros((n, n))
    p_matrix = np.ones((n, n))

    for i, j in combinations(range(n), 2):
        a = df[df["method"] == methods[i]]["judge_score_gemini"].dropna().values
        b = df[df["method"] == methods[j]]["judge_score_gemini"].dropna().values
        if len(a) >= 2 and len(b) >= 2:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            pooled = np.sqrt(
                ((len(a)-1)*np.std(a, ddof=1)**2 + (len(b)-1)*np.std(b, ddof=1)**2)
                / (len(a)+len(b)-2)
            )
            d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0
            d_matrix[i, j] = d
            d_matrix[j, i] = -d
            p_matrix[i, j] = p
            p_matrix[j, i] = p

    n_comp = n * (n - 1) // 2
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(d_matrix, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    labels = [SHORT_NAMES.get(m, m) for m in methods]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="#999")
                continue
            p_bonf = min(p_matrix[i, j] * n_comp, 1.0)
            sig = ("***" if p_bonf < 0.001 else
                   "**"  if p_bonf < 0.01  else
                   "*"   if p_bonf < 0.05  else "")
            color = "white" if abs(d_matrix[i, j]) > 1.5 else "black"
            ax.text(j, i, f"{d_matrix[i,j]:.2f}{sig}",
                    ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, label="Cohen's d (fila vs columna)")
    ax.set_title("Tamaño del Efecto (Cohen's d) — Juez: Gemini 2.5 Flash\n"
                 "* p<0.05  ** p<0.01  *** p<0.001 (Bonferroni)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "heatmap_cohens_d_gemini.png")
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"  -> heatmap_cohens_d_gemini.png")


# ============================================================
# MAIN
# ============================================================

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"ERROR: No se encontró {INPUT_JSON}")
        print("  Ejecuta primero: python rejudge_with_gemini.py")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()
    np.random.seed(42)

    print("=" * 60)
    print("GENERACIÓN DE FIGURAS — Tres Jueces (v3)")
    print("=" * 60)

    df = load_data()
    print(f"  {len(df)} resultados cargados")

    plot_raincloud(df)
    plot_three_judges_comparison(df)
    plot_judge_agreement_heatmap(df)
    plot_heatmap_cohens_d_gemini(df)

    print(f"\n  Todas las figuras en: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
