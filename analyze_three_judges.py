#!/usr/bin/env python3
"""
Análisis estadístico con los tres jueces: LLaMA, Claude y Gemini.

Genera en analysis_v3/:
  - descriptive_stats_gemini.csv
  - pairwise_welch_gemini.csv
  - three_judges_comparison.csv
  - judge_agreement_matrix.csv

Uso:
    python analyze_three_judges.py
"""

import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

INPUT_JSON = "results_camino2_three_judges.json"
OUTPUT_DIR = "analysis_v3"


# ============================================================
# HELPERS
# ============================================================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2)
        / (na + nb - 2)
    )
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0


def effect_size_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    return "large"


def pairwise_welch(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    methods = sorted(df["method"].unique())
    n_comparisons = len(list(combinations(methods, 2)))
    rows = []
    for method_a, method_b in combinations(methods, 2):
        a = df[df["method"] == method_a][metric].dropna().values
        b = df[df["method"] == method_b][metric].dropna().values
        if len(a) < 2 or len(b) < 2:
            rows.append({
                "method_a": method_a, "method_b": method_b,
                "mean_a": np.nan, "mean_b": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "p_bonferroni": np.nan,
                "cohens_d": np.nan, "effect_size": "N/A",
                "significant_bonferroni": False,
                "n_a": len(a), "n_b": len(b),
            })
            continue
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        d = cohens_d(a, b)
        p_bonf = min(p_value * n_comparisons, 1.0)
        rows.append({
            "method_a": method_a, "method_b": method_b,
            "mean_a": round(np.mean(a), 4), "mean_b": round(np.mean(b), 4),
            "t_stat": round(t_stat, 4), "p_value": round(p_value, 6),
            "p_bonferroni": round(p_bonf, 6),
            "cohens_d": round(d, 4), "effect_size": effect_size_label(d),
            "significant_bonferroni": p_bonf < 0.05,
            "n_a": len(a), "n_b": len(b),
        })
    return pd.DataFrame(rows)


# ============================================================
# CARGA
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        rows.append({
            "method": r["method"],
            "n": r.get("n"),
            "start_index": r.get("start_index", 0),
            "judge_score_llama": r.get("judge_score_llama"),
            "judge_score_claude": r.get("judge_score_claude"),
            "judge_score_gemini": r.get("judge_score_gemini"),
        })
    df = pd.DataFrame(rows)
    for col in ["judge_score_llama", "judge_score_claude", "judge_score_gemini"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
    print(f"Cargados {len(df)} resultados")
    print(f"  Métodos: {sorted(df['method'].unique())}")
    for col in ["judge_score_llama", "judge_score_claude", "judge_score_gemini"]:
        print(f"  {col} válidos: {df[col].notna().sum()}/{len(df)}")
    return df


# ============================================================
# 3.1 ESTADÍSTICAS DESCRIPTIVAS GEMINI
# ============================================================

def descriptive_stats_gemini(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("method").agg(
        mean=("judge_score_gemini", "mean"),
        median=("judge_score_gemini", "median"),
        std=("judge_score_gemini", "std"),
        count=("judge_score_gemini", "count"),
        valid=("judge_score_gemini", lambda x: x.notna().sum()),
        failures=("judge_score_gemini", lambda x: x.isna().sum()),
    ).round(4)
    return grouped


# ============================================================
# 3.2 WELCH PAIRWISE GEMINI
# ============================================================

# (usa función pairwise_welch de arriba con metric="judge_score_gemini")


# ============================================================
# 3.3 COMPARACIÓN TRES JUECES
# ============================================================

def three_judges_comparison(df: pd.DataFrame) -> pd.DataFrame:
    methods = sorted(df["method"].unique())
    rows = []
    for method in methods:
        sub = df[df["method"] == method]
        llama = sub["judge_score_llama"].dropna().values
        claude = sub["judge_score_claude"].dropna().values
        gemini = sub["judge_score_gemini"].dropna().values

        # Spearman Claude vs Gemini (sobre casos con ambos scores)
        both = sub[["judge_score_claude", "judge_score_gemini"]].dropna()
        if len(both) >= 3:
            rho, pval = stats.spearmanr(both["judge_score_claude"], both["judge_score_gemini"])
        else:
            rho, pval = np.nan, np.nan

        # Cohen's kappa binarizado (score >= 6 = "buena narrativa")
        if len(both) >= 3:
            c_bin = (both["judge_score_claude"] >= 6).astype(int)
            g_bin = (both["judge_score_gemini"] >= 6).astype(int)
            kappa = _cohen_kappa(c_bin.values, g_bin.values)
        else:
            kappa = np.nan

        rows.append({
            "method": method,
            "llama_mean": round(np.mean(llama), 4) if len(llama) else np.nan,
            "llama_std": round(np.std(llama, ddof=1), 4) if len(llama) > 1 else np.nan,
            "llama_n": len(llama),
            "claude_mean": round(np.mean(claude), 4) if len(claude) else np.nan,
            "claude_std": round(np.std(claude, ddof=1), 4) if len(claude) > 1 else np.nan,
            "claude_n": len(claude),
            "gemini_mean": round(np.mean(gemini), 4) if len(gemini) else np.nan,
            "gemini_std": round(np.std(gemini, ddof=1), 4) if len(gemini) > 1 else np.nan,
            "gemini_n": len(gemini),
            "spearman_claude_vs_gemini": round(rho, 4) if not np.isnan(rho) else np.nan,
            "p_value_spearman": round(pval, 6) if not np.isnan(pval) else np.nan,
            "cohen_kappa_claude_vs_gemini": round(kappa, 4) if not np.isnan(kappa) else np.nan,
        })
    return pd.DataFrame(rows)


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa simple para arrays binarios."""
    n = len(a)
    if n == 0:
        return np.nan
    p_agree = np.mean(a == b)
    p_a1 = np.mean(a)
    p_b1 = np.mean(b)
    p_expected = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if p_expected == 1.0:
        return 1.0
    return (p_agree - p_expected) / (1 - p_expected)


# ============================================================
# 3.4 MATRIZ DE ACUERDO ENTRE JUECES
# ============================================================

def judge_agreement_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Correlaciones Spearman entre los 3 jueces (a nivel de todos los experimentos)."""
    judges = ["judge_score_llama", "judge_score_claude", "judge_score_gemini"]
    labels = ["llama", "claude", "gemini"]
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)

    for i, (j1, l1) in enumerate(zip(judges, labels)):
        for j2, l2 in zip(judges, labels):
            if j1 == j2:
                matrix.loc[l1, l2] = 1.0
                continue
            both = df[[j1, j2]].dropna()
            if len(both) >= 3:
                rho, _ = stats.spearmanr(both[j1].values, both[j2].values)
                matrix.loc[l1, l2] = round(float(rho), 4)
            else:
                matrix.loc[l1, l2] = np.nan

    return matrix


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nCargando {INPUT_JSON}...")
    df = load_data(INPUT_JSON)

    # 3.1 Descriptive stats Gemini
    print("\n[3.1] Estadísticas descriptivas Gemini...")
    stats_g = descriptive_stats_gemini(df)
    out = os.path.join(OUTPUT_DIR, "descriptive_stats_gemini.csv")
    stats_g.to_csv(out)
    print(f"  -> {out}")
    print(stats_g.to_string())

    # 3.2 Welch pairwise Gemini
    print("\n[3.2] Welch pairwise Gemini...")
    welch_g = pairwise_welch(df, "judge_score_gemini")
    out = os.path.join(OUTPUT_DIR, "pairwise_welch_gemini.csv")
    welch_g.to_csv(out, index=False)
    print(f"  -> {out}")
    sig_g = welch_g["significant_bonferroni"].sum()
    print(f"  Comparaciones significativas (Bonferroni): {sig_g}/15")

    # También para Claude y LLaMA (para referencia cruzada)
    welch_c = pairwise_welch(df, "judge_score_claude")
    sig_c = welch_c["significant_bonferroni"].sum()
    welch_l = pairwise_welch(df, "judge_score_llama")
    sig_l = welch_l["significant_bonferroni"].sum()
    print(f"  (Para referencia — Claude: {sig_c}/15, LLaMA: {sig_l}/15)")

    # 3.3 Three judges comparison
    print("\n[3.3] Comparación tres jueces por método...")
    comp = three_judges_comparison(df)
    out = os.path.join(OUTPUT_DIR, "three_judges_comparison.csv")
    comp.to_csv(out, index=False)
    print(f"  -> {out}")
    print(comp[["method", "llama_mean", "claude_mean", "gemini_mean",
                "spearman_claude_vs_gemini"]].to_string(index=False))

    # 3.4 Matriz de acuerdo
    print("\n[3.4] Matriz de acuerdo entre jueces...")
    agree = judge_agreement_matrix(df)
    out = os.path.join(OUTPUT_DIR, "judge_agreement_matrix.csv")
    agree.to_csv(out)
    print(f"  -> {out}")
    print(agree.to_string())

    # Correlación global Claude vs Gemini
    both_cg = df[["judge_score_claude", "judge_score_gemini"]].dropna()
    if len(both_cg) >= 3:
        rho_cg, p_cg = stats.spearmanr(both_cg["judge_score_claude"], both_cg["judge_score_gemini"])
        print(f"\n  Correlación global Claude vs Gemini: ρ = {rho_cg:.4f} (p={p_cg:.4f})")

    # Resumen de rankings
    print("\n[RANKINGS POR MÉTODO]")
    print("  Claude ranking:")
    claude_rank = df.groupby("method")["judge_score_claude"].mean().sort_values(ascending=False)
    for m, v in claude_rank.items():
        print(f"    {m}: {v:.3f}")
    print("  Gemini ranking:")
    gemini_rank = df.groupby("method")["judge_score_gemini"].mean().sort_values(ascending=False)
    for m, v in gemini_rank.items():
        print(f"    {m}: {v:.3f}")

    # Correlación de rankings
    rank_df = pd.DataFrame({"claude": claude_rank, "gemini": gemini_rank}).dropna()
    if len(rank_df) >= 3:
        rho_rank, p_rank = stats.spearmanr(rank_df["claude"], rank_df["gemini"])
        print(f"\n  Correlación de rankings Claude vs Gemini: ρ = {rho_rank:.4f} (p={p_rank:.4f})")

    print(f"\n{'=' * 60}")
    print(f"Análisis completado. Archivos en {OUTPUT_DIR}/")
    print(f"  - Gemini comparaciones sig: {sig_g}/15")
    print(f"  - Claude comparaciones sig: {sig_c}/15")
    print(f"  - LLaMA comparaciones sig:  {sig_l}/15")
    if len(both_cg) >= 3:
        print(f"  - Correlación Claude vs Gemini: ρ = {rho_cg:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
