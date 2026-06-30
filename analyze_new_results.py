#!/usr/bin/env python3
"""
Análisis estadístico de los resultados re-evaluados con Claude Sonnet 4.6.

Genera:
  a) Estadísticas descriptivas por método (judge_score_claude)
  b) Tests pairwise Welch t-test con Bonferroni + Cohen's d
  c) Matriz de Jaccard entre métodos
  d) Comparación juez antiguo vs nuevo (Spearman, varianza)

Uso:
    python analyze_new_results.py
"""

import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from run_experiments_local import calculate_jaccard_similarity

INPUT_JSON = "results_camino2_claude_judge.json"
OUTPUT_DIR = "analysis_v2"

# ============================================================
# CARGA
# ============================================================

def load_rejudged(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)

    rows = []
    for r in data:
        narrative_ids = [d["id"] for d in r.get("narrative", [])]
        rows.append({
            "method": r["method"],
            "model": r.get("model"),
            "n": r["n"],
            "start_index": r.get("start_index", 0),
            "judge_score_llama": r.get("judge_score_llama"),
            "judge_score_claude": r.get("judge_score_claude"),
            "judge_claude_justification": r.get("judge_claude_justification", ""),
            "avg_edge_coherence": r.get("metrics", {}).get("avg_edge_coherence"),
            "wall_time_seconds": r.get("metrics", {}).get("wall_time_seconds"),
            "api_cost_extraction": r.get("metrics", {}).get("api_cost_usd", 0.0),
            "api_cost_claude_judge": r.get("judge_claude_cost_usd", 0.0),
            "narrative_ids": narrative_ids,
        })

    df = pd.DataFrame(rows)
    # Limpiar scores inválidos
    for col in ["judge_score_llama", "judge_score_claude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    print(f"Cargados {len(df)} resultados")
    print(f"  Métodos: {sorted(df['method'].unique())}")
    print(f"  Claude scores válidos: {df['judge_score_claude'].notna().sum()}/{len(df)}")
    print(f"  LLaMA scores válidos: {df['judge_score_llama'].notna().sum()}/{len(df)}")
    return df


# ============================================================
# a) ESTADÍSTICAS DESCRIPTIVAS
# ============================================================

def descriptive_stats_claude(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("method").agg(
        mean=("judge_score_claude", "mean"),
        median=("judge_score_claude", "median"),
        std=("judge_score_claude", "std"),
        count=("judge_score_claude", "count"),
        valid=("judge_score_claude", lambda x: x.notna().sum()),
        failures=("judge_score_claude", lambda x: x.isna().sum()),
    ).round(4)
    return grouped


# ============================================================
# b) WELCH T-TEST PAIRWISE + BONFERRONI
# ============================================================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 +
                          (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
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


def pairwise_welch(df: pd.DataFrame, metric: str = "judge_score_claude") -> pd.DataFrame:
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
            "method_a": method_a,
            "method_b": method_b,
            "mean_a": round(np.mean(a), 4),
            "mean_b": round(np.mean(b), 4),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "p_bonferroni": round(p_bonf, 6),
            "cohens_d": round(d, 4),
            "effect_size": effect_size_label(d),
            "significant_bonferroni": p_bonf < 0.05,
            "n_a": len(a),
            "n_b": len(b),
        })

    return pd.DataFrame(rows)


# ============================================================
# c) MATRIZ DE JACCARD
# ============================================================

def jaccard_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Jaccard promedio entre métodos sobre narrativas con misma config (n, start_index)."""
    methods = sorted(df["method"].unique())
    matrix = pd.DataFrame(np.zeros((len(methods), len(methods))),
                          index=methods, columns=methods)
    counts = pd.DataFrame(np.zeros((len(methods), len(methods))),
                          index=methods, columns=methods)

    # Agrupar por configuración
    for (n, start), group in df.groupby(["n", "start_index"]):
        method_narratives = {}
        for _, row in group.iterrows():
            m = row["method"]
            ids = row["narrative_ids"]
            if m not in method_narratives:
                method_narratives[m] = []
            method_narratives[m].append(ids)

        for ma, mb in combinations(methods, 2):
            if ma not in method_narratives or mb not in method_narratives:
                continue
            # Promedio de Jaccard sobre todas las combinaciones de reps
            jaccards = []
            for ids_a in method_narratives[ma]:
                for ids_b in method_narratives[mb]:
                    jaccards.append(calculate_jaccard_similarity(ids_a, ids_b))
            if jaccards:
                avg_j = np.mean(jaccards)
                matrix.loc[ma, mb] += avg_j * len(jaccards)
                matrix.loc[mb, ma] += avg_j * len(jaccards)
                counts.loc[ma, mb] += len(jaccards)
                counts.loc[mb, ma] += len(jaccards)

    # Promediar
    for ma in methods:
        for mb in methods:
            if ma == mb:
                matrix.loc[ma, mb] = 1.0
            elif counts.loc[ma, mb] > 0:
                matrix.loc[ma, mb] /= counts.loc[ma, mb]

    return matrix.round(4)


# ============================================================
# d) COMPARACIÓN JUEZ ANTIGUO VS NUEVO
# ============================================================

def judge_comparison(df: pd.DataFrame) -> pd.DataFrame:
    methods = sorted(df["method"].unique())
    rows = []

    for method in methods:
        df_m = df[df["method"] == method]
        # Solo filas con ambos scores válidos
        valid = df_m.dropna(subset=["judge_score_llama", "judge_score_claude"])

        if len(valid) < 3:
            rows.append({
                "method": method,
                "n_valid": len(valid),
                "llama_mean": np.nan, "llama_std": np.nan,
                "claude_mean": np.nan, "claude_std": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan,
                "llama_variance": np.nan, "claude_variance": np.nan,
                "variance_ratio": np.nan,
            })
            continue

        llama = valid["judge_score_llama"].values
        claude = valid["judge_score_claude"].values

        rho, p_spearman = stats.spearmanr(llama, claude)

        rows.append({
            "method": method,
            "n_valid": len(valid),
            "llama_mean": round(np.mean(llama), 4),
            "llama_std": round(np.std(llama, ddof=1), 4),
            "claude_mean": round(np.mean(claude), 4),
            "claude_std": round(np.std(claude, ddof=1), 4),
            "spearman_rho": round(rho, 4),
            "spearman_p": round(p_spearman, 6),
            "llama_variance": round(np.var(llama, ddof=1), 4),
            "claude_variance": round(np.var(claude, ddof=1), 4),
            "variance_ratio": round(np.var(claude, ddof=1) / np.var(llama, ddof=1), 4)
                if np.var(llama, ddof=1) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"ERROR: No se encontró {INPUT_JSON}")
        print("  Ejecuta primero: python rejudge_camino2.py")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("ANÁLISIS ESTADÍSTICO — Claude Judge")
    print("=" * 60)

    df = load_rejudged(INPUT_JSON)

    # a) Descriptivas
    print("\n" + "=" * 60)
    print("a) ESTADÍSTICAS DESCRIPTIVAS (judge_score_claude)")
    print("=" * 60)
    desc = descriptive_stats_claude(df)
    print(desc.to_string())
    desc.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats_claude.csv"))

    # b) Pairwise Welch t-test
    print("\n" + "=" * 60)
    print("b) PAIRWISE WELCH T-TEST + BONFERRONI")
    print("=" * 60)
    pw = pairwise_welch(df, "judge_score_claude")
    display_cols = ["method_a", "method_b", "mean_a", "mean_b", "t_stat",
                    "p_value", "p_bonferroni", "cohens_d", "effect_size",
                    "significant_bonferroni"]
    print(pw[display_cols].to_string(index=False))

    sig = pw[pw["significant_bonferroni"]]
    print(f"\n  Diferencias significativas (Bonferroni p < 0.05): {len(sig)}/{len(pw)}")
    if not sig.empty:
        print("\n  Pares significativos:")
        for _, r in sig.iterrows():
            print(f"    {r['method_a']} vs {r['method_b']}: "
                  f"d={r['cohens_d']:.2f} ({r['effect_size']}), "
                  f"p_bonf={r['p_bonferroni']:.6f}")

    pw.to_csv(os.path.join(OUTPUT_DIR, "pairwise_welch_claude.csv"), index=False)

    # c) Matriz Jaccard
    print("\n" + "=" * 60)
    print("c) MATRIZ DE JACCARD ENTRE MÉTODOS")
    print("=" * 60)
    jac = jaccard_matrix(df)
    print(jac.to_string())
    jac.to_csv(os.path.join(OUTPUT_DIR, "jaccard_matrix.csv"))

    # d) Comparación jueces
    print("\n" + "=" * 60)
    print("d) COMPARACIÓN JUEZ LLAMA vs CLAUDE")
    print("=" * 60)
    jcomp = judge_comparison(df)
    print(jcomp.to_string(index=False))
    jcomp.to_csv(os.path.join(OUTPUT_DIR, "judge_comparison.csv"), index=False)

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Archivos generados en: {OUTPUT_DIR}/")
    print(f"  - descriptive_stats_claude.csv")
    print(f"  - pairwise_welch_claude.csv")
    print(f"  - jaccard_matrix.csv")
    print(f"  - judge_comparison.csv")

    # Hallazgos clave
    best = desc["mean"].idxmax()
    print(f"\n  Mejor método (Claude judge): {best} (mean={desc.loc[best, 'mean']:.2f})")
    print(f"  Comparaciones significativas: {len(sig)}/{len(pw)}")

    # Discriminación
    llama_vars = jcomp["llama_variance"].dropna()
    claude_vars = jcomp["claude_variance"].dropna()
    if not llama_vars.empty and not claude_vars.empty:
        print(f"\n  Varianza promedio LLaMA: {llama_vars.mean():.4f}")
        print(f"  Varianza promedio Claude: {claude_vars.mean():.4f}")
        if claude_vars.mean() > llama_vars.mean():
            print("  -> Claude discrimina MÁS que LLaMA (mayor varianza en scores)")
        else:
            print("  -> Claude discrimina MENOS que LLaMA")

    print("=" * 60)


if __name__ == "__main__":
    main()
