"""Generate visualizations for narrative extraction experiment results."""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from analysis.statistics import load_results, cohens_d

plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 150})


def plot_coherence_by_method(df: pd.DataFrame, output_dir: str) -> None:
    """Box plot of average edge coherence grouped by method."""
    methods = sorted(df["method"].dropna().unique())
    data = [df[df["method"] == m]["avg_edge_coherence"].dropna().values for m in methods]

    fig, ax = plt.subplots()
    ax.boxplot(data, tick_labels=methods, patch_artist=True)
    ax.set_ylabel("Average Edge Coherence")
    ax.set_title("Edge Coherence by Extraction Method")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "coherence_boxplot.png"))
    plt.close(fig)


def plot_judge_scores(df: pd.DataFrame, output_dir: str) -> None:
    """Box plot of judge scores by method."""
    df_valid = df.dropna(subset=["judge_score"])
    if df_valid.empty:
        return

    methods = sorted(df_valid["method"].dropna().unique())
    data = [df_valid[df_valid["method"] == m]["judge_score"].values for m in methods]

    fig, ax = plt.subplots()
    ax.boxplot(data, tick_labels=methods, patch_artist=True)
    ax.set_ylabel("Judge Coherence Score (0-10)")
    ax.set_title("LLM Judge Scores by Method")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "judge_boxplot.png"))
    plt.close(fig)


def plot_coherence_by_n(df: pd.DataFrame, output_dir: str) -> None:
    """Grouped bar chart of coherence by method and narrative size n."""
    methods = sorted(df["method"].dropna().unique())
    sizes = sorted(df["n"].dropna().unique())

    x = np.arange(len(sizes))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots()
    for i, method in enumerate(methods):
        means = []
        for n_val in sizes:
            subset = df[(df["method"] == method) & (df["n"] == n_val)]
            means.append(subset["avg_edge_coherence"].mean() if len(subset) > 0 else 0)
        ax.bar(x + i * width, means, width, label=method)

    ax.set_xlabel("Narrative Size (n)")
    ax.set_ylabel("Avg Edge Coherence")
    ax.set_title("Coherence by Method and Narrative Size")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([str(int(s)) for s in sizes])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "coherence_by_n.png"))
    plt.close(fig)


def plot_quality_vs_time(df: pd.DataFrame, output_dir: str) -> None:
    """Scatter plot: quality (coherence) vs time, one point per method/n combo."""
    methods = sorted(df["method"].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    fig, ax = plt.subplots()
    for method, color in zip(methods, colors):
        subset = df[df["method"] == method]
        ax.scatter(
            subset["wall_time_seconds"],
            subset["avg_edge_coherence"],
            label=method, color=color, alpha=0.7, s=50,
        )

    ax.set_xlabel("Wall Time (seconds)")
    ax.set_ylabel("Avg Edge Coherence")
    ax.set_title("Quality vs. Computation Time (Pareto)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "quality_vs_time.png"))
    plt.close(fig)


def plot_effect_size_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Heatmap of pairwise Cohen's d for avg_edge_coherence."""
    methods = sorted(df["method"].dropna().unique())
    n = len(methods)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g1 = df[df["method"] == methods[i]]["avg_edge_coherence"].dropna().values
            g2 = df[df["method"] == methods[j]]["avg_edge_coherence"].dropna().values
            if len(g1) >= 2 and len(g2) >= 2:
                matrix[i, j] = cohens_d(g1, g2)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_title("Pairwise Cohen's d (Edge Coherence)")
    fig.colorbar(im)

    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "effect_size_heatmap.png"))
    plt.close(fig)


def _wrap_title(title: str, max_chars: int = 28) -> str:
    """Wrap a title into multiple lines for node labels."""
    if len(title) <= max_chars:
        return title
    words = title.split()
    lines = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return "\n".join(lines[:3])  # max 3 lines


def plot_narrative_map(result: dict, output_path: str) -> None:
    """Draw a narrative map as a directed graph from a maps result JSON."""
    graph_data = result.get("graph")
    if not graph_data:
        return

    G = nx.DiGraph()

    # Add nodes with date positions
    node_dates: dict[str, datetime] = {}
    for node in graph_data["nodes"]:
        nid = node["id"]
        G.add_node(nid, title=node["title"], date=node["date"])
        node_dates[nid] = datetime.fromisoformat(node["date"])

    # Add edges
    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], weight=edge["coherence_score"])

    # Identify path edges (from the narrative order)
    narrative_ids = [d["id"] for d in result.get("narrative", [])]
    path_edge_set = set()
    for k in range(len(narrative_ids) - 1):
        path_edge_set.add((narrative_ids[k], narrative_ids[k + 1]))

    # Vertical timeline layout (y = date, x = spread)
    # Sort dates top to bottom (earliest at top = highest y)
    dates_sorted = sorted(set(node_dates.values()))
    num_dates = len(dates_sorted)
    date_to_y = {d: (num_dates - 1 - i) * 2.0 for i, d in enumerate(dates_sorted)}

    date_groups: dict[datetime, list[str]] = defaultdict(list)
    for nid, d in node_dates.items():
        date_groups[d].append(nid)

    # Position nodes: path nodes along center, spread others horizontally
    path_set = set(narrative_ids)
    pos = {}
    for d, nodes in date_groups.items():
        y = date_to_y[d]
        # Separate path nodes and non-path nodes
        p_nodes = [n for n in nodes if n in path_set]
        other_nodes = [n for n in nodes if n not in path_set]
        all_ordered = p_nodes + other_nodes
        n_total = len(all_ordered)
        for k, nid in enumerate(all_ordered):
            x = (k - (n_total - 1) / 2) * 2.5
            pos[nid] = (x, y)

    # Add small horizontal jitter based on node position in path to avoid overlap
    for idx, nid in enumerate(narrative_ids):
        if nid in pos:
            x, y = pos[nid]
            # Slight zigzag: even nodes slightly left, odd slightly right
            offset = -0.3 if idx % 2 == 0 else 0.3
            pos[nid] = (x + offset, y)

    # Wrapped titles for labels (placed below nodes)
    labels = {}
    for nid in G.nodes:
        title = G.nodes[nid].get("title", nid)
        labels[nid] = _wrap_title(title, max_chars=30)

    # Separate path edges and cross-edges
    path_edges = [(u, v) for u, v in G.edges if (u, v) in path_edge_set]
    cross_edges = [(u, v) for u, v in G.edges if (u, v) not in path_edge_set]

    fig, ax = plt.subplots(figsize=(10, max(8, num_dates * 2.2)))

    # Draw cross-edges first (behind everything)
    if cross_edges:
        cross_weights = [G[u][v]["weight"] for u, v in cross_edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=cross_edges, ax=ax,
            edge_color=cross_weights, edge_cmap=plt.cm.OrRd,
            edge_vmin=0.3, edge_vmax=1.0,
            width=1.0, arrows=True, arrowsize=8, alpha=0.35,
            connectionstyle="arc3,rad=0.3",
            style="dashed",
        )

    # Draw path edges (thick, solid, prominent)
    if path_edges:
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax,
                               edge_color="#2b6cb0", width=2.8,
                               arrows=True, arrowsize=18,
                               connectionstyle="arc3,rad=0.08",
                               min_source_margin=18, min_target_margin=18)

    # Draw nodes
    node_list = list(G.nodes)
    node_colors = ["#bee3f8" if nid in path_set else "#e2e8f0" for nid in node_list]
    node_borders = ["#2b6cb0" if nid in path_set else "#a0aec0" for nid in node_list]
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, ax=ax,
                           node_size=900, node_color=node_colors,
                           edgecolors=node_borders, linewidths=2.0)

    # Draw labels offset below nodes
    label_pos = {nid: (x, y - 0.45) for nid, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels=labels, ax=ax,
                            font_size=6.5, font_weight="bold",
                            verticalalignment="top")

    # Date annotations on the right side
    for d in dates_sorted:
        y = date_to_y[d]
        ax.annotate(d.strftime("%Y-%m-%d"),
                    xy=(ax.get_xlim()[1] if ax.get_xlim()[1] != 0 else 3, y),
                    xytext=(15, 0), textcoords="offset points",
                    fontsize=8, color="#718096", ha="left", va="center")

    # Edge weight labels on path edges
    for u, v in path_edges:
        w = G[u][v]["weight"]
        mid_x = (pos[u][0] + pos[v][0]) / 2 + 0.4
        mid_y = (pos[u][1] + pos[v][1]) / 2
        ax.annotate(f"{w:.2f}", xy=(mid_x, mid_y), fontsize=7,
                    color="#2b6cb0", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # Title
    method = result.get("method", "maps")
    n_val = result.get("n", "?")
    model = result.get("model", "")
    title_str = f"Narrative Map — {method}"
    if model:
        title_str += f" ({model})"
    title_str += f", n={n_val}"
    avg_coh = result.get("metrics", {}).get("avg_edge_coherence")
    if avg_coh is not None:
        title_str += f"  |  avg coherence: {avg_coh:.3f}"
    ax.set_title(title_str, fontsize=12, fontweight="bold", pad=15)

    # Colorbar for cross-edges
    if cross_edges:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.OrRd,
                                   norm=plt.Normalize(vmin=0.3, vmax=1.0))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("Cross-edge Coherence", fontsize=9)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2b6cb0", lw=2.8, label="Narrative path"),
        Line2D([0], [0], color="#c53030", lw=1.0, alpha=0.5,
               linestyle="dashed", label="Cross-edges"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              framealpha=0.9)

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate experiment plots")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing result JSON files")
    parser.add_argument("--output-dir", default="analysis/plots",
                        help="Directory for plot images")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    plot_coherence_by_method(df, args.output_dir)
    plot_judge_scores(df, args.output_dir)
    plot_coherence_by_n(df, args.output_dir)
    plot_quality_vs_time(df, args.output_dir)
    plot_effect_size_heatmap(df, args.output_dir)

    # Generate narrative map visualizations for results with graph data
    map_count = 0
    for root, _dirs, files in os.walk(args.results_dir):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                if "graph" in data:
                    # Build unique name from path: maps_llm/llama3.2/n6/rep1.json → maps_llm_llama3.2_n6_rep1
                    rel = os.path.relpath(fpath, args.results_dir)
                    unique_name = rel.replace(os.sep, "_").replace(".json", "")
                    map_output = os.path.join(
                        args.output_dir,
                        f"narrative_map_{unique_name}.png",
                    )
                    plot_narrative_map(data, map_output)
                    map_count += 1
            except Exception:
                continue

    print(f"Plots saved to {args.output_dir}/")
    if map_count:
        print(f"  ({map_count} narrative map visualizations generated)")


if __name__ == "__main__":
    main()
