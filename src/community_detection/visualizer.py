"""Community Detection Visualizations for Leiden vs. Louvain.

Generates:
1. Side-by-side or dual panel community graph visualizations showing discovered clusters.
2. Community size distribution chart.
3. Algorithm comparison bar chart (Modularity, Stability ARI, NMI, Community Counts).
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def generate_community_visualizations(
    df_comms: pd.DataFrame,
    repo_graph: nx.Graph,
    metrics: Dict[str, any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate publication-ready figures comparing Leiden and Louvain communities."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Set aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # 1. Algorithm Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Subplot A: Modularity Q
    mods = [metrics["louvain"]["modularity_score"], metrics["leiden"]["modularity_score"]]
    bars1 = axes[0].bar(["Louvain", "Leiden"], mods, color=["#3498db", "#2ecc71"], width=0.5, edgecolor="#2c3e50")
    axes[0].set_title("Modularity Score (Q)\n(Higher = Stronger Community Structure)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Modularity Q")
    axes[0].set_ylim(0, max(mods) * 1.25)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.015, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")

    # Subplot B: Stability Across 5 Random Seeds (ARI)
    aris = [metrics["louvain"]["stability_ari_5seeds"], metrics["leiden"]["stability_ari_5seeds"]]
    bars2 = axes[1].bar(["Louvain", "Leiden"], aris, color=["#3498db", "#2ecc71"], width=0.5, edgecolor="#2c3e50")
    axes[1].set_title("Partition Stability across Seeds (ARI)\n(1.0 = Perfectly Deterministic)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Mean Adjusted Rand Index")
    axes[1].set_ylim(0, 1.15)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")

    # Subplot C: Community Count & Isolated
    counts_data = {
        "Algorithm": ["Louvain", "Leiden"],
        "Detected Communities": [metrics["louvain"]["num_communities"], metrics["leiden"]["num_communities"]],
        "Isolated Repos": [metrics["louvain"]["num_isolated"], metrics["leiden"]["num_isolated"]],
    }
    df_counts = pd.DataFrame(counts_data)
    x = np.arange(len(df_counts))
    width = 0.35
    axes[2].bar(x - width / 2, df_counts["Detected Communities"], width, label="Detected Communities", color="#9b59b6")
    axes[2].bar(x + width / 2, df_counts["Isolated Repos"], width, label="Isolated Repos (Cluster -1)", color="#e74c3c")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df_counts["Algorithm"], fontweight="bold")
    axes[2].set_title("Community Count & Isolated Repositories", fontsize=12, fontweight="bold")
    axes[2].legend()
    axes[2].set_ylim(0, max(metrics["louvain"]["num_communities"], metrics["leiden"]["num_communities"]) * 1.3)
    for i in x:
        axes[2].text(i - width / 2, df_counts["Detected Communities"][i] + 0.3, str(df_counts["Detected Communities"][i]), ha="center", fontweight="bold")
        axes[2].text(i + width / 2, df_counts["Isolated Repos"][i] + 0.3, str(df_counts["Isolated Repos"][i]), ha="center", fontweight="bold")

    plt.tight_layout()
    comp_chart_path = output_dir / "algorithm_comparison_metrics.png"
    plt.savefig(comp_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    results["comparison_chart"] = comp_chart_path

    # 2. Community Size Distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    louvain_sizes = metrics["louvain"]["community_sizes"]
    leiden_sizes = metrics["leiden"]["community_sizes"]

    axes[0].bar(range(1, len(louvain_sizes) + 1), louvain_sizes, color="#3498db", edgecolor="#2980b9")
    axes[0].set_title(f"Louvain Community Sizes (N = {sum(louvain_sizes)} in {len(louvain_sizes)} communities)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Community ID (Ranked by Size)")
    axes[0].set_ylabel("Number of Repositories")
    axes[0].set_xticks(range(1, len(louvain_sizes) + 1))
    for i, v in enumerate(louvain_sizes):
        axes[0].text(i + 1, v + 0.8, str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[1].bar(range(1, len(leiden_sizes) + 1), leiden_sizes, color="#2ecc71", edgecolor="#27ae60")
    axes[1].set_title(f"Leiden Community Sizes (N = {sum(leiden_sizes)} in {len(leiden_sizes)} communities)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Community ID (Ranked by Size)")
    axes[1].set_ylabel("Number of Repositories")
    axes[1].set_xticks(range(1, len(leiden_sizes) + 1))
    for i, v in enumerate(leiden_sizes):
        axes[1].text(i + 1, v + 0.8, str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    size_dist_path = output_dir / "community_size_distribution.png"
    plt.savefig(size_dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    results["size_distribution"] = size_dist_path

    # 3. Community Graph Visualization (Core Network Projection Subgraph)
    # Filter to connected component with high degree to make labels readable
    connected_repos = df_comms[df_comms["is_isolated"] == False]
    top_repos = connected_repos.sort_values(by="degree_projection", ascending=False).head(60)["node_id"].tolist()
    sub_proj = repo_graph.subgraph(top_repos).copy()

    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)
    pos = nx.spring_layout(sub_proj, k=0.55, iterations=90, seed=42)

    # Color nodes by Leiden community
    leiden_comm_map = dict(zip(df_comms["node_id"], df_comms["leiden_community_id"]))
    node_communities = [leiden_comm_map.get(n, -1) for n in sub_proj.nodes()]

    # Unique colors per community
    palette = sns.color_palette("tab10", n_colors=max(node_communities) + 1)
    node_colors = [palette[c % len(palette)] if c != -1 else "#95a5a6" for c in node_communities]
    node_sizes = [max(300, sub_proj.degree(n) * 45) for n in sub_proj.nodes()]

    # Draw edges with opacity based on weight
    edge_weights = [d.get("weight", 1.0) for u, v, d in sub_proj.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths = [1.0 + (w / max_w) * 3.5 for w in edge_weights]

    nx.draw_networkx_edges(sub_proj, pos, ax=ax, alpha=0.35, edge_color="#7f8c8d", width=edge_widths)
    nx.draw_networkx_nodes(
        sub_proj,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        edgecolors="#2c3e50",
        linewidths=1.5,
    )

    # Labels for repositories
    labels = {n: df_comms.loc[df_comms["node_id"] == n, "repo_name"].values[0] for n in sub_proj.nodes()}
    nx.draw_networkx_labels(
        sub_proj,
        pos,
        labels=labels,
        ax=ax,
        font_size=8,
        font_weight="bold",
        font_color="#1a252f",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.85),
    )

    ax.set_title(
        f"Leiden Community Detection on GitHub Repository Co-Contribution Network\n"
        f"(Top 60 Interconnected Repositories Colored by Community Partition, Modularity Q = {metrics['leiden']['modularity_score']})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.axis("off")
    plt.tight_layout()

    network_viz_path = output_dir / "repository_communities_network.png"
    plt.savefig(network_viz_path, dpi=300, bbox_inches="tight")
    plt.close()
    results["network_visualization"] = network_viz_path

    return results
