"""Visualization module for contributor influence analysis.

Generates:
1. Spearman Rank Correlation heatmap between all 5 centrality measures.
2. Contributor influence distribution plots (histograms/KDEs of Degree, Weighted Degree, Betweenness, PageRank).
3. Top contributor leaderboard comparison plot.
"""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_influence_visualizations(
    df_rankings: pd.DataFrame,
    corr_matrix: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Render publication-grade plots for contributor influence metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # 1. Spearman Rank Correlation Heatmap
    corr_df = pd.DataFrame(corr_matrix)
    readable_labels = {
        "degree": "Degree (Repos)",
        "weighted_degree": "Weighted Degree (Commits)",
        "betweenness_centrality": "Betweenness",
        "pagerank": "PageRank",
        "k_core": "K-Core",
    }
    corr_df.rename(index=readable_labels, columns=readable_labels, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        linewidths=1.0,
        cbar_kws={"label": "Spearman Rank Correlation (ρ)"},
        ax=ax,
    )
    ax.set_title("Spearman Rank Correlation Across Contributor Centrality Measures", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    corr_path = output_dir / "influence_ranking_comparison.png"
    plt.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close()
    results["correlation_heatmap"] = corr_path

    # 2. Influence Distribution (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Subplot A: Degree Distribution
    deg_series = df_rankings["degree"]
    axes[0, 0].hist(deg_series, bins=range(1, deg_series.max() + 2), color="#3498db", edgecolor="#2980b9", align="left")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Degree Distribution (Number of Repositories)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Degree (Repositories)")
    axes[0, 0].set_ylabel("Contributor Count (Log Scale)")

    # Subplot B: Weighted Degree (Commits)
    wdeg_series = df_rankings["weighted_degree"]
    axes[0, 1].hist(wdeg_series, bins=50, color="#2ecc71", edgecolor="#27ae60")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Weighted Degree Distribution (Commits Volume)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Commits Across Repositories")
    axes[0, 1].set_ylabel("Contributor Count (Log Scale)")

    # Subplot C: Betweenness Centrality
    betw_non_zero = df_rankings[df_rankings["betweenness_centrality"] > 0]["betweenness_centrality"]
    axes[1, 0].hist(betw_non_zero, bins=40, color="#e67e22", edgecolor="#d35400")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title(f"Betweenness Centrality (Non-Zero Nodes: {len(betw_non_zero)})", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Betweenness Centrality")
    axes[1, 0].set_ylabel("Contributor Count (Log Scale)")

    # Subplot D: PageRank Distribution
    pr_series = df_rankings["pagerank"]
    axes[1, 1].hist(pr_series, bins=50, color="#9b59b6", edgecolor="#8e44ad")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("PageRank Distribution", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("PageRank Score")
    axes[1, 1].set_ylabel("Contributor Count (Log Scale)")

    plt.tight_layout()
    dist_path = output_dir / "influence_distribution.png"
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    results["distribution_plot"] = dist_path

    return results
