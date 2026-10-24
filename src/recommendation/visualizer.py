"""Visualization module for recommendation benchmark metrics.

Generates:
1. Precision@K and Recall@K comparison bar chart.
2. NDCG@5 and NDCG@10 comparison bar chart.
3. Catalog coverage vs. recommendation diversity trade-off scatter plot.
"""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_recommendation_visualizations(
    benchmark_results: List[Dict[str, any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Render publication-grade comparison plots for recommender algorithms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    sns.set_theme(style="whitegrid", palette="muted")
    df = pd.DataFrame(benchmark_results)

    # 1. Precision@K & Recall@K Multi-Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    methods = df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.35

    # Subplot A: Precision & Recall at K=5
    axes[0].bar(x - width / 2, df["precision@5"], width, label="Precision@5", color="#2980b9")
    axes[0].bar(x + width / 2, df["recall@5"], width, label="Recall@5", color="#27ae60")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha="right", fontweight="bold")
    axes[0].set_title("Top-5 Recommendation Accuracy (Precision vs. Recall)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Metric Value")
    axes[0].set_ylim(0, max(df["recall@5"].max(), df["precision@5"].max(), 0.1) * 1.3)
    axes[0].legend(loc="upper left")
    for i in x:
        axes[0].text(i - width / 2, df["precision@5"][i] + 0.005, f"{df['precision@5'][i]:.3f}", ha="center", fontsize=8, fontweight="bold")
        axes[0].text(i + width / 2, df["recall@5"][i] + 0.005, f"{df['recall@5'][i]:.3f}", ha="center", fontsize=8, fontweight="bold")

    # Subplot B: Precision & Recall at K=10
    axes[1].bar(x - width / 2, df["precision@10"], width, label="Precision@10", color="#3498db")
    axes[1].bar(x + width / 2, df["recall@10"], width, label="Recall@10", color="#2ecc71")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha="right", fontweight="bold")
    axes[1].set_title("Top-10 Recommendation Accuracy (Precision vs. Recall)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Metric Value")
    axes[1].set_ylim(0, max(df["recall@10"].max(), df["precision@10"].max(), 0.1) * 1.3)
    axes[1].legend(loc="upper left")
    for i in x:
        axes[1].text(i - width / 2, df["precision@10"][i] + 0.005, f"{df['precision@10'][i]:.3f}", ha="center", fontsize=8, fontweight="bold")
        axes[1].text(i + width / 2, df["recall@10"][i] + 0.005, f"{df['recall@10'][i]:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    acc_chart_path = output_dir / "recommendation_precision_recall_comparison.png"
    plt.savefig(acc_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    results["precision_recall_chart"] = acc_chart_path

    # 2. Ranking Quality: NDCG Comparison
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(x - width / 2, df["ndcg@5"], width, label="NDCG@5", color="#8e44ad")
    ax.bar(x + width / 2, df["ndcg@10"], width, label="NDCG@10", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontweight="bold")
    ax.set_title("Ranking Quality: Normalized Discounted Cumulative Gain (NDCG)", fontsize=13, fontweight="bold")
    ax.set_ylabel("NDCG Score")
    ax.set_ylim(0, max(df["ndcg@10"].max(), 0.1) * 1.35)
    ax.legend(loc="upper left")
    for i in x:
        ax.text(i - width / 2, df["ndcg@5"][i] + 0.005, f"{df['ndcg@5'][i]:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(i + width / 2, df["ndcg@10"][i] + 0.005, f"{df['ndcg@10'][i]:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    ndcg_chart_path = output_dir / "recommendation_ndcg_ranking_comparison.png"
    plt.savefig(ndcg_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    results["ndcg_chart"] = ndcg_chart_path

    # 3. Coverage vs. Diversity Trade-off
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#e74c3c", "#27ae60", "#2980b9", "#f39c12", "#8e44ad", "#16a085"]
    for i, row in df.iterrows():
        ax.scatter(row["catalog_coverage"] * 100, row["mean_community_diversity"], s=280, color=colors[i % len(colors)], label=row["method"], edgecolors="#2c3e50", alpha=0.9)
        ax.annotate(row["method"], (row["catalog_coverage"] * 100 + 0.5, row["mean_community_diversity"]), fontsize=9, fontweight="bold")

    ax.set_title("Catalog Coverage vs. Community Recommendation Diversity", fontsize=13, fontweight="bold")
    ax.set_xlabel("Catalog Coverage (% of all 206 repositories ever recommended)")
    ax.set_ylabel("Mean Community Diversity Score")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    tradeoff_path = output_dir / "coverage_vs_diversity_tradeoff.png"
    plt.savefig(tradeoff_path, dpi=300, bbox_inches="tight")
    plt.close()
    results["coverage_diversity_chart"] = tradeoff_path

    return results
