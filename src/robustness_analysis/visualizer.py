"""Visualization module for network robustness and recommendation degradation."""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_robustness_visualizations(
    df_struct: pd.DataFrame,
    df_recs: pd.DataFrame,
    df_random: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """Render publication-grade plots comparing targeted attacks vs random failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    strategy_palette = {
        "Degree Centrality": "#e74c3c",
        "Betweenness Centrality": "#2980b9",
        "PageRank": "#8e44ad",
        "K-Core Centrality": "#d35400",
        "Weighted Degree": "#27ae60",
        "Random Removal (Mean)": "#7f8c8d",
    }

    # 1. Structural Metric: GCC Fraction vs Removal Percentage
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for strat in df_struct["strategy"].unique():
        sub = df_struct[df_struct["strategy"] == strat].sort_values(by="removal_pct")
        col = strategy_palette.get(strat, "#34495e")
        style = "--o" if "Random" in strat else "-o"
        lw = 2.5 if "Random" not in strat else 2.0
        ax.plot(sub["removal_pct"], sub["gcc_fraction"] * 100, style, label=strat, color=col, linewidth=lw, markersize=7)

    ax.set_title("Network Structural Robustness: Giant Connected Component (GCC) vs. Removal %", fontsize=12, fontweight="bold")
    ax.set_xlabel("Contributor Removal Percentage (%)", fontsize=11)
    ax.set_ylabel("Largest Connected Component (% of Initial Network)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    p1 = output_dir / "network_structural_metrics_vs_removal.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    results["structural_gcc_plot"] = p1

    # 2. Recommendation Degradation: Recall@10 vs Removal Percentage (Collaborative Filtering)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    cf_recs = df_recs[df_recs["model"] == "Collaborative Filtering"]
    for strat in cf_recs["strategy"].unique():
        sub = cf_recs[cf_recs["strategy"] == strat].sort_values(by="removal_pct")
        col = strategy_palette.get(strat, "#34495e")
        style = "--o" if "Random" in strat else "-s"
        lw = 2.5 if "Random" not in strat else 2.0
        ax.plot(sub["removal_pct"], sub["recall@10"] * 100, style, label=strat, color=col, linewidth=lw, markersize=7)

    ax.set_title("Collaborative Filtering Recommendation Degradation: Recall@10 vs. Removal %", fontsize=12, fontweight="bold")
    ax.set_xlabel("Contributor Removal Percentage (%)", fontsize=11)
    ax.set_ylabel("Recall@10 on Surviving Cohort (%)", fontsize=11)
    ax.set_ylim(0, max(cf_recs["recall@10"] * 100) * 1.3)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    p2 = output_dir / "recall10_vs_removal_percentage.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    results["recall10_degradation_plot"] = p2

    # 3. NDCG@10 vs Removal Percentage (Collaborative Filtering)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for strat in cf_recs["strategy"].unique():
        sub = cf_recs[cf_recs["strategy"] == strat].sort_values(by="removal_pct")
        col = strategy_palette.get(strat, "#34495e")
        style = "--o" if "Random" in strat else "-^"
        lw = 2.5 if "Random" not in strat else 2.0
        ax.plot(sub["removal_pct"], sub["ndcg@10"], style, label=strat, color=col, linewidth=lw, markersize=7)

    ax.set_title("Collaborative Filtering Ranking Quality: NDCG@10 vs. Removal %", fontsize=12, fontweight="bold")
    ax.set_xlabel("Contributor Removal Percentage (%)", fontsize=11)
    ax.set_ylabel("NDCG@10 on Surviving Cohort", fontsize=11)
    ax.set_ylim(0, max(cf_recs["ndcg@10"]) * 1.3)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    p3 = output_dir / "ndcg10_vs_removal_percentage.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    results["ndcg10_degradation_plot"] = p3

    # 4. Targeted vs Random Removal Comparison Bar Chart (at 10% removal)
    sub_10 = cf_recs[cf_recs["removal_pct"] == 10.0].sort_values(by="recall@10", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.barh(sub_10["strategy"], sub_10["recall@10"] * 100, color=[strategy_palette.get(s, "#3498db") for s in sub_10["strategy"]], edgecolor="#2c3e50")
    ax.set_title("Recall@10 Comparison at 10% Contributor Removal (Targeted vs. Random)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Surviving Recall@10 (%)", fontsize=11)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    p4 = output_dir / "targeted_vs_random_removal_comparison.png"
    plt.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close()
    results["targeted_vs_random_plot"] = p4

    # 5. Algorithm Comparison under Degree Attack (CF vs Leiden Community-Aware vs Jaccard vs Adamic-Adar)
    deg_recs = df_recs[df_recs["strategy"] == "Degree Centrality"]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    model_colors = {
        "Collaborative Filtering": "#2980b9",
        "Leiden Community-Aware": "#27ae60",
        "Jaccard Similarity": "#e67e22",
        "Adamic-Adar": "#8e44ad",
        "Popularity Baseline": "#95a5a6",
    }
    for model in deg_recs["model"].unique():
        sub = deg_recs[deg_recs["model"] == model].sort_values(by="removal_pct")
        col = model_colors.get(model, "#34495e")
        ax.plot(sub["removal_pct"], sub["recall@10"] * 100, "-o", label=model, color=col, linewidth=2.2, markersize=6)

    ax.set_title("Multi-Model Recommendation Resilience under Targeted Degree Attack", fontsize=12, fontweight="bold")
    ax.set_xlabel("Targeted Degree Removal Percentage (%)", fontsize=11)
    ax.set_ylabel("Recall@10 on Surviving Cohort (%)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    p5 = output_dir / "centrality_strategy_comparison.png"
    plt.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close()
    results["multi_model_attack_plot"] = p5

    # 6. Network Fragmentation vs Recommendation Degradation Scatter Plot
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    # Merge structural fragmentation with CF recall delta
    merged = pd.merge(
        df_struct,
        cf_recs,
        on=["strategy", "removal_pct", "fraction"],
        suffixes=("_struct", "_rec")
    )
    scatter = ax.scatter(
        merged["fragmentation_index"] * 100,
        merged["abs_change_recall@10"] * 100,
        s=120,
        c=merged["removal_pct"],
        cmap="viridis",
        edgecolors="#2c3e50",
        alpha=0.9,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Removal Percentage (%)", fontsize=10)
    ax.set_title("Structural Network Fragmentation vs. Recommendation Degradation (Recall@10 Delta)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Network Fragmentation (% Nodes Disconnected from GCC)", fontsize=11)
    ax.set_ylabel("Absolute Drop in Recall@10 (% Points)", fontsize=11)
    plt.tight_layout()
    p6 = output_dir / "fragmentation_vs_recommendation_degradation.png"
    plt.savefig(p6, dpi=300, bbox_inches="tight")
    plt.close()
    results["fragmentation_vs_degradation_plot"] = p6

    return results
