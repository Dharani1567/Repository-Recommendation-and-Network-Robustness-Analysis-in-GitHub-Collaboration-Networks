"""Contributor Influence and Centrality Analysis Module.

Calculates and analyzes multiple network centrality and influence metrics for contributors:
1. Degree Centrality (Breadth of repository participation)
2. Weighted Degree (Volume of commits across repositories)
3. Exact Betweenness Centrality (Bridge role between network modules)
4. PageRank (Prestige and authority in collaboration flow)
5. K-Core Decomposition (Structural embedding in dense cores)
6. Community Influence Analysis (Cross-community bridging and concentration)
"""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config.settings import Config

logger = logging.getLogger(__name__)


@dataclass
class ContributorInfluenceRecord:
    """Stores all centrality and community influence scores for a single contributor."""

    contributor_node_id: str
    contributor_id: int
    username: str
    degree: int
    weighted_degree: float
    betweenness_centrality: float
    pagerank: float
    k_core: int
    repositories_count: int
    communities_count: int
    dominant_community_id: int
    community_entropy: float
    is_multi_repo: bool
    is_cross_community: bool


class ContributorInfluenceAnalyzer:
    """Computes, benchmarks, and groups contributor influence metrics across the bipartite network."""

    def __init__(
        self,
        bipartite_graph: nx.Graph,
        repo_communities: Dict[str, int],
    ):
        self.bipartite_graph = bipartite_graph
        self.repo_communities = repo_communities

        self.c_nodes = [
            n for n, d in bipartite_graph.nodes(data=True)
            if d.get("node_type") == "contributor" or d.get("bipartite") == 0
        ]
        self.r_nodes = [
            n for n, d in bipartite_graph.nodes(data=True)
            if d.get("node_type") == "repository" or d.get("bipartite") == 1
        ]

    def compute_all_metrics(self) -> pd.DataFrame:
        """Compute all 5 centrality measures and community cross-cutting metrics for all contributors."""
        logger.info("Computing Degree and Weighted Degree...")
        # 1. Degree & Weighted Degree
        degrees = {c: self.bipartite_graph.degree(c) for c in self.c_nodes}
        weighted_degrees = {
            c: sum(d.get("weight", 1.0) for _, _, d in self.bipartite_graph.edges(c, data=True))
            for c in self.c_nodes
        }

        # 2. Exact Normalized Betweenness Centrality via igraph (C-optimized, exact)
        logger.info("Computing exact Betweenness Centrality...")
        node_to_idx = {n: i for i, n in enumerate(self.bipartite_graph.nodes())}
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in self.bipartite_graph.edges()]
        ig_g = ig.Graph(n=len(self.bipartite_graph), edges=edges, directed=False)
        raw_betw = ig_g.betweenness()

        # Normalization factor for undirected graph: 2 / ((N - 1) * (N - 2))
        n_total = len(self.bipartite_graph)
        norm_factor = 2.0 / ((n_total - 1) * (n_total - 2)) if n_total > 2 else 1.0
        c_betw = {c: raw_betw[node_to_idx[c]] * norm_factor for c in self.c_nodes}

        # 3. PageRank on Bipartite Collaboration Graph
        logger.info("Computing PageRank...")
        pr = nx.pagerank(self.bipartite_graph, weight="weight", alpha=0.85, max_iter=200)
        c_pr = {c: pr[c] for c in self.c_nodes}

        # 4. K-Core Decomposition
        logger.info("Computing K-Core Decomposition...")
        core_numbers = nx.core_number(self.bipartite_graph)
        c_kcore = {c: core_numbers[c] for c in self.c_nodes}

        # 5. Community Participation & Bridging Metrics
        logger.info("Analyzing Cross-Community Participation...")
        records: List[Dict[str, Any]] = []

        for c in self.c_nodes:
            cid = self.bipartite_graph.nodes[c].get("contributor_id")
            uname = self.bipartite_graph.nodes[c].get("username", c)
            r_neighbors = list(self.bipartite_graph.neighbors(c))

            # Contributor's communities from their touched repositories
            comm_list = [self.repo_communities.get(r, -1) for r in r_neighbors]
            distinct_comms = set(comm_list) - {-1}  # Ignore isolated cluster -1
            comm_counts = Counter(comm_list)

            dominant_comm = comm_counts.most_common(1)[0][0] if comm_counts else -1

            # Community entropy: -sum(p * log(p))
            total_r = len(r_neighbors)
            entropy = 0.0
            if total_r > 1 and len(distinct_comms) > 1:
                for cnt in comm_counts.values():
                    p = cnt / total_r
                    entropy -= p * np.log2(p)

            deg = degrees[c]
            wdeg = weighted_degrees[c]
            bw = c_betw[c]
            pgr = c_pr[c]
            kcr = c_kcore[c]

            records.append({
                "contributor_node_id": c,
                "contributor_id": int(cid) if cid is not None else 0,
                "username": uname,
                "degree": deg,
                "weighted_degree": wdeg,
                "betweenness_centrality": bw,
                "pagerank": pgr,
                "k_core": kcr,
                "repositories_count": deg,
                "communities_count": len(distinct_comms),
                "dominant_community_id": dominant_comm,
                "community_entropy": round(float(entropy), 4),
                "is_multi_repo": deg > 1,
                "is_cross_community": len(distinct_comms) > 1,
            })

        df = pd.DataFrame(records)

        # Compute ordinal ranks (1 = highest influence)
        df["rank_degree"] = df["degree"].rank(ascending=False, method="min").astype(int)
        df["rank_weighted_degree"] = df["weighted_degree"].rank(ascending=False, method="min").astype(int)
        df["rank_betweenness"] = df["betweenness_centrality"].rank(ascending=False, method="min").astype(int)
        df["rank_pagerank"] = df["pagerank"].rank(ascending=False, method="min").astype(int)
        df["rank_kcore"] = df["k_core"].rank(ascending=False, method="min").astype(int)

        return df

    @staticmethod
    def compute_spearman_correlations(df_metrics: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate pairwise Spearman rank correlations between all 5 centrality measures."""
        metrics = ["degree", "weighted_degree", "betweenness_centrality", "pagerank", "k_core"]
        corr_matrix: Dict[str, Dict[str, float]] = {m: {} for m in metrics}

        for m1 in metrics:
            for m2 in metrics:
                corr, _ = spearmanr(df_metrics[m1], df_metrics[m2])
                corr_matrix[m1][m2] = round(float(corr), 4)

        return corr_matrix

    @staticmethod
    def create_removal_groups(
        df_metrics: pd.DataFrame,
        fractions: List[float] = [0.01, 0.05, 0.10],
    ) -> pd.DataFrame:
        """Construct structured removal candidate groups for Part 7 robustness attack simulations.
        
        Selection Protocol:
        - Strict ranking by: Degree, Betweenness, PageRank, K-Core (with Betweenness as tie-breaker for k-core).
        - Fractions: Top 1%, Top 5%, Top 10% of all contributors.
        - Preserves contributor_id, username, and metric values.
        """
        n_total = len(df_metrics)
        group_rows = []

        ranking_cols = [
            ("degree", "Degree Centrality"),
            ("betweenness_centrality", "Betweenness Centrality"),
            ("pagerank", "PageRank"),
            ("k_core", "K-Core Centrality"),
        ]

        for col, col_name in ranking_cols:
            if col == "k_core":
                sorted_df = df_metrics.sort_values(by=["k_core", "betweenness_centrality"], ascending=False)
            else:
                sorted_df = df_metrics.sort_values(by=col, ascending=False)

            for frac in fractions:
                k_count = max(1, int(round(n_total * frac)))
                top_nodes = sorted_df.head(k_count)

                for rank_idx, (_, row) in enumerate(top_nodes.iterrows(), 1):
                    group_rows.append({
                        "removal_strategy": col_name,
                        "fraction": frac,
                        "fraction_label": f"Top {int(frac*100)}%",
                        "target_rank": rank_idx,
                        "contributor_node_id": row["contributor_node_id"],
                        "contributor_id": row["contributor_id"],
                        "username": row["username"],
                        "metric_name": col,
                        "metric_value": row[col],
                        "degree": row["degree"],
                        "betweenness": row["betweenness_centrality"],
                        "pagerank": row["pagerank"],
                        "k_core": row["k_core"],
                    })

        return pd.DataFrame(group_rows)
