"""Attack simulation and node removal coordinator for Part 7.

Defines:
- Targeted contributor removal orderings:
  1. Degree Centrality
  2. Betweenness Centrality
  3. PageRank
  4. K-Core Centrality (with betweenness as tie-breaker)
  5. Weighted Degree (Commit counts)
- Random-removal baseline with multiple seeds.
- Safe graph mutation that extracts remaining subgraphs without altering baseline files.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class RemovalExperimentConfig:
    strategy_name: str
    fraction: float
    fraction_label: str
    removed_count: int
    seed: Optional[int] = None


class ContributorAttackSimulator:
    """Simulates targeted and random removal of contributor nodes from a collaboration network."""

    def __init__(self, df_rankings: pd.DataFrame):
        self.df_rankings = df_rankings.copy()
        self.total_contributors = len(df_rankings)

        # Pre-sort rankings for each targeted strategy
        self.sorted_rankings = {
            "degree": self.df_rankings.sort_values(by=["degree", "weighted_degree"], ascending=False),
            "betweenness": self.df_rankings.sort_values(by="betweenness_centrality", ascending=False),
            "pagerank": self.df_rankings.sort_values(by="pagerank", ascending=False),
            "k_core": self.df_rankings.sort_values(by=["k_core", "betweenness_centrality"], ascending=False),
            "weighted_degree": self.df_rankings.sort_values(by="weighted_degree", ascending=False),
        }

    def get_targeted_removed_nodes(self, strategy_key: str, fraction: float) -> Set[str]:
        """Return the exact set of contributor_node_ids to remove for a targeted fraction."""
        if strategy_key not in self.sorted_rankings:
            raise ValueError(f"Unknown strategy: {strategy_key}. Expected: {list(self.sorted_rankings.keys())}")

        k = max(1, int(round(self.total_contributors * fraction)))
        top_df = self.sorted_rankings[strategy_key].head(k)
        return set(top_df["contributor_node_id"])

    def get_random_removed_nodes(self, fraction: float, seed: int) -> Set[str]:
        """Return random set of contributor_node_ids for random removal baseline."""
        k = max(1, int(round(self.total_contributors * fraction)))
        rng = np.random.default_rng(seed)
        all_c_nodes = self.df_rankings["contributor_node_id"].values
        chosen = rng.choice(all_c_nodes, size=k, replace=False)
        return set(chosen)

    @staticmethod
    def apply_removal_to_graph(graph: nx.Graph, removed_nodes: Set[str]) -> nx.Graph:
        """Create a new graph with specified nodes removed, preserving original graph."""
        remaining_nodes = [n for n in graph.nodes() if n not in removed_nodes]
        return graph.subgraph(remaining_nodes).copy()
