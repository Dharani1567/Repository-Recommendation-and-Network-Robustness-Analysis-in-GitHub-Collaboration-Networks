"""Automated unit tests for Part 6 Contributor Influence Analysis."""

import json
from pathlib import Path
import unittest
import networkx as nx
import pandas as pd

from src.config.settings import Config
from src.influence_analysis.analyzer import ContributorInfluenceAnalyzer


class TestInfluenceAnalysis(unittest.TestCase):
    """Validation test suite for contributor centrality metrics and removal groups."""

    @classmethod
    def setUpClass(cls):
        cls.inf_dir = Config.DATA_DIR / "influence_analysis"
        cls.rankings_path = cls.inf_dir / "contributor_influence_rankings.csv"
        cls.groups_path = cls.inf_dir / "contributor_removal_groups.csv"
        cls.metrics_path = cls.inf_dir / "influence_metrics.json"
        cls.original_graphml = Config.DATA_DIR / "network" / "collaboration_network.graphml"

        cls.df_rankings = pd.read_csv(cls.rankings_path)
        cls.df_groups = pd.read_csv(cls.groups_path)
        with open(cls.metrics_path, "r", encoding="utf-8") as f:
            cls.metrics = json.load(f)

    def test_original_network_remains_unchanged(self):
        """Verify original bipartite graph has not been mutated or modified."""
        G = nx.read_graphml(self.original_graphml)
        self.assertEqual(G.number_of_nodes(), 4965)
        self.assertEqual(G.number_of_edges(), 5758)

    def test_contributor_record_integrity(self):
        """Verify all 4,759 contributors are evaluated without duplicates."""
        self.assertEqual(len(self.df_rankings), 4759)
        self.assertEqual(self.df_rankings["contributor_id"].nunique(), 4759)
        self.assertEqual(self.df_rankings["contributor_node_id"].nunique(), 4759)

        # Check column completeness
        required_cols = [
            "contributor_id", "username", "degree", "weighted_degree",
            "betweenness_centrality", "pagerank", "k_core",
            "rank_degree", "rank_betweenness", "rank_pagerank", "rank_kcore"
        ]
        for col in required_cols:
            self.assertIn(col, self.df_rankings.columns)

    def test_removal_group_sizes_and_validity(self):
        """Verify removal groups contain only valid contributor IDs and match exact fractions."""
        total_contributors = 4759
        n_1pct = round(total_contributors * 0.01)  # 48
        n_5pct = round(total_contributors * 0.05)  # 238
        n_10pct = round(total_contributors * 0.10) # 476

        strategies = ["Degree Centrality", "Betweenness Centrality", "PageRank", "K-Core Centrality"]

        for strat in strategies:
            sub_1 = self.df_groups[(self.df_groups["removal_strategy"] == strat) & (self.df_groups["fraction"] == 0.01)]
            sub_5 = self.df_groups[(self.df_groups["removal_strategy"] == strat) & (self.df_groups["fraction"] == 0.05)]
            sub_10 = self.df_groups[(self.df_groups["removal_strategy"] == strat) & (self.df_groups["fraction"] == 0.10)]

            self.assertEqual(len(sub_1), n_1pct)
            self.assertEqual(len(sub_5), n_5pct)
            self.assertEqual(len(sub_10), n_10pct)

            # Check that all IDs exist in df_rankings
            valid_ids = set(self.df_rankings["contributor_id"])
            self.assertTrue(set(sub_10["contributor_id"]).issubset(valid_ids))

    def test_kcore_calculation_range(self):
        """Verify k-core numbers are positive integers."""
        kcore_vals = self.df_rankings["k_core"]
        self.assertTrue((kcore_vals >= 1).all())
        self.assertTrue((kcore_vals <= 10).all())


if __name__ == "__main__":
    unittest.main()
