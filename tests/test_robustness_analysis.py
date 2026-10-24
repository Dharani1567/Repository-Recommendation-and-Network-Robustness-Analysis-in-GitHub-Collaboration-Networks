"""Unit tests for Part 7 Network Robustness and Contributor Removal Simulation."""

import json
from pathlib import Path
import unittest
import networkx as nx
import pandas as pd

from src.config.settings import Config
from src.recommendation.data_splitter import create_train_test_split
from src.robustness_analysis.attack_simulator import ContributorAttackSimulator


class TestNetworkRobustness(unittest.TestCase):
    """Validation test suite for robustness attack simulation, metrics, and leakage controls."""

    @classmethod
    def setUpClass(cls):
        cls.rob_dir = Config.DATA_DIR / "robustness_analysis"
        cls.struct_csv = cls.rob_dir / "robustness_metrics.csv"
        cls.recs_csv = cls.rob_dir / "recommendation_robustness.csv"
        cls.comms_csv = cls.rob_dir / "community_robustness.csv"
        cls.rand_csv = cls.rob_dir / "random_removal_results.csv"
        cls.summary_json = cls.rob_dir / "robustness_summary.json"
        cls.original_graphml = Config.DATA_DIR / "network" / "collaboration_network.graphml"

        cls.df_struct = pd.read_csv(cls.struct_csv)
        cls.df_recs = pd.read_csv(cls.recs_csv)
        cls.df_comms = pd.read_csv(cls.comms_csv)
        cls.df_rand = pd.read_csv(cls.rand_csv)

        cls.G_orig = nx.read_graphml(cls.original_graphml)
        df_rankings = pd.read_csv(Config.DATA_DIR / "influence_analysis" / "contributor_influence_rankings.csv")
        cls.simulator = ContributorAttackSimulator(df_rankings)

    def test_original_graph_remains_unchanged(self):
        """Verify baseline network graphml has exact 4,965 nodes and 5,758 edges."""
        self.assertEqual(self.G_orig.number_of_nodes(), 4965)
        self.assertEqual(self.G_orig.number_of_edges(), 5758)

    def test_removed_nodes_absent_from_mutated_graphs(self):
        """Verify that nodes selected for removal do not exist in post-removal graph."""
        removed_nodes = self.simulator.get_targeted_removed_nodes("degree", fraction=0.01)
        self.assertEqual(len(removed_nodes), 48)

        G_post = self.simulator.apply_removal_to_graph(self.G_orig, removed_nodes)
        self.assertEqual(G_post.number_of_nodes(), 4965 - 48)

        # Confirm zero removed nodes are present
        for node in removed_nodes:
            self.assertNotIn(node, G_post)

    def test_only_contributor_nodes_removed(self):
        """Verify that removal targets ONLY contributor nodes and never repository nodes."""
        for strat in ["degree", "betweenness", "pagerank", "k_core", "weighted_degree"]:
            removed_nodes = self.simulator.get_targeted_removed_nodes(strat, fraction=0.05)
            for node in removed_nodes:
                self.assertTrue(node.startswith("C_"), f"Node {node} is not a contributor node!")
                self.assertEqual(self.G_orig.nodes[node].get("node_type"), "contributor")

    def test_removal_counts_match_fractions(self):
        """Verify removal counts match exact fractions (48, 238, 476)."""
        sub = self.df_struct[self.df_struct["strategy"] == "Degree Centrality"]
        self.assertEqual(sub.loc[sub["removal_pct"] == 1.0, "removed_count"].values[0], 48)
        self.assertEqual(sub.loc[sub["removal_pct"] == 5.0, "removed_count"].values[0], 238)
        self.assertEqual(sub.loc[sub["removal_pct"] == 10.0, "removed_count"].values[0], 476)

    def test_random_removal_reproducibility(self):
        """Verify random removal selection is deterministic given fixed seed."""
        set1 = self.simulator.get_random_removed_nodes(0.01, seed=42)
        set2 = self.simulator.get_random_removed_nodes(0.01, seed=42)
        self.assertEqual(set1, set2)

        set3 = self.simulator.get_random_removed_nodes(0.01, seed=100)
        self.assertNotEqual(set1, set3)

    def test_recommendation_degradation_recorded(self):
        """Verify recommendation metrics are recorded and baseline matches Part 5."""
        cf_base = self.df_recs[(self.df_recs["strategy"] == "Baseline (No Removal)") & (self.df_recs["model"] == "Collaborative Filtering")]
        self.assertAlmostEqual(cf_base["baseline_recall@10"].values[0], 0.3790, places=3)
        self.assertAlmostEqual(cf_base["baseline_ndcg@10"].values[0], 0.2589, places=3)


if __name__ == "__main__":
    unittest.main()
