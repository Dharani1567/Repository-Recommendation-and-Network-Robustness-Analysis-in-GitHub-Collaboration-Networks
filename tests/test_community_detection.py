"""Automated validation tests for Part 4 Community Detection (Leiden vs. Louvain)."""

import json
from pathlib import Path
import unittest
import networkx as nx
import pandas as pd

from src.config.settings import Config
from src.community_detection.detector import RepositoryCommunityDetector


class TestCommunityDetection(unittest.TestCase):
    """Validation test suite for community detection outputs and integrity."""

    @classmethod
    def setUpClass(cls):
        cls.comm_dir = Config.DATA_DIR / "community_detection"
        cls.csv_path = cls.comm_dir / "repository_communities.csv"
        cls.json_path = cls.comm_dir / "community_detection_metrics.json"
        cls.graphml_path = cls.comm_dir / "repository_communities.graphml"
        cls.original_graphml = Config.DATA_DIR / "network" / "collaboration_network.graphml"

        cls.df_comms = pd.read_csv(cls.csv_path)
        with open(cls.json_path, "r", encoding="utf-8") as f:
            cls.metrics = json.load(f)

    def test_original_network_file_unchanged(self):
        """Verify original network graphml exists and has exact verified counts."""
        self.assertTrue(self.original_graphml.exists())
        G = nx.read_graphml(self.original_graphml)
        self.assertEqual(G.number_of_nodes(), 4965)
        self.assertEqual(G.number_of_edges(), 5758)

    def test_all_repositories_assigned_or_marked_isolated(self):
        """Verify every repository in the projection has a valid community or -1 if isolated."""
        self.assertEqual(len(self.df_comms), 206)
        self.assertEqual(self.df_comms["repo_id"].nunique(), 206)

        # Isolated repos must have community -1
        isolated = self.df_comms[self.df_comms["is_isolated"] == True]
        self.assertEqual(len(isolated), 12)
        for _, row in isolated.iterrows():
            self.assertEqual(row["louvain_community_id"], -1)
            self.assertEqual(row["leiden_community_id"], -1)

        # Connected repos must have positive community IDs
        connected = self.df_comms[self.df_comms["is_isolated"] == False]
        self.assertEqual(len(connected), 194)
        self.assertTrue((connected["louvain_community_id"] > 0).all())
        self.assertTrue((connected["leiden_community_id"] > 0).all())

    def test_no_duplicate_repositories(self):
        """Verify no duplicate node_ids or repo_ids."""
        self.assertFalse(self.df_comms["node_id"].duplicated().any())
        self.assertFalse(self.df_comms["repo_id"].duplicated().any())

    def test_community_sizes_sum_to_assigned_repositories(self):
        """Verify sum of community sizes matches connected repositories count (194)."""
        louvain_sizes = self.metrics["louvain"]["community_sizes"]
        leiden_sizes = self.metrics["leiden"]["community_sizes"]
        self.assertEqual(sum(louvain_sizes), 194)
        self.assertEqual(sum(leiden_sizes), 194)

    def test_modularity_calculation_validity(self):
        """Verify modularity is calculated on the same graph and is > 0.40."""
        q_louvain = self.metrics["louvain"]["modularity_score"]
        q_leiden = self.metrics["leiden"]["modularity_score"]
        self.assertGreater(q_louvain, 0.40)
        self.assertGreater(q_leiden, 0.40)
        self.assertLess(q_louvain, 1.0)
        self.assertLess(q_leiden, 1.0)

    def test_reproducibility_with_fixed_seed(self):
        """Verify repeated runs with the same seed produce identical partitions."""
        detector = RepositoryCommunityDetector(graphml_path=self.original_graphml)
        res1 = detector.run_leiden(seed=42)
        res2 = detector.run_leiden(seed=42)
        self.assertEqual(res1.partition_map, res2.partition_map)
        self.assertEqual(res1.modularity_score, res2.modularity_score)

        l_res1 = detector.run_louvain(seed=42)
        l_res2 = detector.run_louvain(seed=42)
        self.assertEqual(l_res1.partition_map, l_res2.partition_map)
        self.assertEqual(l_res1.modularity_score, l_res2.modularity_score)


if __name__ == "__main__":
    unittest.main()
