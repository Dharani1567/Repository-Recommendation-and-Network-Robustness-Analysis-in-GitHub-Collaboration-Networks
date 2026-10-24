"""Unit tests for Part 5 Recommendation System."""

import json
from pathlib import Path
import unittest
import networkx as nx
import pandas as pd

from src.config.settings import Config
from src.recommendation.data_splitter import create_train_test_split
from src.recommendation.recommenders import (
    PopularityRecommender,
    CommunityAwareRecommender,
    CollaborativeFilteringRecommender,
    TopologicalSimilarityRecommender,
    Node2VecRecommender,
)
from src.recommendation.evaluator import compute_ndcg


class TestRecommendationSystem(unittest.TestCase):
    """Validation test suite for recommendation algorithms and leakage controls."""

    @classmethod
    def setUpClass(cls):
        cls.rec_dir = Config.DATA_DIR / "recommendation"
        cls.metrics_path = cls.rec_dir / "recommendation_metrics.json"
        cls.examples_path = cls.rec_dir / "recommendation_examples.csv"
        cls.original_graphml = Config.DATA_DIR / "network" / "collaboration_network.graphml"

        cls.bipartite_graph = nx.read_graphml(cls.original_graphml)
        cls.split = create_train_test_split(cls.bipartite_graph, test_ratio=0.2, seed=42)

    def test_original_graph_remains_unchanged(self):
        """Verify original network graph has not been modified or mutated."""
        self.assertEqual(self.bipartite_graph.number_of_nodes(), 4965)
        self.assertEqual(self.bipartite_graph.number_of_edges(), 5758)

    def test_leakage_control_no_heldout_edges_in_train(self):
        """Verify no test interactions exist in the training graph."""
        for c_node, held_out in self.split.test_interactions.items():
            for r_node in held_out:
                self.assertFalse(
                    self.split.train_graph.has_edge(c_node, r_node),
                    f"Leakage detected: edge ({c_node}, {r_node}) exists in train graph!",
                )

    def test_known_repositories_never_recommended(self):
        """Verify that recommenders never recommend a repository already known in training."""
        models = [
            PopularityRecommender().fit(self.split.train_graph),
            CommunityAwareRecommender().fit(self.split.train_graph, self.split.train_leiden_communities),
            CollaborativeFilteringRecommender().fit(self.split.train_graph),
        ]
        sample_contributors = self.split.test_contributors[:15]
        for c in sample_contributors:
            known = set(self.split.train_graph.neighbors(c))
            for model in models:
                recs = model.recommend(c, top_k=10, exclude_known=True)
                for r in recs:
                    self.assertNotIn(
                        r["repository_node"],
                        known,
                        f"Recommender {model.name} recommended known repository {r['repository_node']} to {c}!",
                    )

    def test_ndcg_calculation_logic(self):
        """Verify NDCG calculation on edge cases."""
        # Top-1 hit
        gt = {"R_1", "R_2"}
        recs = ["R_1", "R_99", "R_98"]
        score = compute_ndcg(recs, gt, k=3)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Zero hits
        score_zero = compute_ndcg(["R_99", "R_98"], gt, k=2)
        self.assertEqual(score_zero, 0.0)

    def test_empty_candidate_set_graceful_handling(self):
        """Verify model does not crash on unseen/isolated node."""
        cf = CollaborativeFilteringRecommender().fit(self.split.train_graph)
        recs = cf.recommend("non_existent_node", top_k=5)
        self.assertEqual(recs, [])

        comm = CommunityAwareRecommender().fit(self.split.train_graph, self.split.train_leiden_communities)
        recs_comm = comm.recommend("non_existent_node", top_k=5)
        self.assertEqual(len(recs_comm), 5)  # falls back gracefully to global ranking


if __name__ == "__main__":
    unittest.main()
