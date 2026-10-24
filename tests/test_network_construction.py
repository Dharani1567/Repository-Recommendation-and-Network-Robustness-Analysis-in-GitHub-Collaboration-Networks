"""Unit tests for Part 3 Bipartite Network Construction and Validation."""

import unittest
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd

from src.network_construction.graph_builder import BipartiteCollaborationNetwork


class TestBipartiteNetworkConstruction(unittest.TestCase):
    """Test suite for bipartite network creation and integrity checks."""

    def setUp(self):
        self.df_repos = pd.DataFrame([
            {"repo_id": 101, "repo_name": "repoA", "full_name": "org/repoA", "stars_count": 100, "language": "Python"},
            {"repo_id": 102, "repo_name": "repoB", "full_name": "org/repoB", "stars_count": 200, "language": "Python"},
        ])
        self.df_contribs = pd.DataFrame([
            {"contributor_id": 1, "login": "alice", "is_bot": False},
            {"contributor_id": 2, "login": "bob", "is_bot": False},
            {"contributor_id": 999, "login": "bot[bot]", "is_bot": True},
        ])
        self.df_edges = pd.DataFrame([
            {"contributor_id": 1, "repo_id": 101, "contributions_count": 10, "relationship_type": "commits"},
            {"contributor_id": 1, "repo_id": 102, "contributions_count": 5, "relationship_type": "commits"},
            {"contributor_id": 2, "repo_id": 102, "contributions_count": 20, "relationship_type": "commits"},
            {"contributor_id": 999, "repo_id": 101, "contributions_count": 50, "relationship_type": "commits"},
        ])

    def test_bipartite_construction_and_validation(self):
        """Verify strict bipartiteness and bot exclusion."""
        builder = BipartiteCollaborationNetwork()
        B, report = builder.build_from_dataframes(
            self.df_repos, self.df_contribs, self.df_edges, filter_bots=True
        )

        self.assertEqual(report.validation_status, "PASSED")
        self.assertTrue(report.is_strictly_bipartite)
        self.assertEqual(report.contributor_nodes, 2)
        self.assertEqual(report.repository_nodes, 2)
        self.assertEqual(report.total_edges, 3)  # bot edge filtered

        # Check node attributes
        self.assertEqual(B.nodes["C_1"]["username"], "alice")
        self.assertEqual(B.nodes["C_1"]["node_type"], "contributor")
        self.assertEqual(B.nodes["R_101"]["repo_name"], "repoA")
        self.assertEqual(B.nodes["R_101"]["node_type"], "repository")

    def test_network_statistics_calculation(self):
        """Verify degree calculation distinguishing contributors and repos."""
        builder = BipartiteCollaborationNetwork()
        builder.build_from_dataframes(self.df_repos, self.df_contribs, self.df_edges, filter_bots=True)
        stats = builder.calculate_statistics()

        # Alice contributes to 2 repos, Bob contributes to 1
        self.assertEqual(stats["contributor_degree_distribution"]["max"], 2)
        self.assertEqual(stats["contributor_degree_distribution"]["min"], 1)
        self.assertEqual(stats["contributor_degree_distribution"]["single_repo_contributors"], 1)
        self.assertEqual(stats["contributor_degree_distribution"]["multi_repo_contributors"], 1)

        # Repos: repoA has 1 contributor, repoB has 2
        self.assertEqual(stats["repository_degree_distribution"]["min"], 1)
        self.assertEqual(stats["repository_degree_distribution"]["max"], 2)


if __name__ == "__main__":
    unittest.main()
