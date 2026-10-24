"""Unit and integration tests for Part 10 Streamlit Research Dashboard."""

import json
import unittest
from pathlib import Path
import pandas as pd
from src.config.settings import Config


class TestStreamlitDashboard(unittest.TestCase):
    """Test suite ensuring data loaders, schemas, and metrics for the dashboard."""

    def setUp(self):
        self.data_dir = Config.DATA_DIR
        self.final_summary_json = self.data_dir / "final_analysis" / "final_analysis_summary.json"
        self.val_summary_json = self.data_dir / "research_validation" / "validation_summary.json"
        self.rec_metrics_json = self.data_dir / "recommendation" / "recommendation_metrics.json"
        self.comm_metrics_json = self.data_dir / "community_detection" / "community_detection_metrics.json"
        self.baseline_comp_csv = self.data_dir / "research_validation" / "baseline_comparison.csv"
        self.rob_metrics_csv = self.data_dir / "robustness_analysis" / "robustness_metrics.csv"
        self.influence_csv = self.data_dir / "influence_analysis" / "contributor_influence_rankings.csv"

    def test_all_dashboard_data_files_exist(self):
        """Ensure all files required by the dashboard exist and are accessible."""
        self.assertTrue(self.final_summary_json.exists(), "final_analysis_summary.json missing")
        self.assertTrue(self.val_summary_json.exists(), "validation_summary.json missing")
        self.assertTrue(self.rec_metrics_json.exists(), "recommendation_metrics.json missing")
        self.assertTrue(self.comm_metrics_json.exists(), "community_detection_metrics.json missing")
        self.assertTrue(self.baseline_comp_csv.exists(), "baseline_comparison.csv missing")
        self.assertTrue(self.rob_metrics_csv.exists(), "robustness_metrics.csv missing")
        self.assertTrue(self.influence_csv.exists(), "contributor_influence_rankings.csv missing")

    def test_reconciled_matched_baseline_values(self):
        """Verify that baseline_comparison.csv contains reconciled matched baseline values."""
        df = pd.read_csv(self.baseline_comp_csv)
        
        # Verify 1% Degree Removal CF values
        cf_1pct = df[(df["model"] == "CollaborativeFiltering") & (df["strategy"] == "Degree Centrality") & (df["fraction_removed"] == 0.01)]
        self.assertFalse(cf_1pct.empty)
        row_1 = cf_1pct.iloc[0]
        self.assertAlmostEqual(row_1["original_baseline_recall10"], 0.3790, places=3)
        self.assertAlmostEqual(row_1["matched_baseline_recall10"], 0.3631, places=3)
        self.assertAlmostEqual(row_1["post_removal_recall10"], 0.2957, places=3)
        self.assertAlmostEqual(row_1["true_degradation_recall10"], -0.0674, places=3)

        # Verify 5% Degree Removal CF values (co-occurrence collapse)
        cf_5pct = df[(df["model"] == "CollaborativeFiltering") & (df["strategy"] == "Degree Centrality") & (df["fraction_removed"] == 0.05)]
        self.assertFalse(cf_5pct.empty)
        row_5 = cf_5pct.iloc[0]
        self.assertAlmostEqual(row_5["matched_baseline_recall10"], 0.3169, places=3)
        self.assertAlmostEqual(row_5["post_removal_recall10"], 0.0000, places=3)
        self.assertAlmostEqual(row_5["true_degradation_recall10"], -0.3169, places=3)

    def test_community_detection_metrics_structure(self):
        """Verify Leiden and Louvain parameters loaded by dashboard."""
        with open(self.comm_metrics_json, "r", encoding="utf-8") as f:
            comm_data = json.load(f)

        self.assertIn("leiden", comm_data)
        self.assertIn("louvain", comm_data)
        self.assertEqual(comm_data["leiden"]["num_communities"], 10)
        self.assertEqual(comm_data["louvain"]["num_communities"], 9)
        self.assertAlmostEqual(comm_data["leiden"]["modularity_score"], 0.4643, places=3)

    def test_recommendation_benchmark_structure(self):
        """Verify models and metrics displayed on the Recommendation Benchmark page."""
        with open(self.rec_metrics_json, "r", encoding="utf-8") as f:
            rec_data = json.load(f)

        results = rec_data.get("benchmark_results", [])
        self.assertEqual(len(results), 6)
        models = [m["method"] for m in results]
        self.assertIn("Collaborative Filtering", models)
        self.assertIn("Node2Vec Embeddings", models)
        self.assertIn("Leiden Community-Aware", models)
        self.assertIn("Popularity Baseline", models)

    def test_robustness_metrics_trajectory(self):
        """Verify structural robustness degradation data."""
        df = pd.read_csv(self.rob_metrics_csv)
        self.assertIn("gcc_fraction", df.columns)
        self.assertIn("avg_shortest_path_length_gcc", df.columns)

        # Baseline GCC fraction is ~0.9233
        base_row = df[df["fraction"] == 0.0].iloc[0]
        self.assertAlmostEqual(base_row["gcc_fraction"], 0.9233, places=3)

        # 10% Betweenness removal collapses GCC to 185 nodes (~0.0412 of remaining nodes)
        betw_10 = df[(df["strategy_key"] == "betweenness") & (df["fraction"] == 0.10)].iloc[0]
        self.assertAlmostEqual(betw_10["gcc_fraction"], 0.0412, places=3)
        self.assertEqual(int(betw_10["gcc_nodes"]), 185)


if __name__ == "__main__":
    unittest.main()
