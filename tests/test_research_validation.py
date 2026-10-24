"""Unit tests for Part 8 Research Validation and Consistency Audit."""

import json
import unittest
from pathlib import Path
import pandas as pd
from src.config.settings import Config


class TestResearchValidationAudit(unittest.TestCase):
    """Test suite ensuring empirical validity, data integrity, and artifacts for Part 8."""

    def setUp(self):
        self.validation_dir = Config.DATA_DIR / "research_validation"
        self.metric_audit_csv = self.validation_dir / "metric_audit.csv"
        self.cohort_audit_csv = self.validation_dir / "evaluation_cohort_audit.csv"
        self.baseline_comp_csv = self.validation_dir / "baseline_comparison.csv"
        self.summary_json = self.validation_dir / "validation_summary.json"
        self.report_md = self.validation_dir / "validation_report.md"

    def test_artifacts_exist(self):
        """Verify all Part 8 audit artifacts were successfully created."""
        self.assertTrue(self.metric_audit_csv.exists(), "metric_audit.csv is missing")
        self.assertTrue(self.cohort_audit_csv.exists(), "evaluation_cohort_audit.csv is missing")
        self.assertTrue(self.baseline_comp_csv.exists(), "baseline_comparison.csv is missing")
        self.assertTrue(self.summary_json.exists(), "validation_summary.json is missing")
        self.assertTrue(self.report_md.exists(), "validation_report.md is missing")

    def test_centrality_correlation_divergence(self):
        """Verify the mathematical resolution of rank correlation between all vs multi-repo contributors."""
        with open(self.summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)

        key_comp = summary["centrality_correlation_audit"]["key_comparisons"]
        
        # When leaf nodes are included, rank correlation is inflated (> 0.95)
        self.assertGreater(key_comp["degree_vs_betweenness_all"], 0.95)
        self.assertGreater(key_comp["betweenness_vs_kcore_all"], 0.95)

        # When evaluated on multi-repo bridging developers (n=562), correlation drops sharply
        self.assertLess(key_comp["degree_vs_betweenness_multi_repo"], 0.65)
        self.assertLess(key_comp["betweenness_vs_kcore_multi_repo"], 0.50)

    def test_zero_valid_recommendations_cooccurrence_collapse(self):
        """Verify the mathematical explanation of zero valid recommendations under degree attack."""
        with open(self.summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)

        zero_audit = summary["zero_valid_recommendation_audit"]
        self.assertEqual(zero_audit["surviving_multi_repo_contributors_in_train_graph"], 0)
        self.assertEqual(zero_audit["surviving_evaluation_users_count"], 324)
        # All 324 surviving users have degree exactly 1 in G_train
        self.assertEqual(zero_audit["surviving_evaluation_users_degree_in_train"].get("1"), 324)

    def test_matched_baseline_consistency(self):
        """Verify that matched baselines are calculated for surviving cohorts."""
        df = pd.read_csv(self.baseline_comp_csv)
        self.assertIn("matched_baseline_recall10", df.columns)
        self.assertIn("true_degradation_recall10", df.columns)

        # For Collaborative Filtering under random removal, matched baseline should be close to original baseline
        rand_cf = df[(df["model"] == "CollaborativeFiltering") & (df["strategy"].str.contains("Random"))]
        self.assertFalse(rand_cf.empty)
        for _, r in rand_cf.iterrows():
            self.assertAlmostEqual(r["original_baseline_recall10"], 0.3790, places=3)
            self.assertGreater(r["matched_baseline_recall10"], 0.30)

    def test_cohort_audit_tracking(self):
        """Verify evaluation cohort survival counts across removal fractions."""
        df = pd.read_csv(self.cohort_audit_csv)
        self.assertIn("original_eval_users", df.columns)
        self.assertIn("surviving_eval_users", df.columns)
        
        # Original eval users must always be 562
        self.assertTrue((df["original_eval_users"] == 562).all())
        
        # Surviving + removed must equal 562
        self.assertTrue((df["surviving_eval_users"] + df["removed_eval_users"] == 562).all())


if __name__ == "__main__":
    unittest.main()
