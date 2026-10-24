"""Unit tests for GitHub Data Collection module, mock handling, bot detection, and deduplication."""

import unittest
from unittest.mock import MagicMock

from src.data_collection.collector import (
    ContributionRecord,
    ContributorRecord,
    GitHubCollector,
    RepositoryRecord,
)
from src.data_collection.storage import DatasetStorageManager


class TestDataCollection(unittest.TestCase):
    """Test suite for data collection logic and data quality controls."""

    def test_bot_detection(self):
        """Verify bot filtering accounts for standard suffix and login heuristics."""
        self.assertTrue(GitHubCollector.is_bot_account("dependabot[bot]"))
        self.assertTrue(GitHubCollector.is_bot_account("github-actions[bot]"))
        self.assertTrue(GitHubCollector.is_bot_account("snyk-bot"))
        self.assertTrue(GitHubCollector.is_bot_account("anyuser", user_type="Bot"))
        self.assertFalse(GitHubCollector.is_bot_account("torvalds", user_type="User"))
        self.assertFalse(GitHubCollector.is_bot_account("octocat", user_type="User"))

    def test_repository_record_serialization(self):
        """Verify repository data model handles topics and fields cleanly."""
        repo = RepositoryRecord(
            repo_id=12345,
            repo_name="sample-repo",
            full_name="org/sample-repo",
            owner_login="org",
            topics=["machine-learning", "python"],
            stars_count=500,
        )
        d = repo.to_dict()
        self.assertEqual(d["repo_id"], 12345)
        self.assertEqual(d["topics"], "machine-learning;python")
        self.assertEqual(d["stars_count"], 500)

    def test_deduplication_and_quality_report(self):
        """Verify DatasetStorageManager removes duplicates and reports correct counts."""
        repo1 = RepositoryRecord(repo_id=1, repo_name="r1", full_name="org/r1", owner_login="org")
        repo1_dup = RepositoryRecord(repo_id=1, repo_name="r1", full_name="org/r1", owner_login="org")

        c1 = ContributorRecord(contributor_id=101, login="dev1")
        c1_dup = ContributorRecord(contributor_id=101, login="dev1")
        c2 = ContributorRecord(contributor_id=102, login="dev2")

        rel1 = ContributionRecord(contributor_id=101, contributor_login="dev1", repo_id=1, repo_full_name="org/r1", contributions_count=10)
        rel1_dup = ContributionRecord(contributor_id=101, contributor_login="dev1", repo_id=1, repo_full_name="org/r1", contributions_count=10)
        rel2 = ContributionRecord(contributor_id=102, contributor_login="dev2", repo_id=1, repo_full_name="org/r1", contributions_count=5)

        report = DatasetStorageManager.process_and_save(
            repositories=[repo1, repo1_dup],
            contributors=[c1, c1_dup, c2],
            contributions=[rel1, rel1_dup, rel2],
        )

        self.assertEqual(report.raw_counts["repositories"], 2)
        self.assertEqual(report.duplicates_removed["repositories"], 1)
        self.assertEqual(report.clean_counts["repositories"], 1)

        self.assertEqual(report.raw_counts["contributors"], 3)
        self.assertEqual(report.duplicates_removed["contributors"], 1)
        self.assertEqual(report.clean_counts["contributors"], 2)

        self.assertEqual(report.raw_counts["contributions"], 3)
        self.assertEqual(report.duplicates_removed["contributions"], 1)
        self.assertEqual(report.clean_counts["contributions"], 2)


if __name__ == "__main__":
    unittest.main()
