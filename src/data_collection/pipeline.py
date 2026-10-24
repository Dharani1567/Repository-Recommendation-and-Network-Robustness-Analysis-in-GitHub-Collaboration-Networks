"""GitHub Data Collection Pipeline Runner."""

import argparse
import logging
import sys
from typing import List, Optional

from src.config.settings import Config
from src.data_collection.collector import (
    ContributionRecord,
    ContributorRecord,
    GitHubCollector,
    RepositoryRecord,
)
from src.data_collection.github_client import GitHubClient, RateLimitExceededError
from src.data_collection.storage import DataQualityReport, DatasetStorageManager

# Core representative open-source repositories spanning machine learning, data engineering, and web frameworks
DEFAULT_SEED_REPOS = [
    # Machine Learning & AI
    "scikit-learn/scikit-learn",
    "pytorch/pytorch",
    "huggingface/transformers",
    # Data & Scientific Computing
    "pandas-dev/pandas",
    "numpy/numpy",
    "scipy/scipy",
    "networkx/networkx",
    # Web & API Frameworks
    "pallets/flask",
    "tiangolo/fastapi",
    "encode/starlette",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_collection")


def run_collection_pipeline(
    seed_repos: Optional[List[str]] = None,
    max_contributors_per_repo: int = 40,
    max_repos: Optional[int] = None,
) -> DataQualityReport:
    """Execute end-to-end data collection pipeline using GitHub REST API."""
    Config.ensure_directories()
    targets = (seed_repos or DEFAULT_SEED_REPOS)
    if max_repos:
        targets = targets[:max_repos]

    client = GitHubClient()
    collector = GitHubCollector(client)

    logger.info(f"Checking GitHub API rate limit before collection...")
    rate_status = client.check_rate_limit()
    rate_info = rate_status.get("rate", {})
    logger.info(
        f"GitHub Rate Limit - Limit: {rate_info.get('limit', 'unknown')}, "
        f"Remaining: {rate_info.get('remaining', 'unknown')}"
    )

    all_repositories: List[RepositoryRecord] = []
    all_contributors: List[ContributorRecord] = []
    all_contributions: List[ContributionRecord] = []

    for full_name in targets:
        logger.info(f"Collecting repository metadata: {full_name}...")
        try:
            repo = collector.fetch_repository(full_name)
            if not repo:
                logger.warning(f"Could not retrieve repository: {full_name}")
                continue
            all_repositories.append(repo)

            logger.info(f"Collecting contributors for {full_name} (max: {max_contributors_per_repo})...")
            contribs, edges = collector.fetch_contributors_for_repo(
                repo, max_contributors=max_contributors_per_repo
            )
            all_contributors.extend(contribs)
            all_contributions.extend(edges)
            logger.info(f"Retrieved {len(contribs)} contributor relationships for {full_name}.")

        except RateLimitExceededError as e:
            logger.error(f"Rate limit exceeded while collecting {full_name}: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error while collecting {full_name}: {e}", exc_info=True)

    # Re-check rate limit status after collection run
    final_rate = client.check_rate_limit().get("rate", {})

    logger.info("Processing data validation, deduplication, and persistence...")
    report = DatasetStorageManager.process_and_save(
        repositories=all_repositories,
        contributors=all_contributors,
        contributions=all_contributions,
        api_stats=client.stats,
        rate_limit_info=final_rate,
    )

    logger.info("Data collection pipeline completed successfully.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect GitHub data for collaboration network.")
    parser.add_argument("--max-repos", type=int, default=10, help="Maximum number of seed repos to collect.")
    parser.add_argument("--max-contributors", type=int, default=30, help="Max contributors per repo.")
    args = parser.parse_args()

    report = run_collection_pipeline(
        max_repos=args.max_repos,
        max_contributors_per_repo=args.max_contributors,
    )
    print(f"Collection complete: {report.clean_counts['repositories']} repos, "
          f"{report.clean_counts['contributors']} contributors, "
          f"{report.clean_counts['contributions']} relationships.")
