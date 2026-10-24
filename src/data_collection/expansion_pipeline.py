"""CLI and automated pipeline to execute dataset expansion across curated research domains.

Operates in:
1. Live REST API mode (when GITHUB_TOKEN is configured)
2. Safe preservation mode: preserves original datasets in `data/processed/` and `data/raw/`,
   saving expanded datasets into `data/expanded_processed/` and `data/expanded_raw/`.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.config.settings import Config
from src.data_collection.collector import (
    ContributionRecord,
    ContributorRecord,
    GitHubCollector,
    RepositoryRecord,
)
from src.data_collection.curated_repos import EXPANDED_SEED_REPOSITORIES, get_all_target_repositories
from src.data_collection.github_client import GitHubClient, RateLimitExceededError
from src.data_collection.storage import DatasetStorageManager
from src.network_construction.suitability import analyze_dataset_and_network

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("expand_dataset")


def run_expansion_pipeline(
    max_repos: int = 250,
    max_contributors_per_repo: int = 30,
    output_dir: Optional[Path] = None,
    raw_output_dir: Optional[Path] = None,
) -> dict:
    """Collect expanded dataset from GitHub across ML, Data Science, Web, and DevTools."""
    out_dir = output_dir or (Config.DATA_DIR / "expanded_processed")
    raw_out = raw_output_dir or (Config.DATA_DIR / "expanded_raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)

    client = GitHubClient()
    collector = GitHubCollector(client)

    # Check rate limit status
    rate_info = client.check_rate_limit().get("rate", {})
    limit = rate_info.get("limit", 60)
    remaining = rate_info.get("remaining", 0)

    logger.info(f"GitHub Rate Limit - Limit: {limit}, Remaining: {remaining}")

    all_targets = get_all_target_repositories()[:max_repos]
    logger.info(f"Targeting {len(all_targets)} curated repositories across 4 domains.")

    # If rate limit is severely constrained (unauthenticated rate limit: remaining < 10)
    if remaining < 10 and not Config.validate_api_token():
        logger.warning(
            "GITHUB_TOKEN is not configured and unauthenticated rate limit is nearly exhausted. "
            "Cannot safely query 200+ repositories without hitting HTTP 403 rate limits."
        )
        return {
            "status": "rate_limited_or_missing_token",
            "message": f"Rate limit remaining is {remaining}/{limit}. A personal GitHub access token (GITHUB_TOKEN) is required to query ~200-500 repositories.",
            "rate_info": rate_info,
        }

    collected_repos: List[RepositoryRecord] = []
    collected_contribs: List[ContributorRecord] = []
    collected_edges: List[ContributionRecord] = []

    for idx, full_name in enumerate(all_targets, 1):
        logger.info(f"[{idx}/{len(all_targets)}] Fetching repository: {full_name}...")
        try:
            repo = collector.fetch_repository(full_name)
            if not repo:
                logger.warning(f"Could not retrieve repository: {full_name}")
                continue
            collected_repos.append(repo)

            contribs, edges = collector.fetch_contributors_for_repo(
                repo, max_contributors=max_contributors_per_repo
            )
            collected_contribs.extend(contribs)
            collected_edges.extend(edges)

        except RateLimitExceededError as e:
            logger.error(f"Rate limit exceeded during expansion at repository {full_name}: {e}")
            break
        except Exception as e:
            logger.error(f"Error collecting {full_name}: {e}")

    # Process and save expanded dataset in isolated directory
    report = DatasetStorageManager.process_and_save(
        repositories=collected_repos,
        contributors=collected_contribs,
        contributions=collected_edges,
        api_stats=client.stats,
        rate_limit_info=client.check_rate_limit().get("rate", {}),
        output_dir=out_dir,
    )

    # Save copy of report to expanded directory
    report.save_markdown(out_dir / "expansion_quality_report.md")

    # Analyze network suitability
    suitability = analyze_dataset_and_network(out_dir, dataset_name="Expanded Dataset")
    with open(out_dir / "network_suitability_report.json", "w", encoding="utf-8") as f:
        json.dump(suitability, f, indent=2)

    return {
        "status": "success",
        "report": report.to_dict(),
        "suitability": suitability,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand GitHub collaboration dataset.")
    parser.add_argument("--max-repos", type=int, default=250, help="Maximum repos to collect.")
    parser.add_argument("--max-contributors", type=int, default=30, help="Max contributors per repo.")
    args = parser.parse_args()

    result = run_expansion_pipeline(
        max_repos=args.max_repos,
        max_contributors_per_repo=args.max_contributors,
    )
    print(json.dumps(result, indent=2))
