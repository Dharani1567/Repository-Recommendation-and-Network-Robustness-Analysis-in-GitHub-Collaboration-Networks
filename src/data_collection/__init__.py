"""GitHub data collection module exports."""

from .github_client import GitHubClient, GitHubAPIError, RateLimitExceededError
from .collector import (
    GitHubCollector,
    RepositoryRecord,
    ContributorRecord,
    ContributionRecord,
)
from .storage import DatasetStorageManager, DataQualityReport
from .pipeline import run_collection_pipeline, DEFAULT_SEED_REPOS

__all__ = [
    "GitHubClient",
    "GitHubAPIError",
    "RateLimitExceededError",
    "GitHubCollector",
    "RepositoryRecord",
    "ContributorRecord",
    "ContributionRecord",
    "DatasetStorageManager",
    "DataQualityReport",
    "run_collection_pipeline",
    "DEFAULT_SEED_REPOS",
]
