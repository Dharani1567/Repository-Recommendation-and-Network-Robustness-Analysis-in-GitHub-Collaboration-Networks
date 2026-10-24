"""Data models and extractor for Repositories, Contributors, and Contribution edges."""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from src.data_collection.github_client import GitHubClient

logger = logging.getLogger(__name__)

# Known bot account patterns to detect automated activity
KNOWN_BOT_SUFFIXES = ("[bot]", "-bot", "-ci", "-actions", "-app")
KNOWN_BOT_LOGINS = {
    "dependabot[bot]",
    "dependabot",
    "github-actions[bot]",
    "github-actions",
    "actions-user",
    "stale[bot]",
    "greenkeeper[bot]",
    "snyk-bot",
    "codecov-commenter",
}


@dataclass
class RepositoryRecord:
    repo_id: int
    repo_name: str
    full_name: str
    owner_login: str
    description: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    language: Optional[str] = None
    stars_count: int = 0
    forks_count: int = 0
    open_issues_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pushed_at: Optional[str] = None
    size: int = 0
    default_branch: str = "main"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["topics"] = ";".join(self.topics) if self.topics else ""
        return d


@dataclass
class ContributorRecord:
    contributor_id: int
    login: str
    user_type: str = "User"
    site_admin: bool = False
    html_url: Optional[str] = None
    is_bot: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContributionRecord:
    contributor_id: int
    contributor_login: str
    repo_id: int
    repo_full_name: str
    contributions_count: int
    relationship_type: str = "commits"
    collected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GitHubCollector:
    """Orchestrates collection of repositories, contributors, and their collaboration edges."""

    def __init__(self, client: Optional[GitHubClient] = None):
        self.client = client or GitHubClient()

    @staticmethod
    def is_bot_account(login: str, user_type: str = "User") -> bool:
        """Identify automated bot / CI accounts."""
        if user_type.lower() == "bot":
            return True
        lower_login = login.lower()
        if lower_login in KNOWN_BOT_LOGINS:
            return True
        for suffix in KNOWN_BOT_SUFFIXES:
            if lower_login.endswith(suffix):
                return True
        return False

    def fetch_repository(self, owner_repo: str) -> Optional[RepositoryRecord]:
        """Fetch metadata for a single repository (e.g. 'pallets/flask')."""
        data = self.client.get(f"/repos/{owner_repo}")
        if not data or not isinstance(data, dict):
            return None

        repo_id = data.get("id")
        if repo_id is None:
            return None

        owner_info = data.get("owner", {})
        owner_login = owner_info.get("login", "") if isinstance(owner_info, dict) else ""

        return RepositoryRecord(
            repo_id=repo_id,
            repo_name=data.get("name", ""),
            full_name=data.get("full_name", owner_repo),
            owner_login=owner_login,
            description=data.get("description"),
            topics=data.get("topics", []) if isinstance(data.get("topics"), list) else [],
            language=data.get("language"),
            stars_count=int(data.get("stargazers_count", 0)),
            forks_count=int(data.get("forks_count", 0)),
            open_issues_count=int(data.get("open_issues_count", 0)),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            pushed_at=data.get("pushed_at"),
            size=int(data.get("size", 0)),
            default_branch=data.get("default_branch", "main"),
        )

    def fetch_contributors_for_repo(
        self,
        repo: RepositoryRecord,
        max_contributors: int = 100,
    ) -> Tuple[List[ContributorRecord], List[ContributionRecord]]:
        """Fetch contributors and contribution relationship edges for a repository."""
        contributors: List[ContributorRecord] = []
        contributions: List[ContributionRecord] = []

        # contributors endpoint returns list of users with contributions count
        raw_contributors = self.client.get_paginated(
            f"/repos/{repo.full_name}/contributors",
            params={"anon": "false"},
            max_pages=(max_contributors + 99) // 100,
            per_page=min(100, max_contributors),
        )

        for c_data in raw_contributors:
            if not isinstance(c_data, dict):
                continue

            c_id = c_data.get("id")
            login = c_data.get("login")
            if not c_id or not login:
                logger.warning(f"Skipping contributor with missing id/login in {repo.full_name}")
                continue

            user_type = c_data.get("type", "User")
            is_bot = self.is_bot_account(login, user_type)
            contributions_cnt = int(c_data.get("contributions", 0))

            contributor = ContributorRecord(
                contributor_id=c_id,
                login=login,
                user_type=user_type,
                site_admin=bool(c_data.get("site_admin", False)),
                html_url=c_data.get("html_url"),
                is_bot=is_bot,
            )
            contributors.append(contributor)

            contribution = ContributionRecord(
                contributor_id=c_id,
                contributor_login=login,
                repo_id=repo.repo_id,
                repo_full_name=repo.full_name,
                contributions_count=contributions_cnt,
            )
            contributions.append(contribution)

            if len(contributors) >= max_contributors:
                break

        return contributors, contributions

    def search_seed_repositories(
        self,
        query: str = "topic:machine-learning language:python stars:>1000",
        sort: str = "stars",
        order: str = "desc",
        limit: int = 20,
    ) -> List[RepositoryRecord]:
        """Search GitHub for repositories matching a research query."""
        repos: List[RepositoryRecord] = []
        raw_data = self.client.get(
            "/search/repositories",
            params={"q": query, "sort": sort, "order": order, "per_page": min(100, limit)},
        )

        if not raw_data or "items" not in raw_data:
            return repos

        for item in raw_data.get("items", []):
            repo_id = item.get("id")
            if not repo_id:
                continue
            owner = item.get("owner", {})
            owner_login = owner.get("login", "") if isinstance(owner, dict) else ""
            repo = RepositoryRecord(
                repo_id=repo_id,
                repo_name=item.get("name", ""),
                full_name=item.get("full_name", ""),
                owner_login=owner_login,
                description=item.get("description"),
                topics=item.get("topics", []) if isinstance(item.get("topics"), list) else [],
                language=item.get("language"),
                stars_count=int(item.get("stargazers_count", 0)),
                forks_count=int(item.get("forks_count", 0)),
                open_issues_count=int(item.get("open_issues_count", 0)),
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at"),
                pushed_at=item.get("pushed_at"),
                size=int(item.get("size", 0)),
                default_branch=item.get("default_branch", "main"),
            )
            repos.append(repo)
            if len(repos) >= limit:
                break

        return repos
