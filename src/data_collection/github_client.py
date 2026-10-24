"""GitHub REST API Client with rate-limiting, exponential backoff, pagination, and local caching."""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

from src.config.settings import Config

logger = logging.getLogger(__name__)


class GitHubAPIError(Exception):
    """Base exception for GitHub API errors."""
    pass


class RateLimitExceededError(GitHubAPIError):
    """Raised when GitHub API rate limit is exceeded."""
    pass


class GitHubClient:
    """Robust GitHub REST API Client.
    
    Features:
    - Safe token authentication (never logged or exposed)
    - Rate limit tracking (checks X-RateLimit headers)
    - Exponential backoff retry on network / transient server errors
    - Automatic pagination handling
    - Local disk caching to avoid redundant API queries
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        timeout: int = 15,
        max_retries: int = 4,
    ):
        self.token = token or Config.GITHUB_TOKEN
        # Validate token is not the placeholder text
        if self.token == "your_personal_access_token_here":
            self.token = None

        self.base_url = (base_url or Config.GITHUB_API_BASE_URL).rstrip("/")
        self.cache_dir = cache_dir or Config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

        # Track API usage stats for reproducibility reports
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_errors": 0,
            "rate_limit_hits": 0,
            "last_rate_limit": None,
        }

    def _get_headers(self) -> Dict[str, str]:
        """Construct request headers. Token is passed securely without being logged."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-Collaboration-Network-Research/1.0",
        }
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    def _get_cache_path(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Path:
        """Compute a deterministic hash for caching."""
        serialized = f"{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{cache_key}.json"

    def check_rate_limit(self) -> Dict[str, Any]:
        """Fetch current rate limit status from GitHub."""
        url = f"{self.base_url}/rate_limit"
        try:
            resp = self.session.get(url, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                self.stats["last_rate_limit"] = data.get("rate")
                return data
        except Exception as e:
            logger.warning(f"Could not retrieve rate limit: {e}")
        return {}

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Any:
        """Perform an authenticated GET request with retry, backoff, and cache."""
        cache_path = self._get_cache_path(endpoint, params)
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.stats["cache_hits"] += 1
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Cache read error for {cache_path}: {e}")

        self.stats["cache_misses"] += 1
        url = endpoint if endpoint.startswith("http") else f"{self.base_url}/{endpoint.lstrip('/')}"

        backoff = 2
        for attempt in range(1, self.max_retries + 1):
            self.stats["total_requests"] += 1
            try:
                resp = self.session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=self.timeout,
                )

                # Record rate limits from response headers
                limit = resp.headers.get("X-RateLimit-Limit")
                remaining = resp.headers.get("X-RateLimit-Remaining")
                reset_ts = resp.headers.get("X-RateLimit-Reset")
                if remaining is not None and limit is not None:
                    self.stats["last_rate_limit"] = {
                        "limit": int(limit),
                        "remaining": int(remaining),
                        "reset": int(reset_ts) if reset_ts else None,
                    }

                # Handle Rate Limiting (403 or 429)
                if resp.status_code in (403, 429) and remaining == "0":
                    self.stats["rate_limit_hits"] += 1
                    reset_time = int(reset_ts) if reset_ts else int(time.time() + 60)
                    wait_seconds = max(1, reset_time - int(time.time())) + 2
                    logger.warning(
                        f"GitHub API Rate Limit exceeded. Remaining: 0. Reset in {wait_seconds}s."
                    )
                    # If wait is excessively long (> 120s), raise error rather than blocking indefinitely
                    if wait_seconds > 120:
                        raise RateLimitExceededError(
                            f"GitHub API rate limit exhausted. Resets at {reset_time} ({wait_seconds}s remaining)."
                        )
                    time.sleep(wait_seconds)
                    continue

                # Handle 404 (Not Found) or 451 (Unavailable For Legal Reasons)
                if resp.status_code in (404, 451):
                    logger.warning(f"GitHub resource not found or unavailable: {url} (HTTP {resp.status_code})")
                    return None

                # Handle 401 Unauthorized
                if resp.status_code == 401:
                    self.stats["api_errors"] += 1
                    raise GitHubAPIError("GitHub API authentication failed. Verify GITHUB_TOKEN.")

                # Handle transient 5xx server errors
                if resp.status_code >= 500:
                    self.stats["api_errors"] += 1
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    resp.raise_for_status()

                # Raise for any other 4xx errors
                if resp.status_code >= 400:
                    self.stats["api_errors"] += 1
                    logger.error(f"HTTP {resp.status_code} error fetching {url}: {resp.text[:200]}")
                    return None

                data = resp.json()
                if use_cache:
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(data, f)
                    except Exception as e:
                        logger.warning(f"Failed to write cache for {cache_path}: {e}")

                return data

            except requests.exceptions.RequestException as e:
                self.stats["api_errors"] += 1
                if attempt < self.max_retries:
                    logger.warning(f"Network error on attempt {attempt}/{self.max_retries}: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                    raise GitHubAPIError(f"Network failure reaching GitHub API: {e}")

        return None

    def get_paginated(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: int = 5,
        per_page: int = 100,
        use_cache: bool = True,
    ) -> List[Any]:
        """Fetch multiple pages of results for an endpoint."""
        items: List[Any] = []
        page_params = dict(params or {})
        page_params["per_page"] = per_page

        for page in range(1, max_pages + 1):
            page_params["page"] = page
            data = self.get(endpoint, params=page_params, use_cache=use_cache)
            if not data or not isinstance(data, list):
                break
            items.extend(data)
            if len(data) < per_page:
                break
        return items
