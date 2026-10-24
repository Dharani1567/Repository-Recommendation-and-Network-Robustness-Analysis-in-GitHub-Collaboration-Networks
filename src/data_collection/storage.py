import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

from src.config.settings import Config
from src.data_collection.collector import (
    ContributionRecord,
    ContributorRecord,
    RepositoryRecord,
)

logger = logging.getLogger(__name__)


class DataQualityReport:
    """Tracks collection metrics, duplicates, dropped records, and quality checks."""

    def __init__(self):
        self.collection_date = datetime.now(timezone.utc).isoformat()
        self.raw_counts = {"repositories": 0, "contributors": 0, "contributions": 0}
        self.clean_counts = {"repositories": 0, "contributors": 0, "contributions": 0}
        self.duplicates_removed = {"repositories": 0, "contributors": 0, "contributions": 0}
        self.invalid_records_dropped = {"repositories": 0, "contributors": 0, "contributions": 0}
        self.bots_detected = 0
        self.filtering_reasons: List[str] = []
        self.api_stats: Dict[str, Any] = {}
        self.rate_limit_info: Dict[str, Any] = {}
        self.data_limitations: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_date": self.collection_date,
            "raw_counts": self.raw_counts,
            "clean_counts": self.clean_counts,
            "duplicates_removed": self.duplicates_removed,
            "invalid_records_dropped": self.invalid_records_dropped,
            "bots_detected": self.bots_detected,
            "filtering_reasons": self.filtering_reasons,
            "api_stats": self.api_stats,
            "rate_limit_info": self.rate_limit_info,
            "data_limitations": self.data_limitations,
        }

    def save_markdown(self, output_path: Path) -> None:
        """Render a reproducible markdown report."""
        report_md = f"""# Data Collection & Quality Report

- **Collection Date (UTC)**: {self.collection_date}
- **Repositories Collected**: {self.clean_counts['repositories']} (Raw: {self.raw_counts['repositories']}, Duplicates: {self.duplicates_removed['repositories']})
- **Contributors Collected**: {self.clean_counts['contributors']} (Raw: {self.raw_counts['contributors']}, Duplicates: {self.duplicates_removed['contributors']})
- **Contribution Relationships**: {self.clean_counts['contributions']} (Raw: {self.raw_counts['contributions']}, Duplicates: {self.duplicates_removed['contributions']})
- **Bots / Automated Accounts Flagged**: {self.bots_detected}

## Data Quality & Deduplication

| Entity | Raw Collected | Duplicates Removed | Invalid Dropped | Clean Saved |
| :--- | :--- | :--- | :--- | :--- |
| **Repositories** | {self.raw_counts['repositories']} | {self.duplicates_removed['repositories']} | {self.invalid_records_dropped['repositories']} | {self.clean_counts['repositories']} |
| **Contributors** | {self.raw_counts['contributors']} | {self.duplicates_removed['contributors']} | {self.invalid_records_dropped['contributors']} | {self.clean_counts['contributors']} |
| **Contributions** | {self.raw_counts['contributions']} | {self.duplicates_removed['contributions']} | {self.invalid_records_dropped['contributions']} | {self.clean_counts['contributions']} |

### Documented Filtering & Validation Rules
"""
        for reason in self.filtering_reasons:
            report_md += f"- {reason}\n"

        report_md += f"""
## API & Rate Limit Metrics

- **Total API Requests Issued**: {self.api_stats.get('total_requests', 0)}
- **Cache Hits**: {self.api_stats.get('cache_hits', 0)}
- **Cache Misses**: {self.api_stats.get('cache_misses', 0)}
- **API Errors Encountered**: {self.api_stats.get('api_errors', 0)}
- **Rate Limit Hits**: {self.api_stats.get('rate_limit_hits', 0)}
- **Rate Limit Remaining**: {self.rate_limit_info.get('remaining', 'N/A')} / {self.rate_limit_info.get('limit', 'N/A')}

## Data Limitations

"""
        for limitation in self.data_limitations:
            report_md += f"- {limitation}\n"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_md)


class DatasetStorageManager:
    """Validates, deduplicates, and saves datasets in CSV and JSON formats."""

    @classmethod
    def process_and_save(
        cls,
        repositories: List[RepositoryRecord],
        contributors: List[ContributorRecord],
        contributions: List[ContributionRecord],
        api_stats: Optional[Dict[str, Any]] = None,
        rate_limit_info: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
    ) -> DataQualityReport:
        """Deduplicate records, enforce data quality, write datasets, and produce report."""
        report = DataQualityReport()
        report.api_stats = api_stats or {}
        report.rate_limit_info = rate_limit_info or {}

        report.raw_counts["repositories"] = len(repositories)
        report.raw_counts["contributors"] = len(contributors)
        report.raw_counts["contributions"] = len(contributions)

        # 1. Deduplicate & Validate Repositories
        seen_repo_ids = set()
        clean_repos: List[RepositoryRecord] = []
        for r in repositories:
            if not r.repo_id or not r.full_name:
                report.invalid_records_dropped["repositories"] += 1
                continue
            if r.repo_id in seen_repo_ids:
                report.duplicates_removed["repositories"] += 1
                continue
            seen_repo_ids.add(r.repo_id)
            clean_repos.append(r)

        report.clean_counts["repositories"] = len(clean_repos)
        report.filtering_reasons.append(
            f"Repositories deduplicated by unique `repo_id`. Dropped {report.duplicates_removed['repositories']} duplicate(s)."
        )

        # 2. Deduplicate & Validate Contributors
        seen_contrib_ids = set()
        clean_contributors: List[ContributorRecord] = []
        for c in contributors:
            if not c.contributor_id or not c.login:
                report.invalid_records_dropped["contributors"] += 1
                continue
            if c.is_bot:
                report.bots_detected += 1
            if c.contributor_id in seen_contrib_ids:
                report.duplicates_removed["contributors"] += 1
                continue
            seen_contrib_ids.add(c.contributor_id)
            clean_contributors.append(c)

        report.clean_counts["contributors"] = len(clean_contributors)
        report.filtering_reasons.append(
            f"Contributors deduplicated by unique `contributor_id`. Dropped {report.duplicates_removed['contributors']} duplicate(s)."
        )

        # 3. Deduplicate & Validate Contribution Relationships
        seen_edges = set()
        clean_contributions: List[ContributionRecord] = []
        for rel in contributions:
            if not rel.contributor_id or not rel.repo_id:
                report.invalid_records_dropped["contributions"] += 1
                continue
            edge_key = (rel.contributor_id, rel.repo_id)
            if edge_key in seen_edges:
                report.duplicates_removed["contributions"] += 1
                continue
            # Ensure referenced repo and contributor exist in clean sets
            if rel.repo_id not in seen_repo_ids or rel.contributor_id not in seen_contrib_ids:
                report.invalid_records_dropped["contributions"] += 1
                continue
            seen_edges.add(edge_key)
            clean_contributions.append(rel)

        report.clean_counts["contributions"] = len(clean_contributions)
        report.filtering_reasons.append(
            f"Contribution relationships deduplicated by composite key `(contributor_id, repo_id)`. Dropped {report.duplicates_removed['contributions']} duplicate edge(s)."
        )

        # Document research limitations
        report.data_limitations.extend([
            "GitHub REST API /repos/{owner}/{repo}/contributors endpoint reflects up to the top contributors with commit counts.",
            "Anonymous contributions (without linked GitHub accounts) are omitted to preserve stable network node identifiers.",
            "Bots are flagged (`is_bot` flag) based on username suffix patterns and account type for modular filtering during preprocessing.",
        ])

        # 4. Save to files (both CSV and JSON)
        target_dir = output_dir or Config.PROCESSED_DATA_DIR
        raw_target_dir = Config.RAW_DATA_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        raw_target_dir.mkdir(parents=True, exist_ok=True)

        # DataFrames
        df_repos = pd.DataFrame([r.to_dict() for r in clean_repos])
        df_contribs = pd.DataFrame([c.to_dict() for c in clean_contributors])
        df_edges = pd.DataFrame([e.to_dict() for e in clean_contributions])

        # Save processed CSVs
        df_repos.to_csv(target_dir / "repositories.csv", index=False)
        df_contribs.to_csv(target_dir / "contributors.csv", index=False)
        df_edges.to_csv(target_dir / "contributions.csv", index=False)

        # Save raw JSONs for lossless preservation
        with open(raw_target_dir / "repositories.json", "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in clean_repos], f, indent=2)
        with open(raw_target_dir / "contributors.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in clean_contributors], f, indent=2)
        with open(raw_target_dir / "contributions.json", "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in clean_contributions], f, indent=2)

        # Save Data Quality Report
        report.save_markdown(Config.OUTPUT_DIR / "data_collection_report.md")
        with open(Config.OUTPUT_DIR / "data_collection_report.json", "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report
