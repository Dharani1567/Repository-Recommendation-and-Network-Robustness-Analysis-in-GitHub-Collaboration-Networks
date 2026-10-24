# Data Collection & Quality Report

- **Collection Date (UTC)**: 2026-10-24T18:44:51.589439+00:00
- **Repositories Collected**: 206 (Raw: 208, Duplicates: 2)
- **Contributors Collected**: 4809 (Raw: 6078, Duplicates: 1269)
- **Contribution Relationships**: 6018 (Raw: 6078, Duplicates: 60)
- **Bots / Automated Accounts Flagged**: 266

## Data Quality & Deduplication

| Entity | Raw Collected | Duplicates Removed | Invalid Dropped | Clean Saved |
| :--- | :--- | :--- | :--- | :--- |
| **Repositories** | 208 | 2 | 0 | 206 |
| **Contributors** | 6078 | 1269 | 0 | 4809 |
| **Contributions** | 6078 | 60 | 0 | 6018 |

### Documented Filtering & Validation Rules
- Repositories deduplicated by unique `repo_id`. Dropped 2 duplicate(s).
- Contributors deduplicated by unique `contributor_id`. Dropped 1269 duplicate(s).
- Contribution relationships deduplicated by composite key `(contributor_id, repo_id)`. Dropped 60 duplicate edge(s).

## API & Rate Limit Metrics

- **Total API Requests Issued**: 420
- **Cache Hits**: 8
- **Cache Misses**: 420
- **API Errors Encountered**: 0
- **Rate Limit Hits**: 0
- **Rate Limit Remaining**: 4559 / 5000

## Data Limitations

- GitHub REST API /repos/{owner}/{repo}/contributors endpoint reflects up to the top contributors with commit counts.
- Anonymous contributions (without linked GitHub accounts) are omitted to preserve stable network node identifiers.
- Bots are flagged (`is_bot` flag) based on username suffix patterns and account type for modular filtering during preprocessing.
