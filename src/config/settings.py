"""Configuration loader and environment settings management."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base directory paths (project root hh)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Project-wide configuration values loaded safely from environment variables."""

    GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
    GITHUB_API_BASE_URL: str = os.getenv("GITHUB_API_BASE_URL", "https://api.github.com")
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    CACHE_DIR: Path = BASE_DIR / os.getenv("CACHE_DIR", "data/cache")
    OUTPUT_DIR: Path = DATA_DIR / "outputs"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate_api_token(cls) -> bool:
        """Verify whether GITHUB_TOKEN is configured without logging or exposing secret value."""
        return bool(cls.GITHUB_TOKEN and cls.GITHUB_TOKEN != "your_personal_access_token_here")

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary workspace folders if they do not exist."""
        for path in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.CACHE_DIR, cls.OUTPUT_DIR]:
            path.mkdir(parents=True, exist_ok=True)
