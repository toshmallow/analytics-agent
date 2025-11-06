"""Configuration management for the analytics agent."""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        gemini_key_str = os.getenv("GEMINI_API_KEY", "")
        self.GEMINI_API_KEYS: list[str] = [
            key.strip() for key in gemini_key_str.split(",") if key.strip()
        ]
        self.GCP_PROJECT_ID: Optional[str] = os.getenv("GCP_PROJECT_ID")
        self.GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

    def validate(self) -> None:
        """Validate required configuration."""
        if not self.GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID environment variable is required")

        if not self.GEMINI_API_KEYS:
            raise ValueError("GEMINI_API_KEY environment variable is required")
