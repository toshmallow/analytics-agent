"""LLM manager with rate limit handling and key rotation for Gemini."""

import logging
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from analytics_agent.config import Config

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is hit."""


class LLMManager:
    """Manages LLM instance with Gemini and API key rotation."""

    GEMINI_MODEL = "gemini-2.5-flash"

    def __init__(self, config: Config) -> None:
        """Initialize LLM manager with Gemini and key rotation.

        Args:
            config: Application configuration
        """
        self.config = config

        if not config.GEMINI_API_KEYS:
            raise ValueError("GEMINI_API_KEY is required")

        self.api_keys = config.GEMINI_API_KEYS
        self.model_name = self.GEMINI_MODEL

        logger.info(
            "Initialized with Gemini model: %s, API keys available: %s",
            self.model_name,
            len(self.api_keys),
        )

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error.

        Args:
            error: Exception to check

        Returns:
            True if rate limit error
        """
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit",
            "rate_limit",
            "429",
            "too many requests",
            "quota exceeded",
            "resource exhausted",
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)

    def _create_llm(self, api_key: str) -> BaseChatModel:
        """Create Gemini LLM instance.

        Args:
            api_key: API key to use

        Returns:
            Chat model instance
        """
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=0,
        )

    def invoke_with_tools_and_retry(
        self, tools: List, messages: List, retries_per_key: int = 2
    ) -> BaseMessage:
        """Invoke LLM with tools bound, with automatic retry on rate limits.

        Args:
            tools: Tools to bind to the LLM
            messages: Messages to send
            retries_per_key: Number of retries per API key

        Returns:
            LLM response

        Raises:
            RateLimitError: If all keys exhausted
        """
        for key_index, api_key in enumerate(self.api_keys):
            logger.info(
                "Trying API key #%s/%s",
                key_index + 1,
                len(self.api_keys),
            )

            try:
                llm = self._create_llm(api_key).bind_tools(tools)
                llm_with_retry = llm.with_retry(
                    retry_if_exception_type=(Exception,),
                    stop_after_attempt=retries_per_key,
                    wait_exponential_jitter=True,
                )
                return llm_with_retry.invoke(messages)
            except Exception as e:
                if self._is_rate_limit_error(e):
                    logger.warning(
                        "Rate limit hit for key #%s, exhausted after %s attempts",
                        key_index + 1,
                        retries_per_key,
                    )
                    if key_index < len(self.api_keys) - 1:
                        logger.info("Trying next API key...")
                        continue
                else:
                    raise

        raise RateLimitError(
            f"All {len(self.api_keys)} API keys exhausted after {retries_per_key} retries each"
        )
