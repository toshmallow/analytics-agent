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

    def _get_message_length(self, messages: List) -> int:
        """Get total character count from messages.

        Args:
            messages: List of messages

        Returns:
            Total character count
        """
        total_chars = 0
        for message in messages:
            if hasattr(message, "content") and message.content:
                total_chars += len(str(message.content))
        return total_chars

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
        logger.info("=" * 80)
        logger.info("LLM API CALL with model %s - START", self.model_name)
        message_length = self._get_message_length(messages)
        logger.info("Total message length: %s characters", message_length)

        for key_index, api_key in enumerate(self.api_keys):
            logger.debug(
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
                response = llm_with_retry.invoke(messages)
                logger.info("LLM API CALL - SUCCESS")
                token_usage_found = False

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = response.usage_metadata
                    logger.debug("Found usage_metadata: %s", usage)

                    if isinstance(usage, dict):
                        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
                        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
                        total_tokens = usage.get("total_tokens")
                    else:
                        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
                        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
                        total_tokens = getattr(usage, "total_tokens", None)

                    if input_tokens is not None or output_tokens is not None:
                        logger.info("Token Usage:")
                        if input_tokens is not None:
                            logger.info("  - Input tokens: %s", input_tokens)
                        if output_tokens is not None:
                            logger.info("  - Output tokens: %s", output_tokens)
                        if total_tokens is not None:
                            logger.info("  - Total tokens: %s", total_tokens)
                        token_usage_found = True

                if not token_usage_found and hasattr(response, "response_metadata") and response.response_metadata:
                    # Alternative metadata location
                    metadata = response.response_metadata
                    logger.debug("Found response_metadata: %s", metadata)

                    if "token_usage" in metadata:
                        token_usage = metadata["token_usage"]
                        logger.info("Token Usage:")
                        logger.info("  - Input tokens: %s", token_usage.get("prompt_tokens", "N/A"))
                        logger.info("  - Output tokens: %s", token_usage.get("completion_tokens", "N/A"))
                        logger.info("  - Total tokens: %s", token_usage.get("total_tokens", "N/A"))
                        token_usage_found = True
                    elif "usage_metadata" in metadata:
                        usage = metadata["usage_metadata"]
                        logger.info("Token Usage:")
                        logger.info("  - Input tokens: %s", usage.get("input_tokens", "N/A"))
                        logger.info("  - Output tokens: %s", usage.get("output_tokens", "N/A"))
                        logger.info("  - Total tokens: %s", usage.get("total_tokens", "N/A"))
                        token_usage_found = True

                if not token_usage_found:
                    logger.warning("Token usage information not available in response")
                    logger.debug("Response object: %s", response)

                logger.info("=" * 80)
                return response
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
                    logger.error("LLM API CALL - FAILED")
                    logger.error("Error: %s", str(e))
                    logger.info("=" * 80)
                    raise

        logger.error("LLM API CALL - FAILED (all keys exhausted)")
        logger.info("=" * 80)
        raise RateLimitError(
            f"All {len(self.api_keys)} API keys exhausted after {retries_per_key} retries each"
        )
