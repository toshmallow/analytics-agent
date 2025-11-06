"""Base class for analytics agent tools."""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.tools import StructuredTool


class BaseTool(ABC):
    """Abstract base class for analytics agent tools."""

    @abstractmethod
    def get_tools(self) -> List[StructuredTool]:
        """Get list of LangChain tools.

        Returns:
            List of StructuredTool instances
        """

    @staticmethod
    @abstractmethod
    def format_tool_call(tool_name: str, tool_args: dict) -> str:
        """Format a tool call for display in CLI.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool

        Returns:
            Formatted string representation of the tool call
        """

    @staticmethod
    @abstractmethod
    def format_tool_result(tool_name: str, result_content: str) -> str:
        """Format a tool result for display in CLI.

        Args:
            tool_name: Name of the tool that was called
            result_content: The result content returned by the tool

        Returns:
            Formatted string representation of the tool result
        """

    def get_tool_names(self) -> List[str]:
        """Get list of tool names provided by this tool class.

        Returns:
            List of tool names
        """
        return [tool.name for tool in self.get_tools()]
