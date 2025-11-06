"""Tools for exporting data and files to the local file system."""

import json
import shutil
from pathlib import Path
from typing import Annotated, List, Literal

import pandas as pd
from langchain_core.tools import StructuredTool

from analytics_agent.tools.base import BaseTool


class FileExportTools(BaseTool):
    """File export tools for the analytics agent."""

    def __init__(self, default_output_dir: str = "exports") -> None:
        """Initialize file export tools.

        Args:
            default_output_dir: Default directory for exports
        """
        self.default_output_dir = Path(default_output_dir)
        self.default_output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def format_tool_call(tool_name: str, tool_args: dict) -> str:
        """Format a tool call for display in CLI.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool

        Returns:
            Formatted string representation of the tool call
        """
        output = f"📁 Calling: {tool_name}"

        if tool_name == "export_data_to_file":
            if "file_path" in tool_args:
                output += f"\n   Output: {tool_args['file_path']}"
            if "file_format" in tool_args:
                output += f"\n   Format: {tool_args['file_format']}"
        elif tool_name == "copy_file":
            if "source_path" in tool_args and "destination_path" in tool_args:
                output += f"\n   From: {tool_args['source_path']}"
                output += f"\n   To: {tool_args['destination_path']}"

        return output

    @staticmethod
    def format_tool_result(tool_name: str, result_content: str) -> str:
        """Format a tool result for display in CLI.

        Args:
            tool_name: Name of the tool that was called
            result_content: The result content returned by the tool

        Returns:
            Formatted string representation of the tool result
        """
        output = "✅ Tool result:"

        if tool_name in ["export_data_to_file", "copy_file"]:
            if "successfully" in result_content.lower():
                if "Saved to:" in result_content:
                    file_path = result_content.split("Saved to: ")[1]
                    output += f"\n   💾 File saved: {file_path}"
                elif "Copied to:" in result_content:
                    file_path = result_content.split("Copied to: ")[1]
                    output += f"\n   📋 File copied: {file_path}"
            elif "Error:" in result_content:
                output += f"\n   ⚠️  {result_content}"

        return output

    def _parse_data(self, data: str) -> pd.DataFrame:
        """Parse data string into a pandas DataFrame.

        Args:
            data: Data as JSON string or formatted table string

        Returns:
            DataFrame containing the parsed data
        """
        try:
            data_list = json.loads(data)
            if isinstance(data_list, list) and len(data_list) > 0:
                return pd.DataFrame(data_list)
        except json.JSONDecodeError:
            pass

        lines = data.strip().split("\n")
        if len(lines) < 2:
            raise ValueError("Data must have at least header and one row")

        rows = []
        for line in lines:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                rows.append(parts)

        if not rows:
            raise ValueError("No valid data rows found")

        df = pd.DataFrame(rows[1:], columns=rows[0])

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        return df

    def export_data_to_file(
        self,
        data: Annotated[str, "Data to export as JSON string or formatted table"],
        file_path: Annotated[str, "File path where data should be saved"],
        file_format: Annotated[
            Literal["csv", "json", "txt"], "Output format (csv, json, or txt)"
        ] = "csv",
    ) -> str:
        """Export data to a file in the local file system.

        Args:
            data: Data to export as JSON string or formatted table
            file_path: Path where the file should be saved (relative or absolute)
            file_format: Output format - csv, json, or txt

        Returns:
            Status message with the saved file path
        """
        try:
            output_path = Path(file_path)
            if not output_path.is_absolute():
                output_path = self.default_output_dir / output_path

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if file_format == "txt":
                output_path.write_text(data, encoding="utf-8")
            else:
                df = self._parse_data(data)

                if file_format == "csv":
                    df.to_csv(output_path, index=False)
                elif file_format == "json":
                    df.to_json(output_path, orient="records", indent=2)

            return f"Data exported successfully! Saved to: {output_path}"

        except Exception as e:
            return f"Error exporting data: {str(e)}"

    def copy_file(
        self,
        source_path: Annotated[str, "Source file path to copy from"],
        destination_path: Annotated[str, "Destination file path to copy to"],
    ) -> str:
        """Copy a file from source to destination in the local file system.

        Args:
            source_path: Path to the source file
            destination_path: Path where the file should be copied to

        Returns:
            Status message with the destination path
        """
        try:
            src = Path(source_path)
            if not src.exists():
                return f"Error: Source file does not exist: {source_path}"

            dest = Path(destination_path)
            if not dest.is_absolute():
                dest = self.default_output_dir / dest

            dest.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src, dest)

            return f"File copied successfully! Copied to: {dest}"

        except Exception as e:
            return f"Error copying file: {str(e)}"

    def get_tools(self) -> List[StructuredTool]:
        """Get list of LangChain tools.

        Returns:
            List of StructuredTool instances
        """
        return [
            StructuredTool.from_function(
                func=self.export_data_to_file,
                name="export_data_to_file",
                description=(
                    "Export data to a file in the local file system. Supports CSV, JSON, "
                    "and TXT formats. The data should be provided as a JSON string or "
                    "formatted table. Use this to save query results or any data to disk."
                ),
            ),
            StructuredTool.from_function(
                func=self.copy_file,
                name="copy_file",
                description=(
                    "Copy a file from one location to another in the local file system. "
                    "Useful for moving visualization files or any other files to a "
                    "specific location requested by the user."
                ),
            ),
        ]
