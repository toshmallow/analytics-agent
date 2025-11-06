"""Tools for visualizing data from analytics queries."""

import io
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Literal, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import StructuredTool

from analytics_agent.tools.base import BaseTool


class VisualizationTools(BaseTool):
    """Visualization tools for the analytics agent."""

    def __init__(self, output_dir: str = "visualizations") -> None:
        """Initialize visualization tools.

        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def format_tool_call(tool_name: str, tool_args: dict) -> str:
        """Format a tool call for display in CLI.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool

        Returns:
            Formatted string representation of the tool call
        """
        output = f"📊 Calling: {tool_name}"

        if tool_name == "create_visualization" and "chart_type" in tool_args:
            output += f"\n   Chart Type: {tool_args['chart_type']}"
            if "title" in tool_args:
                output += f"\n   Title: {tool_args['title']}"

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

        if tool_name == "create_visualization":
            if "Saved to:" in result_content:
                file_path = result_content.split("Saved to: ")[1]
                output += f"\n   📈 Chart saved: {file_path}"

        return output

    def _parse_data(self, data: str) -> pd.DataFrame:
        """Parse CSV data string into a pandas DataFrame.

        Args:
            data: Data as CSV string with headers in the first row

        Returns:
            DataFrame containing the parsed data

        Raises:
            ValueError: If data cannot be parsed as CSV
        """
        try:
            df = pd.read_csv(io.StringIO(data))
            if df.empty:
                raise ValueError("CSV data is empty")
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
            raise ValueError(
                f"Data must be in CSV format with headers in the first row. Error: {str(e)}"
            ) from e

    def create_visualization(
        self,
        data: Annotated[
            str,
            "Data to visualize as CSV string with headers in the first row",
        ],
        chart_type: Annotated[
            Literal["bar", "line", "scatter", "pie"],
            "Type of chart to create (bar, line, scatter, or pie)",
        ],
        x_column: Annotated[str, "Column name for x-axis"],
        y_column: Annotated[Optional[str], "Column name for y-axis (not needed for pie)"] = None,
        title: Annotated[Optional[str], "Chart title"] = None,
        x_label: Annotated[Optional[str], "X-axis label"] = None,
        y_label: Annotated[Optional[str], "Y-axis label"] = None,
        output_path: Annotated[
            Optional[str], "Custom output path for the visualization file"
        ] = None,
    ) -> str:
        """Create a visualization from data and save to file system.

        Args:
            data: Data to visualize as CSV string with headers in the first row.
            chart_type: Type of chart (bar, line, scatter, or pie)
            x_column: Column name for x-axis
            y_column: Column name for y-axis (not needed for pie charts)
            title: Optional chart title
            x_label: Optional x-axis label
            y_label: Optional y-axis label
            output_path: Custom output path (if not provided, saves to default exports directory)

        Returns:
            Path to the saved visualization file
        """
        try:
            df = self._parse_data(data)

            if x_column not in df.columns:
                return f"Error: Column '{x_column}' not found. Available columns: {', '.join(df.columns)}"

            if chart_type != "pie" and y_column and y_column not in df.columns:
                return f"Error: Column '{y_column}' not found. Available columns: {', '.join(df.columns)}"

            plt.figure(figsize=(10, 6))

            if chart_type == "bar":
                if y_column:
                    plt.bar(df[x_column], df[y_column])
                else:
                    value_counts = df[x_column].value_counts()
                    plt.bar(value_counts.index, value_counts.values)
            elif chart_type == "line":
                if y_column:
                    plt.plot(df[x_column], df[y_column], marker="o")
                else:
                    value_counts = df[x_column].value_counts().sort_index()
                    plt.plot(value_counts.index, value_counts.values, marker="o")
            elif chart_type == "scatter":
                if y_column:
                    plt.scatter(df[x_column], df[y_column])
                else:
                    return "Error: Scatter plot requires both x_column and y_column"
            elif chart_type == "pie":
                if y_column:
                    plt.pie(df[y_column], labels=df[x_column], autopct="%1.1f%%")
                else:
                    value_counts = df[x_column].value_counts()
                    plt.pie(value_counts.values, labels=value_counts.index, autopct="%1.1f%%")

            if title:
                plt.title(title)
            if x_label and chart_type != "pie":
                plt.xlabel(x_label)
            if y_label and chart_type != "pie":
                plt.ylabel(y_label)

            plt.tight_layout()

            if output_path:
                filepath = Path(output_path)
                if not filepath.is_absolute():
                    # Strip 'exports' prefix if present to avoid nested directories
                    path_parts = filepath.parts
                    if len(path_parts) > 0 and path_parts[0] in ['exports', 'export']:
                        filepath = Path(*path_parts[1:]) if len(path_parts) > 1 else Path(filepath.name)
                    filepath = self.output_dir / filepath
                filepath.parent.mkdir(parents=True, exist_ok=True)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{chart_type}_chart_{timestamp}.png"
                filepath = self.output_dir / filename

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            return f"Visualization created successfully! Saved to: {filepath}"

        except Exception as e:
            return f"Error creating visualization: {str(e)}"

    def get_tools(self) -> List[StructuredTool]:
        """Get list of LangChain tools.

        Returns:
            List of StructuredTool instances
        """
        return [
            StructuredTool.from_function(
                func=self.create_visualization,
                name="create_visualization",
                description=(
                    "Create a visualization (bar, line, scatter, or pie chart) from data and "
                    "save it to the file system. The data must be provided as a CSV string "
                    "with headers in the first row. For bar and line charts, y_column is optional "
                    "(will create a frequency chart if omitted). For pie charts, y_column can be "
                    "values or omitted for frequency. For scatter plots, both x and y columns are "
                    "required. By default, saves to the exports directory unless output_path is specified."
                ),
            ),
        ]
