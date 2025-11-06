"""Tools for the analytics agent to interact with BigQuery."""

from typing import Annotated, List

import pandas as pd
from langchain_core.tools import StructuredTool

from analytics_agent.clients.bigquery import BigQueryClient
from analytics_agent.tools.base import BaseTool


class BigQueryTools(BaseTool):
    """BigQuery tools for the analytics agent."""

    def __init__(self, bigquery_client: BigQueryClient) -> None:
        """Initialize BigQuery tools.

        Args:
            bigquery_client: BigQuery client instance
        """
        self.client = bigquery_client

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

        if tool_name == "execute_bigquery_sql" and "query" in tool_args:
            output += f"\n   SQL: {tool_args['query']}"
        elif tool_name == "list_bigquery_tables" and "dataset_id" in tool_args:
            output += f"\n   Dataset: {tool_args['dataset_id']}"
        elif tool_name == "get_bigquery_table_schema":
            if "dataset_id" in tool_args and "table_id" in tool_args:
                output += f"\n   Table: {tool_args['dataset_id']}.{tool_args['table_id']}"

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

        if tool_name == "execute_bigquery_sql":
            if result_content == "No results returned.":
                output += "\n   Query returned 0 rows"
            else:
                lines = result_content.split("\n")
                # Subtract 1 for header row, and check for truncation note
                if "[Note: Results truncated" in result_content:
                    truncated_msg = [
                        line for line in lines if line.startswith("[Note: Results truncated")
                    ]
                    if truncated_msg:
                        output += f"\n   {truncated_msg[0]}"
                    # Row count is lines - 1 (header) - 1 (truncation note)
                    row_count = len([l for l in lines if l and not l.startswith("[Note:")])
                    if row_count > 0:
                        row_count -= 1  # Subtract header
                else:
                    # Row count is lines - 1 (header)
                    row_count = max(0, len(lines) - 1)
                    if row_count > 0:
                        output += f"\n   Query returned {row_count} rows"
        elif tool_name == "list_bigquery_datasets":
            datasets = result_content.replace("Available datasets: ", "")
            dataset_list = [d.strip() for d in datasets.split(",") if d.strip()]
            output += f"\n   Found {len(dataset_list)} datasets"
        elif tool_name == "list_bigquery_tables":
            if "Tables in" in result_content:
                tables_part = result_content.split(": ", 1)[1] if ": " in result_content else ""
                table_list = [t.strip() for t in tables_part.split(",") if t.strip()]
                output += f"\n   Found {len(table_list)} tables"
        elif tool_name == "get_bigquery_table_schema":
            lines = [
                line
                for line in result_content.split("\n")
                if line.strip() and "Schema for" not in line
            ]
            output += f"\n   Schema has {len(lines)} columns"

        return output

    def execute_bigquery_sql(self, query: Annotated[str, "SQL query to execute"]) -> str:
        """Execute a SQL query against BigQuery and return the results as CSV.

        Args:
            query: The SQL query to execute

        Returns:
            CSV formatted string of query results
        """
        # Get full results to know total count
        results = self.client.execute_query(query)

        if not results:
            return "No results returned."

        # Convert to DataFrame
        df_full = pd.DataFrame(results)
        total_rows = len(df_full)

        # Limit rows for LLM context
        max_rows_to_llm = 1000
        truncated = total_rows > max_rows_to_llm

        if truncated:
            df = df_full.head(max_rows_to_llm)
        else:
            df = df_full

        # Convert to CSV string for LangChain
        csv_str = df.to_csv(index=False)

        if truncated:
            csv_str += f"\n[Note: Results truncated to {max_rows_to_llm} rows. Total rows: {total_rows}]"

        return csv_str

    def list_bigquery_datasets(self) -> str:
        """List all available BigQuery datasets in the project.

        Returns:
            String representation of available datasets
        """
        datasets = self.client.list_datasets()
        return f"Available datasets: {', '.join(datasets)}"

    def list_bigquery_tables(
        self, dataset_id: Annotated[str, "Dataset ID to list tables from"]
    ) -> str:
        """List all tables in a specific BigQuery dataset.

        Args:
            dataset_id: The ID of the dataset

        Returns:
            String representation of available tables
        """
        tables = self.client.list_tables(dataset_id)
        return f"Tables in {dataset_id}: {', '.join(tables)}"

    def get_bigquery_table_schema(
        self,
        dataset_id: Annotated[str, "Dataset ID"],
        table_id: Annotated[str, "Table ID"],
    ) -> str:
        """Get the schema of a specific BigQuery table.

        Args:
            dataset_id: The ID of the dataset
            table_id: The ID of the table

        Returns:
            String representation of the table schema
        """
        schema = self.client.get_table_schema(dataset_id, table_id)
        schema_str = "\n".join([f"{field['name']}: {field['type']}" for field in schema])
        return f"Schema for {dataset_id}.{table_id}:\n{schema_str}"

    def get_tools(self) -> List[StructuredTool]:
        """Get list of LangChain tools.

        Returns:
            List of StructuredTool instances
        """
        return [
            StructuredTool.from_function(
                func=self.execute_bigquery_sql,
                name="execute_bigquery_sql",
                description=(
                    "Execute a SQL query against BigQuery and return the results as CSV format. "
                    "Results are automatically truncated to 1000 rows if the query returns more rows. "
                    "The CSV format includes column headers in the first row."
                ),
            ),
            StructuredTool.from_function(
                func=self.list_bigquery_datasets,
                name="list_bigquery_datasets",
                description="List all available BigQuery datasets in the project",
            ),
            StructuredTool.from_function(
                func=self.list_bigquery_tables,
                name="list_bigquery_tables",
                description="List all tables in a specific BigQuery dataset",
            ),
            StructuredTool.from_function(
                func=self.get_bigquery_table_schema,
                name="get_bigquery_table_schema",
                description="Get the schema of a specific BigQuery table",
            ),
        ]
