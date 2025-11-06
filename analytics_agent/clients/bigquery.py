"""BigQuery client wrapper for executing queries."""

from typing import Any, Dict, List, Optional

from google.cloud import bigquery


class BigQueryClient:
    """Client for interacting with BigQuery."""

    def __init__(self, project_id: Optional[str] = None) -> None:
        """Initialize BigQuery client.

        Args:
            project_id: GCP project ID.
        """
        self.project_id = project_id
        self.client = bigquery.Client(project=self.project_id)

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a BigQuery SQL query and return results.

        Args:
            query: SQL query string to execute

        Returns:
            List of dictionaries containing query results
        """
        query_job = self.client.query(query)
        results = query_job.result()
        return [dict(row) for row in results]

    def list_datasets(self) -> List[str]:
        """List all datasets in the project.

        Returns:
            List of dataset IDs
        """
        datasets = list(self.client.list_datasets())
        return [dataset.dataset_id for dataset in datasets]

    def list_tables(self, dataset_id: str) -> List[str]:
        """List all tables in a dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            List of table IDs
        """
        dataset_ref = self.client.dataset(dataset_id)
        tables = list(self.client.list_tables(dataset_ref))
        return [table.table_id for table in tables]

    def get_table_schema(self, dataset_id: str, table_id: str) -> List[Dict[str, str]]:
        """Get the schema of a table.

        Args:
            dataset_id: Dataset ID
            table_id: Table ID

        Returns:
            List of dictionaries with field name and type
        """
        table_ref = self.client.dataset(dataset_id).table(table_id)
        table = self.client.get_table(table_ref)
        return [{"name": field.name, "type": field.field_type} for field in table.schema]
