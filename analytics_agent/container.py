"""Dependency injection container for the analytics agent."""

from dependency_injector import containers, providers

from analytics_agent.agent.core import AnalyticsAgent
from analytics_agent.clients.bigquery import BigQueryClient
from analytics_agent.config import Config
from analytics_agent.tools.bigquery import BigQueryTools
from analytics_agent.tools.file_export import FileExportTools
from analytics_agent.tools.ml_analysis import MLAnalysisTools
from analytics_agent.tools.visualization import VisualizationTools


class Container(containers.DeclarativeContainer):
    """Application DI container."""

    config = providers.Singleton(Config)

    bigquery_client = providers.Factory(
        BigQueryClient,
        project_id=config.provided.GCP_PROJECT_ID,
    )

    bigquery_tools = providers.Factory(
        BigQueryTools,
        bigquery_client=bigquery_client,
    )

    visualization_tools = providers.Factory(
        VisualizationTools,
        output_dir="exports",
    )

    file_export_tools = providers.Factory(
        FileExportTools,
        default_output_dir="exports",
    )

    ml_analysis_tools = providers.Factory(
        MLAnalysisTools,
        output_dir="exports",
    )

    tools = providers.List(
        bigquery_tools,
        visualization_tools,
        file_export_tools,
        ml_analysis_tools,
    )

    analytics_agent = providers.Factory(
        AnalyticsAgent,
        tools=tools,
        config=config,
    )


container = Container()
