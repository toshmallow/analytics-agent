"""Analytics Agent - An agentic AI system for analyzing BigQuery data."""

from analytics_agent.agent import AnalyticsAgent
from analytics_agent.config import Config
from analytics_agent.container import container

__version__ = "0.1.0"
__all__ = ["AnalyticsAgent", "Config", "container"]
