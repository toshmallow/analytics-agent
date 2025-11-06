"""Interface modules for CLI and web."""

from analytics_agent.interface.cli import main as cli_main
from analytics_agent.interface.web import main as web_main

__all__ = ["cli_main", "web_main"]
