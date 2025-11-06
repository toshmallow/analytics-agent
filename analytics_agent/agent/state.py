"""State definitions for the LangGraph agent."""

from typing import Annotated, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """State for the analytics agent."""

    messages: Annotated[List[BaseMessage], add_messages]
    analysis_context: Optional[str]
