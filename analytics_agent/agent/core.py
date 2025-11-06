"""LangGraph agent for BigQuery analytics."""

from typing import Any, Generator, List, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from analytics_agent.agent.llm_manager import LLMManager
from analytics_agent.agent.state import AgentState
from analytics_agent.config import Config
from analytics_agent.tools.base import BaseTool


def create_analytics_agent(tools: List[BaseTool], config: Config) -> CompiledStateGraph:
    """Create a LangGraph agent for BigQuery analytics.

    Args:
        tools: List of tool instances to make available to the agent
        config: Application configuration

    Returns:
        Compiled StateGraph agent
    """
    langchain_tools = []
    for tool in tools:
        langchain_tools.extend(tool.get_tools())
    llm_manager = LLMManager(config)

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Determine whether to continue or end the agent loop."""
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    def call_model(state: AgentState) -> dict:
        """Call the language model with the current state."""
        messages = state["messages"]

        system_message = SystemMessage(
            content="""
            You are an expert data analyst with access to BigQuery, visualization, and file export tools.
            Your role is to help users analyze their data by:
            - Understanding their analytical questions
            - Exploring available datasets and tables
            - Writing and executing SQL queries
            - Creating visualizations when explicitly requested
            - Exporting data and files to the local file system
            - Interpreting results and providing insights

            IMPORTANT GUIDELINES:
            - ALWAYS start by briefly explaining your plan and what you're going to do
            - When calling tools, provide clear reasoning about why you're using each tool
            - Execute tools proactively without asking for permission
            - MINIMIZE the number of queries - combine multiple conditions into a single query whenever possible
            - DO NOT repeat or echo raw query results in your response
            - Instead, provide summaries, key findings, and insights
            - Focus on answering the user's question with clear explanations
            - Use concise language and highlight important patterns or trends

            VISUALIZATION GUIDELINES:
            - ONLY create visualizations when the user explicitly requests them
            - Do NOT create visualizations automatically or proactively
            - The user must use words like "visualize", "chart", "plot", "graph", or "show me a [chart type]"
            - Choose appropriate chart types: bar (comparisons), line (trends), scatter (relationships), pie (proportions)
            - Always provide meaningful titles and axis labels
            - The data for visualization must be in CSV format with column headers in the first row
            - Visualizations are ALWAYS saved to the exports/ directory by default
            - If the user specifies a custom path, use the output_path parameter
            - ALWAYS inform the user of the file path where the visualization was saved
            - After creating a visualization, check the tool result for the saved file path and include it in your response

            FILE EXPORT GUIDELINES:
            - Use export_data_to_file to save query results to CSV, JSON, or TXT files
            - Use copy_file to move visualization images to custom locations requested by the user
            - Relative paths are saved to the exports/ directory
            - Absolute paths are used as-is
            - Always confirm the file path after exporting

            QUERY EXECUTION AND DATA LIMITS:
            - MINIMIZE the number of queries executed - this is critical for performance
            - ALWAYS prefer a single query with multiple conditions over multiple separate queries
            - If you need to analyze data with different filters or conditions, combine them into ONE query using:
              * Multiple WHERE conditions with AND/OR operators
              * CASE statements for conditional logic
              * UNION ALL for combining different filter sets
              * Subqueries or CTEs (Common Table Expressions) when needed
            - Query execution and data source reads will ONLY output the first 1000 records
            - The remaining records will be omitted to optimize performance
            - BigQuery query results are returned in CSV format with column headers in the first row
            - ALWAYS use ORDER BY clause in your queries to ensure you get the most relevant records
            - Order your results to suit your analysis (e.g., ORDER BY date DESC, amount DESC, etc.)
            - DO NOT attempt to paginate query results - pagination takes too much time for large data volumes
            - Instead, optimize your analysis plan:
              * Use aggregations (GROUP BY, COUNT, SUM, AVG) to summarize large datasets
              * Apply WHERE clauses to filter data before limiting
              * Use ORDER BY to get the most relevant records within the 1000 record limit
              * Design queries that answer questions with aggregated or filtered data
            """
        )

        full_messages = [system_message] + messages
        response = llm_manager.invoke_with_tools_and_retry(langchain_tools, full_messages)

        return {
            "messages": [response],
            "analysis_context": response.content if response.content else "",
        }

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(langchain_tools))

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile()


class AnalyticsAgent:
    """High-level interface for the analytics agent."""

    def __init__(self, tools: List[BaseTool], config: Config) -> None:
        """Initialize the analytics agent.

        Args:
            tools: List of tool instances to make available to the agent
            config: Application configuration
        """
        self.tools = tools
        self.config = config
        self.graph = create_analytics_agent(tools, config)
        self.conversation_state: AgentState = {"messages": []}

    def analyze(self, question: str) -> Generator[dict[str, Any], None, None]:
        """Analyze data with streaming to show reasoning process.

        Args:
            question: The analytical question to answer

        Yields:
            Dict with event information for each step
        """
        self.conversation_state["messages"].append(HumanMessage(content=question))

        final_state = None
        for event in self.graph.stream(self.conversation_state):
            final_state = event
            yield event

        if final_state:
            for node_output in final_state.values():
                if "messages" in node_output:
                    self.conversation_state = cast(AgentState, node_output)

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_state = {"messages": []}
