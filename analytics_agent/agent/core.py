"""LangGraph agent for BigQuery analytics."""

from typing import Any, Dict, Generator, List, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from analytics_agent.agent.llm_manager import LLMManager
from analytics_agent.agent.state import AgentState
from analytics_agent.config import Config
from analytics_agent.tools.base import BaseTool


def extract_working_context_from_messages(
    messages: List, tools: List[BaseTool]
) -> Dict[str, str]:
    """Extract dataset and table information from recent messages.

    Args:
        messages: List of conversation messages
        tools: List of tool instances that can extract context

    Returns:
        Dict with 'dataset' and 'table' keys if found
    """
    context = {}

    # Look at recent messages in reverse order (most recent first)
    for message in reversed(messages[-10:]):  # Check last 10 messages
        # Check if message has tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                # Handle both dict and object access patterns
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "")
                    args = tool_call.get("args", {})
                else:
                    tool_name = getattr(tool_call, "name", "")
                    args = getattr(tool_call, "args", {})

                # Ask each tool to extract context if it can
                for tool in tools:
                    extracted = tool.extract_working_context(tool_name, args)
                    if extracted:
                        context.update(extracted)
                        # Return immediately if we found dataset and table
                        if "dataset" in context and "table" in context:
                            return context

    return context


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

        # Extract working context from recent messages
        working_context = extract_working_context_from_messages(messages, tools)

        # Merge with existing context, preferring newly extracted context
        existing_context = state.get("working_context", {})
        if existing_context:
            # Keep existing context but override with new if found
            working_context = {**existing_context, **working_context}

        # Build context-aware system message
        context_info = ""
        if working_context:
            dataset = working_context.get("dataset")
            table = working_context.get("table")
            if dataset and table:
                context_info = f"\n\nCURRENT WORKING CONTEXT:\n- You are currently working with the table: `{dataset}.{table}`\n- When the user refers to 'this data', 'this table', or 'this dataset', they mean `{dataset}.{table}`\n- DO NOT ask which dataset or table to use if the context is clear from the conversation\n- DO NOT list datasets/tables again unless the user asks about different data or you need to explore a new data source\n"
            elif dataset:
                context_info = f"\n\nCURRENT WORKING CONTEXT:\n- You are currently working with the dataset: `{dataset}`\n- When the user refers to 'this data' or 'this dataset', they mean `{dataset}`\n- DO NOT ask which dataset to use if the context is clear from the conversation\n"

        system_message = SystemMessage(
            content=f"""
            You are an expert data analyst with access to BigQuery, visualization, and file export tools.
            Your role is to help users analyze their data by:
            - Understanding their analytical questions
            - Exploring available datasets and tables
            - Writing and executing SQL queries
            - Creating visualizations when explicitly requested
            - Exporting data and files to the local file system
            - Interpreting results and providing insights{context_info}

            OUTPUT FORMATTING GUIDELINES:
            - ALWAYS format your responses using MARKDOWN syntax for better readability
            - Use proper markdown elements to organize your response:
              * Use **bold** for emphasis on key findings and important numbers
              * Use headings (## or ###) to structure different sections of your response
              * Use bullet points (-) or numbered lists (1.) to present multiple items
              * Use code blocks (```sql) for SQL queries
              * Use inline code (`backticks`) for table names, column names, and technical terms
              * Use tables (markdown tables) when presenting structured data comparisons
            - Structure your responses in a clear, logical flow:
              1. Brief summary/overview of what you found
              2. Key findings with supporting data
              3. Detailed analysis if needed
              4. Actionable insights or recommendations
            - Break up long responses into sections with headings
            - Use horizontal rules (---) to separate major sections when appropriate
            - Keep paragraphs short and focused (2-4 sentences max)

            IMPORTANT GUIDELINES:
            - ALWAYS start by briefly explaining your plan and what you're going to do
            - When calling tools, provide clear reasoning about why you're using each tool
            - Execute tools proactively without asking for permission
            - MINIMIZE the number of queries - combine multiple conditions into a single query whenever possible
            - DO NOT repeat or echo raw query results in your response
            - Instead, provide summaries, key findings, and insights
            - Focus on answering the user's question with clear explanations
            - Use concise language and highlight important patterns or trends

            CONTEXT AWARENESS:
            - Pay close attention to the conversation history and what data sources you've already explored
            - If you've already used a specific dataset or table in recent messages, remember it
            - When the user says "this data", "this table", "this ecommerce data", etc., they are referring to the data you most recently worked with
            - DO NOT ask the user to specify which dataset/table unless you genuinely need clarification
            - DO NOT re-list datasets or tables if you've already done so in the current conversation
            - Only explore new datasets/tables if the user asks about different data or if the context clearly requires it

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

            ⚠️ CRITICAL DATA LIMIT WARNING:
            - Query execution and data source reads will ONLY return the FIRST 100 ROWS
            - This is a hard limit that CANNOT be changed or worked around
            - Any remaining records beyond 100 rows will be COMPLETELY OMITTED
            - BigQuery query results are returned in CSV format with column headers in the first row

            REQUIRED STRATEGIES FOR DATA ANALYSIS:
            - You MUST use aggregation or sampling techniques when analyzing data
            - NEVER rely on raw data reads for analysis of large datasets
            - ALWAYS design your queries with these strategies:

            1. **AGGREGATION (Preferred)**: Use SQL aggregation to summarize data:
              * GROUP BY with COUNT, SUM, AVG, MIN, MAX for summaries
              * DISTINCT counts for unique value analysis
              * Statistical functions like STDDEV, PERCENTILE_CONT
              * Example: Instead of reading all transactions, get SUM(amount) GROUP BY customer_id

            2. **FILTERING**: Apply WHERE clauses to focus on relevant subsets:
              * Filter by date ranges, categories, or specific conditions
              * Use HAVING with GROUP BY for post-aggregation filtering
              * Example: WHERE date >= '2024-01-01' AND category = 'electronics'

            3. **RANDOM SAMPLING**: When you need representative samples of large datasets:
              * Use ORDER BY RAND() LIMIT 100 for random sampling
              * Use TABLESAMPLE SYSTEM (10 PERCENT) for BigQuery's native sampling
              * Example: SELECT * FROM table TABLESAMPLE SYSTEM (5 PERCENT) LIMIT 100

            4. **STRATEGIC ORDERING**: Use ORDER BY to get the most relevant records:
              * ORDER BY date DESC for recent records
              * ORDER BY amount DESC for top values
              * Example: Get top 100 customers by ORDER BY total_spent DESC LIMIT 100

            - DO NOT attempt to paginate query results - pagination is not supported
            - DO NOT read raw data multiple times to build a complete picture
            - ALWAYS think: "Can I aggregate this data instead of reading raw rows?"
            """
        )

        full_messages = [system_message] + messages
        response = llm_manager.invoke_with_tools_and_retry(langchain_tools, full_messages)

        return {
            "messages": [response],
            "analysis_context": response.content if response.content else "",
            "working_context": working_context,
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
        self.conversation_state: AgentState = {"messages": [], "working_context": {}}

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
        self.conversation_state = {"messages": [], "working_context": {}}
