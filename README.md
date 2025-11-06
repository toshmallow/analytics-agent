# Analytics Agent

An agentic AI system for analyzing BigQuery data using LangGraph, powered by Google Gemini.

## Overview

This project provides an intelligent agent that can:

- 🔍 Explore BigQuery datasets and tables
- 💾 Generate and execute SQL queries
- 📊 Create data visualizations (charts and graphs)
- 📁 Export data to files (CSV, JSON, TXT)
- 🧠 Analyze data and provide insights
- 💬 Answer analytical questions in natural language
- 🌐 Interactive web interface with streaming responses

## Architecture

The agent is built using:

- **LangGraph**: For creating the agent workflow and state management
- **LangChain**: For LLM integration and tool orchestration
- **Google Gemini 2.0 Flash**: For intelligent query understanding and generation
- **Google Cloud BigQuery**: For data storage and querying
- **Flask**: For the web interface
- **Matplotlib & Pandas**: For data visualization and manipulation
- **Poetry**: For dependency management
- **dependency-injector**: For dependency injection and IoC

### Key Features

- 🤖 **Google Gemini Integration**: Powered by Gemini 2.0 Flash (experimental)
- 🔄 **API Key Rotation**: Automatically rotates through multiple API keys
- 🛡️ **Rate Limit Handling**: Automatically retries with exponential backoff
- 💰 **Cost Optimization**: Uses Gemini's free-tier quota (15 RPM per key)
- 🔧 **Dependency Injection**: Clean, testable architecture
- 🌐 **Web Interface**: Modern chat-based UI with streaming responses
- 📊 **Data Visualization**: Create bar, line, scatter, and pie charts
- 📁 **File Export**: Save query results and visualizations to disk
- 💬 **Conversational Memory**: Maintains context across multiple questions

## Project Structure

```
analytics-agent/
├── analytics_agent/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── container.py           # Dependency injection container
│   ├── agent/                 # Core agent implementation
│   │   ├── __init__.py
│   │   ├── core.py            # LangGraph agent logic
│   │   ├── llm_manager.py     # LLM with rate limit handling
│   │   └── state.py           # Agent state definitions
│   ├── clients/               # External service clients
│   │   ├── __init__.py
│   │   └── bigquery.py        # BigQuery client wrapper
│   ├── tools/                 # Agent tools
│   │   ├── __init__.py
│   │   ├── base.py            # Base tool class
│   │   ├── bigquery.py        # BigQuery tools
│   │   ├── visualization.py   # Data visualization tools
│   │   └── file_export.py     # File export tools
│   └── interface/             # User interfaces
│       ├── __init__.py
│       ├── cli.py             # Command-line interface
│       ├── web.py             # Flask web interface
│       ├── static/            # Web assets (CSS, JS)
│       │   ├── style.css
│       │   └── script.js
│       └── templates/         # HTML templates
│           └── index.html
├── exports/                   # Output directory for files and charts
├── pyproject.toml
├── poetry.lock
├── Makefile
├── .gitignore
└── README.md
```

## Setup

### Prerequisites

- Python 3.11 or higher
- Poetry installed
- Google Cloud Platform account with BigQuery enabled
- Google Gemini API key(s)

### Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd analytics-agent
```

2. Install dependencies using Poetry:

```bash
poetry install
```

Or using the Makefile:

```bash
make install
```

3. Set up environment variables:

Create a `.env` file in the project root:

```bash
# Gemini API Key (required)
# Single key
GEMINI_API_KEY=your_gemini_key

# OR multiple keys (comma-separated for key rotation)
GEMINI_API_KEY=key1,key2,key3

# BigQuery Configuration (required)
GCP_PROJECT_ID=your_project_id

# Google Cloud Credentials (optional - if not set, uses default gcloud auth)
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json

# Flask Configuration (optional - for web interface)
FLASK_SECRET_KEY=your-secret-key-for-sessions
PORT=8080
```

4. Configure Google Cloud authentication:

```bash
gcloud auth application-default login
```

## Usage

### Command Line Interface

Run the agent using the CLI for an interactive terminal experience:

```bash
poetry run python -m analytics_agent.interface.cli
```

Or using the Makefile:

```bash
make run
```

**CLI Features:**
- Interactive question-answer loop
- Real-time tool execution feedback
- Conversation history (use `reset` to clear)
- Type `exit` or `quit` to end the session

### Web Interface

Run the web interface for a modern chat-based experience:

```bash
poetry run python -m analytics_agent.interface.web
```

Or using the Makefile:

```bash
make run-web
```

Then open your browser and navigate to `http://localhost:8080`

**Web Features:**
- 💬 Modern chat interface with streaming responses
- 🎨 Beautiful UI with syntax highlighting
- 📊 Inline visualization rendering
- 💾 Automatic conversation persistence per session
- 🔄 Real-time tool execution updates
- 📱 Responsive design

### Query BigQuery Public Datasets

The agent can query any BigQuery dataset by specifying the full dataset path in your query (e.g., `bigquery-public-data.samples`).

**Example queries to try:**

- "List all tables in the bigquery-public-data.samples dataset"
- "Show me the schema of the shakespeare table in bigquery-public-data.samples"
- "What are the top 10 most common words in Shakespeare's works? Query bigquery-public-data.samples.shakespeare"
- "How many births were there in California in 1990? Query bigquery-public-data.samples.natality"
- "Visualize the top 10 words as a bar chart" (after querying Shakespeare data)
- "Export the results to a CSV file"

**Available public datasets:**

- `bigquery-public-data.samples` - Sample datasets (shakespeare, natality, etc.)
- `bigquery-public-data.usa_names` - US baby names
- `bigquery-public-data.stackoverflow` - Stack Overflow data
- And many more at [Google Cloud Public Datasets](https://cloud.google.com/bigquery/public-data)

### Creating Visualizations

The agent can create visualizations when explicitly requested:

```
> Show me the top 10 products by revenue from the sales table

> Now create a bar chart of this data
```

**Supported chart types:**
- **Bar charts**: For comparisons
- **Line charts**: For trends over time
- **Scatter plots**: For relationships between variables
- **Pie charts**: For proportions

All visualizations are saved to the `exports/` directory by default.

### Exporting Data

The agent can export query results to files:

```
> Export the sales data to a CSV file named "sales_2024.csv"

> Save the top customers to a JSON file
```

**Supported export formats:**
- CSV (comma-separated values)
- JSON (structured data)
- TXT (plain text)

### Programmatic Usage

You can also use the agent programmatically:

```python
from analytics_agent import container

# Get agent instance from DI container
agent = container.analytics_agent()

questions = [
    "List all tables in the bigquery-public-data.samples dataset",
    "What are the top 5 words in Shakespeare's works?",
]

for question in questions:
    print(f"\nQuestion: {question}")

    # Stream responses
    for event in agent.analyze(question):
        for node_name, node_output in event.items():
            if node_name == "agent":
                messages = node_output.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    if last_message.content:
                        print(f"Response: {last_message.content}")
```

### Custom Configuration & Dependency Override

```python
from analytics_agent import container
from analytics_agent.clients.bigquery import BigQueryClient

# Override specific dependencies
container.bigquery_client.override(
    BigQueryClient(project_id="custom-project")
)

# The entire dependency tree is automatically updated
agent = container.analytics_agent()
```

## Agent Capabilities

The analytics agent has access to the following tools:

### BigQuery Tools

1. **execute_bigquery_sql**: Execute SQL queries against BigQuery
2. **list_bigquery_datasets**: List all available datasets
3. **list_bigquery_tables**: List tables in a specific dataset
4. **get_bigquery_table_schema**: Get the schema of a table

### Visualization Tools

1. **create_visualization**: Create charts (bar, line, scatter, pie) from data
   - Automatically saves to `exports/` directory
   - Supports custom titles, labels, and formatting
   - Accepts data in CSV format

### File Export Tools

1. **export_data_to_file**: Save data to CSV, JSON, or TXT files
2. **copy_file**: Copy files to different locations

The agent uses these tools autonomously to:

- Understand data structure
- Generate appropriate SQL queries
- Execute queries and interpret results
- Create visualizations when requested
- Export data to files
- Provide insights and recommendations

## Rate Limit Handling & API Key Rotation

The agent includes **automatic rate limit handling** with API key rotation:

### How It Works

1. **Automatic Detection**: Detects rate limit errors (429, quota exceeded, etc.)
2. **API Key Rotation**: Automatically switches to next available API key
3. **Exponential Backoff**: Retries with increasing delays (1s, 2s, 5s)
4. **Cooldown Management**: 60-second cooldown per API key after rate limit
5. **Retry Logic**: Retries failed requests up to 3 times with different keys

### Multiple API Keys Configuration

To maximize your rate limits, you can provide multiple Gemini API keys:

```bash
# Single key (basic usage)
GEMINI_API_KEY=your_key_here

# Multiple keys (recommended for heavy usage)
GEMINI_API_KEY=key1,key2,key3,key4
```

**Rotation Strategy:**

1. ⚠️ **Key #1** hits rate limit → 🔄 Rotates to **Key #2**
2. ⚠️ **Key #2** hits rate limit → 🔄 Rotates to **Key #3**
3. ⚠️ **Key #3** hits rate limit → 🔄 Rotates to **Key #4**
4. ⏰ After 60 seconds, **Key #1** becomes available again
5. 🔁 Cycles back through keys as they become available

**Advantages:**

- **Extended usage** before hitting rate limits
- **Automatic failover** between keys
- **Maximizes free-tier quota** across multiple API keys

### Gemini Model

The agent uses Google's Gemini 2.0 Flash (experimental) model which offers:

- Fast response times
- Generous free tier quota (15 RPM per key)
- High-quality outputs
- Function calling support

## Dependency Injection

The project uses a Dependency Injection (DI) container to manage dependencies, making the code:

- **Testable**: Easy to mock dependencies in tests
- **Maintainable**: Clear dependency graph
- **Flexible**: Easy to swap implementations
- **Configurable**: Override dependencies as needed

### Container Structure

The DI container (`analytics_agent/container.py`) manages:

- **Config**: Application configuration from environment variables
- **BigQueryClient**: Client for interacting with BigQuery
- **BigQueryTools**: BigQuery tools with injected client
- **VisualizationTools**: Visualization tools with output directory
- **FileExportTools**: File export tools with output directory
- **AnalyticsAgent**: Main agent with all injected tools

### Dependency Graph

```
Config (Singleton)
  ├─> BigQueryClient (Factory)
  │     └─> BigQueryTools (Factory)
  ├─> VisualizationTools (Factory)
  ├─> FileExportTools (Factory)
  └─> AnalyticsAgent (Factory)
        └─> All Tools
```

### Benefits

1. **Testing**: Override dependencies with mocks

```python
from analytics_agent import container
from unittest.mock import Mock

# Mock the client
mock_client = Mock()
container.bigquery_client.override(mock_client)

# All dependencies downstream are automatically updated
agent = container.analytics_agent()
```

2. **Multiple Environments**: Different configurations per environment

```python
from analytics_agent import container
from analytics_agent.config import Config

# Development
dev_config = Config()
dev_config.GCP_PROJECT_ID = "dev-project"
container.config.override(dev_config)

# Production
prod_config = Config()
prod_config.GCP_PROJECT_ID = "prod-project"
container.config.override(prod_config)
```

3. **Accessing Individual Components**: Get any component from the container

```python
from analytics_agent import container

# Get individual services
config = container.config()
bigquery_client = container.bigquery_client()
bigquery_tools = container.bigquery_tools()
visualization_tools = container.visualization_tools()
file_export_tools = container.file_export_tools()
agent = container.analytics_agent()
```

## Development

### Code Quality

Format code with Black:

```bash
poetry run black analytics_agent/
```

Or using the Makefile:

```bash
make format
```

Lint with Ruff:

```bash
poetry run ruff check analytics_agent/
```

Or using the Makefile:

```bash
make lint
```

Type checking with mypy:

```bash
poetry run mypy analytics_agent/
```

Or using the Makefile:

```bash
make type-check
```

Clean up cache files:

```bash
make clean
```

### Available Make Commands

- `make install` - Install dependencies
- `make run` - Run CLI interface
- `make run-web` - Run web interface
- `make lint` - Run linter
- `make format` - Format code and fix linting issues
- `make type-check` - Run type checker
- `make clean` - Clean cache files

## Performance Considerations

### Query Optimization

The agent is designed to minimize BigQuery costs and execution time:

- **Result Limiting**: Query results are limited to 1000 records for performance
- **Smart Ordering**: Uses ORDER BY to get the most relevant records first
- **Query Consolidation**: Combines multiple conditions into single queries
- **Aggregation First**: Uses GROUP BY and aggregations for large datasets

### Data Output

- Query results are returned in CSV format with headers
- The first 1000 records are returned to optimize performance
- Use ORDER BY in queries to ensure you get the most relevant data
- For large datasets, use aggregations (COUNT, SUM, AVG, etc.)

## Extension Ideas

Some ideas for extending the agent's capabilities:

- ✅ Add more data visualization capabilities (IMPLEMENTED)
- ✅ Add export functionality for reports (IMPLEMENTED)
- ✅ Create web interface (IMPLEMENTED)
- 🔜 Integrate with additional data sources (PostgreSQL, Snowflake, etc.)
- 🔜 Add query optimization suggestions
- 🔜 Implement data quality checks
- 🔜 Create scheduled analysis workflows
- 🔜 Add authentication and multi-user support to web interface
- 🔜 Implement data caching for faster responses

### Extensibility

The modular structure makes it easy to add new capabilities:

- Add new tools in `analytics_agent/tools/` (e.g., `statistics.py`, `forecasting.py`)
- Add new clients in `analytics_agent/clients/` (e.g., `postgres.py`, `snowflake.py`)
- Add new interfaces in `analytics_agent/interface/` (e.g., `api.py`, `slack_bot.py`)
- Register new tools in `container.py`

## Troubleshooting

### Common Issues

**Configuration Error**
```
Configuration error: GCP_PROJECT_ID environment variable is required
```
Solution: Make sure your `.env` file contains all required variables.

**Authentication Error**
```
Error: Could not automatically determine credentials
```
Solution: Run `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS`.

**Rate Limit Error**
```
Rate limit exceeded
```
Solution: Add multiple API keys in your `.env` file (comma-separated).

**Port Already in Use (Web Interface)**
```
Address already in use
```
Solution: Change the port by setting `PORT` in your `.env` file or kill the process using port 8080.

## License

MIT License
