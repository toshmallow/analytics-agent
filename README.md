# Analytics Agent

An agentic AI system for analyzing BigQuery data using LangGraph, powered by large language models.

## Overview

This project provides an intelligent agent that can:

- Explore BigQuery datasets and tables
- Generate and execute SQL queries
- Analyze data and provide insights
- Answer analytical questions in natural language

## Architecture

The agent is built using:

- **LangGraph**: For creating the agent workflow and state management
- **LangChain**: For LLM integration and tool orchestration
- **Google Cloud BigQuery**: For data storage and querying
- **Poetry**: For dependency management
- **dependency-injector**: For dependency injection and IoC

### Key Features

- 🤖 **Google Gemini Integration**: Powered by Gemini 2.0 Flash
- 🔄 **API Key Rotation**: Automatically rotates through multiple API keys
- 🛡️ **Rate Limit Handling**: Automatically retries with exponential backoff
- 💰 **Cost Optimization**: Uses Gemini's free-tier quota
- 🔧 **Dependency Injection**: Clean, testable architecture

## Project Structure

```
analytics-agent/
├── analytics_agent/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── container.py        # Dependency injection container
│   ├── agent/              # Core agent implementation
│   │   ├── __init__.py
│   │   ├── core.py         # LangGraph agent logic
│   │   ├── llm_manager.py  # LLM with rate limit handling
│   │   └── state.py        # Agent state definitions
│   ├── clients/            # External service clients
│   │   ├── __init__.py
│   │   └── bigquery.py     # BigQuery client wrapper
│   ├── tools/              # Agent tools
│   │   ├── __init__.py
│   │   └── bigquery.py     # BigQuery tools
│   └── interface/          # User interfaces
│       ├── __init__.py
│       └── cli.py          # Command-line interface
├── examples/
│   └── basic_usage.py      # Usage examples
├── pyproject.toml
├── Makefile
├── .gitignore
└── README.md
```

## Setup

### Prerequisites

- Python 3.11 or higher
- Poetry installed
- Google Cloud Platform account with BigQuery enabled
- Google Gemini API key

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

3. Set up environment variables:
   Create a `.env` file in the project root:

**Configuration:**

```bash
# Gemini API Key (required)
# Single key
GEMINI_API_KEY=your_gemini_key

# OR multiple keys (comma-separated for key rotation)
GEMINI_API_KEY=key1,key2,key3

# BigQuery Configuration (required)
GCP_PROJECT_ID=your_project_id
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
```

4. Configure Google Cloud authentication:

```bash
gcloud auth application-default login
```

## Development Setup

### VS Code Debugging

The project includes VS Code debugging configuration in `.vscode/launch.json`:

**Analytics Agent** - Debug the CLI application

- Uses your `.env` file automatically
- Press **F5** to start debugging
- Set breakpoints by clicking left of line numbers
- Step through code with **F10** (step over) and **F11** (step into)
- Use Debug Console to inspect variables

## Usage

### Command Line Interface

Run the agent using the CLI:

```bash
poetry run python -m analytics_agent.interface.cli
```

Or using the Makefile:

```bash
make run
```

### Query BigQuery Public Datasets

The agent can query any BigQuery dataset by specifying the full dataset path in your query (e.g., `bigquery-public-data.samples`).

**Example queries to try:**

- "List all tables in the bigquery-public-data.samples dataset"
- "Show me the schema of the shakespeare table in bigquery-public-data.samples"
- "What are the top 10 most common words in Shakespeare's works? Query bigquery-public-data.samples.shakespeare"
- "How many births were there in California in 1990? Query bigquery-public-data.samples.natality"

**Available public datasets:**

- `bigquery-public-data.samples` - Sample datasets (shakespeare, natality, etc.)
- `bigquery-public-data.usa_names` - US baby names
- `bigquery-public-data.stackoverflow` - Stack Overflow data
- And many more at [Google Cloud Public Datasets](https://cloud.google.com/bigquery/public-data)

### Programmatic Usage

You can also use the agent programmatically

```python
from analytics_agent import container

# Get agent instance from DI container
agent = container.analytics_agent()

questions = [
    "List all available datasets",
    "Show me the schema of the sales table",
    "What is the total revenue by region?",
]

for question in questions:
    print(f"\nQuestion: {question}")
    response = agent.analyze(question)
    print(f"Answer: {response}\n")
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
- **BigQueryTools**: Tools for the agent with injected BigQuery client
- **AnalyticsAgent**: Main agent with injected tools

### Dependency Graph

```
Config (Singleton)
  └─> BigQueryClient (Factory)
        └─> BigQueryTools (Factory)
              └─> AnalyticsAgent (Factory)
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
agent = container.analytics_agent()
```

## Development

### Code Quality

Format code with Black:

```bash
poetry run black analytics_agent/ examples/
```

Lint with Ruff:

```bash
poetry run ruff check analytics_agent/ examples/
```

Type checking with mypy:

```bash
poetry run mypy analytics_agent/
```

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

## Agent Capabilities

The analytics agent has access to the following tools:

### BigQuery Tools

1. **execute_bigquery_sql**: Execute SQL queries against BigQuery
2. **list_bigquery_datasets**: List all available datasets
3. **list_bigquery_tables**: List tables in a specific dataset
4. **get_bigquery_table_schema**: Get the schema of a table

The agent uses these tools autonomously to:

- Understand data structure
- Generate appropriate SQL queries
- Execute queries and interpret results
- Provide insights and recommendations

### Extensibility

The modular structure makes it easy to add new capabilities:

- Add new tools in `analytics_agent/tools/` (e.g., `pandas.py`, `visualization.py`)
- Add new clients in `analytics_agent/clients/` (e.g., `postgres.py`, `snowflake.py`)
- Add new interfaces in `analytics_agent/interface/` (e.g., `web.py`, `api.py`)

## Extension Ideas

- Add more data visualization capabilities
- Integrate with additional data sources
- Add query optimization suggestions
- Implement data quality checks
- Add export functionality for reports
- Create scheduled analysis workflows

## License

MIT License
