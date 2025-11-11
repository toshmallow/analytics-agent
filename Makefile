.PHONY: install lint format type-check clean run

install:
	poetry install

lint:
	poetry run ruff check analytics_agent/ examples/

format:
	poetry run black analytics_agent/ examples/
	poetry run ruff check --fix analytics_agent/ examples/

type-check:
	poetry run mypy analytics_agent/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-cli:
	poetry run python -m analytics_agent.interface.cli

run:
	poetry run python -m analytics_agent.interface.web

