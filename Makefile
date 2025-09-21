format:
	uv run black .
	uv run isort .
	uv run ruff check --fix .