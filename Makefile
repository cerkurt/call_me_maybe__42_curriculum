
# Required targets from call_me_maybe v1.5
.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	@/usr/bin/time -p uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run --with flake8 flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run --with flake8 flake8 .
	uv run mypy . --strict
