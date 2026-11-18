
# Required targets from call_me_maybe v1.5
.PHONY: install run debug clean lint lint-strict test test-helpers test-io test-model test-model-smoke

install:
	uv sync

run:
	@/usr/bin/time -p uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache
	rm -rf data/output


lint:
	uv run --with flake8 flake8 .
	uv run python -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run --with flake8 flake8 .
	uv run python -m mypy . --strict

.PHONY: test-errors test-21 test-42 compare-21 compare-42

test-errors:
	@/usr/bin/time -p uv run python test_suite/run_error_cases.py

test-21:
	@/usr/bin/time -p uv run python -m src \
		--functions_definition test_suite/02_prompts_21/functions.json \
		--input test_suite/02_prompts_21/prompts.json \
		--output data/output/results_21.json

test-42:
	@/usr/bin/time -p uv run python -m src \
		--functions_definition test_suite/03_prompts_42/functions.json \
		--input test_suite/03_prompts_42/prompts.json \
		--output data/output/results_42.json

compare-21:
	uv run python test_suite/compare_results.py \
		--expected test_suite/02_prompts_21/expected.json \
		--actual data/output/results_21.json

compare-42:
	uv run python test_suite/compare_results.py \
		--expected test_suite/03_prompts_42/expected.json \
		--actual data/output/results_42.json
