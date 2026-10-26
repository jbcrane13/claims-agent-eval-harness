.PHONY: test eval gate baseline demo-naive demo-regression live-anthropic live-gemini replay fixtures

test:
	uv run pytest -q

eval:
	uv run claims-eval run --model sim:careful

gate:
	uv run claims-eval gate

baseline: eval
	uv run claims-eval baseline

# The naive agent has a blind spot on duplicate claims. Expect the gate to block.
demo-naive:
	uv run claims-eval run --model sim:naive --out results/naive
	-uv run claims-eval gate --results results/naive/run.json

# A regressed build that follows planted instructions and leaks PHI. Expect the gate to block.
demo-regression:
	uv run claims-eval run --model sim:regressed --out results/regressed
	-uv run claims-eval gate --results results/regressed/run.json

# Live runs record a cassette so the same run can be replayed later for free.
live-anthropic:
	uv sync --extra live
	uv run claims-eval run --model anthropic --record cassettes/anthropic.jsonl --out results/anthropic

live-gemini:
	uv sync --extra live
	uv run claims-eval run --model gemini --record cassettes/gemini.jsonl --out results/gemini

replay:
	uv run claims-eval run --model replay:cassettes/anthropic.jsonl --out results/replay

fixtures:
	uv run python scripts/build_fixtures.py
