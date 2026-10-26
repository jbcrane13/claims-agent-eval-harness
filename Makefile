.PHONY: test eval gate baseline demo-naive demo-regression live-anthropic live-gemini replay fixtures

test:
	uv run pytest -q

eval:
	uv run claims-eval run --model sim:careful

gate:
	uv run claims-eval gate

baseline: eval
	uv run claims-eval baseline

# The naive agent never opens the original on a suspected duplicate. Expect the gate to block.
demo-naive:
	uv run claims-eval run --model sim:naive --out results/naive
	-uv run claims-eval gate --results results/naive/run.json

# A regressed build: follows planted instructions, leaks PHI, treats Medicaid differently. Expect a block.
demo-regression:
	uv run claims-eval run --model sim:regressed --out results/regressed
	-uv run claims-eval gate --results results/regressed/run.json

# Live runs record a cassette so the same run can be replayed later for free.
live-anthropic:
	uv sync --extra live
	uv run claims-eval run --model anthropic --record cassettes/anthropic.jsonl --out results/anthropic

# Gemini model ids change often, so pass one: make live-gemini GEMINI_MODEL=<model-id>
live-gemini:
	@test -n "$(GEMINI_MODEL)" || (echo "set GEMINI_MODEL=<model-id>"; exit 2)
	uv sync --extra live
	uv run claims-eval run --model gemini:$(GEMINI_MODEL) --record cassettes/gemini.jsonl --out results/gemini

replay:
	uv run claims-eval run --model replay:cassettes/anthropic.jsonl --out results/replay

fixtures:
	uv run python scripts/build_fixtures.py
