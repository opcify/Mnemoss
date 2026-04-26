# Mnemoss build / bench / lint targets.
#
# The dreaming-validation harness lives under bench/ and runs on demand.
# It hits the network (OpenAI embeddings + OpenRouter LLMs) so it's
# never part of CI by default.

.PHONY: ablate-dreaming ablate-dreaming-binary ablate-dreaming-pareto \
        bench-tests test lint typecheck

# ─── dreaming-validation harness ───────────────────────────────────

# Binary decision gate: full pipeline vs dreaming-off on the topology
# corpus. If recall@10 gap < 5pp, the per-phase study is moot — see
# docs/dreaming-decision.md. Run this BEFORE the full matrix.
ablate-dreaming-binary:
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set"; exit 2; fi
	python -m bench.ablate_dreaming --binary
	python -m bench.plot_pareto

# Full ablation matrix: 14 conditions on the topology corpus. Only run
# this if the binary gate passes. ~minutes wallclock with OpenAI
# embedder + free-tier OpenRouter rate limits.
ablate-dreaming:
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set"; exit 2; fi
	python -m bench.ablate_dreaming --full
	python -m bench.plot_pareto

# Render the Pareto chart from existing results without re-running.
ablate-dreaming-pareto:
	python -m bench.plot_pareto

# Bench harness unit tests (ARI math, bootstrap CI, etc.). These do
# NOT hit the network and ARE safe to run in CI.
bench-tests:
	pytest bench/tests/

# ─── core test / lint / typecheck ──────────────────────────────────

test:
	pytest

lint:
	ruff check src tests bench

typecheck:
	mypy --strict src/mnemoss
