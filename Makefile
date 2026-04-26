# Mnemoss build / bench / lint targets.
#
# The dreaming-validation harness lives under bench/ and runs on demand.
# It hits the network (OpenAI embeddings + OpenRouter LLMs) so it's
# never part of CI by default.
#
# API keys come from environment variables. The harness expects:
#   OPENAI_API_KEY      — for the embedder
#   OPENROUTER_API_KEY  — for the Consolidate + judge LLMs
#
# If a .env file exists at the repo root, the bench targets auto-load
# it before invoking python. Copy .env.example to .env and fill in
# values, or set the vars in your shell rc.

.PHONY: ablate-dreaming ablate-dreaming-binary ablate-dreaming-pareto \
        ablate-dreaming-pressure ablate-dreaming-pressure-binary \
        ablate-dreaming-pressure-plot pressure-corpus-gen \
        gist-quality gist-quality-plot \
        bench-tests test lint typecheck

# Auto-load .env if it exists. ``include .env`` reads KEY=VALUE
# lines as Make assignments; ``export`` propagates them into recipe
# shells. We export only the keys the bench harness reads — no
# globbing across the whole .env so a typo stays loud.
ifneq (,$(wildcard ./.env))
include .env
export OPENAI_API_KEY
export OPENROUTER_API_KEY
endif

# ─── dreaming-validation harness ───────────────────────────────────

# Binary decision gate: full pipeline vs dreaming-off on the topology
# corpus. If recall@10 gap < 5pp, the per-phase study is moot — see
# docs/dreaming-decision.md. Run this BEFORE the full matrix.
ablate-dreaming-binary:
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set (check .env)"; exit 2; fi
	python -m bench.ablate_dreaming --binary
	python -m bench.plot_pareto

# Full ablation matrix: 14 conditions on the topology corpus. Only run
# this if the binary gate passes. ~minutes wallclock with OpenAI
# embedder + free-tier OpenRouter rate limits.
ablate-dreaming:
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set (check .env)"; exit 2; fi
	python -m bench.ablate_dreaming --full
	python -m bench.plot_pareto

# Render the Pareto chart from existing results without re-running.
ablate-dreaming-pareto:
	python -m bench.plot_pareto

# Pressure decision gate: full vs dreaming_off on the synthetic
# accumulating-pressure corpus. Tests Dispose + Rebalance combined
# effect on recall@10 and top-K cleanliness.
ablate-dreaming-pressure-binary:
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set (check .env)"; exit 2; fi
	python -m bench.ablate_dreaming --pressure-binary
	python -m bench.plot_pressure

# Pressure full matrix: 7 conditions focused on Dispose + Rebalance.
ablate-dreaming-pressure:
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set (check .env)"; exit 2; fi
	python -m bench.ablate_dreaming --pressure-full
	python -m bench.plot_pressure

# Render the pressure-effect chart from existing results.
ablate-dreaming-pressure-plot:
	python -m bench.plot_pressure

# (Re)generate the pressure corpus JSONL. Deterministic per --seed.
# Already-committed default is seed 42.
pressure-corpus-gen:
	python -m bench.fixtures.pressure_corpus_gen --seed 42

# Pairwise LLM-as-judge for Consolidate's gist quality. Topology
# corpus only (judging gists makes sense per-cluster, not at scale).
# Uses deepseek/deepseek-v4-flash on OpenRouter as a different model
# family from Consolidate's tencent/hy3-preview:free to mitigate
# self-preference bias.
gist-quality:
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set (check .env)"; exit 2; fi
	python -m bench.gist_quality
	python -m bench.plot_gist

# Render the gist-quality bar chart from existing results.
gist-quality-plot:
	python -m bench.plot_gist

# Bench harness unit tests (ARI math, bootstrap CI, corpus shape).
# These do NOT hit the network and ARE safe to run in CI.
bench-tests:
	pytest bench/tests/

# ─── core test / lint / typecheck ──────────────────────────────────

test:
	pytest

lint:
	ruff check src tests bench

typecheck:
	mypy --strict src/mnemoss
