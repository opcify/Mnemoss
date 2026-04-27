# Mnemoss launch work: common commands.
#
# The design doc (~/.gstack/projects/opcify-Mnemoss/...-design-*.md)
# cites `make launch-bench` as the one-command reproducibility path
# for Chart 1. That lives here.
#
# Everything else in the file is a thin convenience wrapper around
# `python -m ...` commands the rest of the project exposes. No magic.
#
# The dreaming-validation harness lives under bench/ablate_dreaming.py
# and runs on demand. It hits the network (OpenAI embeddings +
# OpenRouter LLMs) so it's never part of CI by default.
#
# API keys:
#   OPENAI_API_KEY            — embedder (text-embedding-3-small)
#   OPENROUTER_API_KEY        — Consolidate + judge LLMs
#   GEMINI_API_KEY            — record-simulation (Gemini 2.5 Flash)
#   GOOGLE_API_KEY            — alternate Gemini auth
#   MNEMOSS_BENCH_BUDGET_USD  — optional spend cap

# ─── .env auto-load ────────────────────────────────────────────────
#
# Copy .env.example to .env and fill in your keys. The `-include`
# (with the leading dash) makes it optional: `.env` missing is not
# an error. The `export` line propagates the listed vars to every
# recipe's subprocess (which is how Python sees them via os.environ).
# The Python CLIs also call load_dotenv() themselves as a backstop
# for direct invocation without `make`.

-include .env
export OPENAI_API_KEY GEMINI_API_KEY GOOGLE_API_KEY OPENROUTER_API_KEY MNEMOSS_BENCH_BUDGET_USD

PY ?= python
RESULTS_DIR ?= bench/results
FIGURES_DIR ?= docs/figures_out
DEMO_OUT ?= demo/out

.PHONY: help
help:
	@awk 'BEGIN { FS = ":.*## "; printf "\nUsage: make \033[36m<target>\033[0m\n\nTargets:\n" } \
	     /^[a-zA-Z_-]+:.*?## / { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\nMnemoss launch — quick tour:\n"
	@printf "  1. make test                # run the full unit suite\n"
	@printf "  2. make launch-bench-smoke  # quick sanity, no API key\n"
	@printf "  3. make launch-bench        # real benchmark (needs OPENAI_API_KEY)\n"
	@printf "  4. make figures chart1      # render all SVGs\n"
	@printf "  5. make record-simulation   # record Scene 1 (needs GEMINI_API_KEY)\n\n"


# ─── dev loop ──────────────────────────────────────────────────────

.PHONY: test
test: ## run pytest (non-integration)
	$(PY) -m pytest -m "not integration" -q

.PHONY: test-integration
test-integration: ## run the integration-marked tests (model downloads)
	$(PY) -m pytest -m integration -q

.PHONY: lint
lint: ## ruff check
	ruff check src tests bench demo docs

.PHONY: format
format: ## ruff format (rewrites files)
	ruff format src tests bench demo docs

.PHONY: format-check
format-check: ## ruff format --check (CI-style)
	ruff format --check src tests bench demo docs

.PHONY: typecheck
typecheck: ## mypy strict on src/mnemoss
	mypy --strict src/mnemoss


# ─── launch comparison (Chart 1) ───────────────────────────────────

.PHONY: locomo-data
locomo-data: bench/data/locomo_memories.jsonl bench/data/locomo_queries.jsonl ## prepare the LoCoMo corpus

bench/data/locomo10.json:
	@mkdir -p bench/data
	curl -sL -o $@ https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

bench/data/locomo_memories.jsonl bench/data/locomo_queries.jsonl: bench/data/locomo10.json
	$(PY) -m bench.data.prepare_locomo

.PHONY: launch-bench-smoke
launch-bench-smoke: locomo-data ## quick sanity run (no network, ~1 min)
	@mkdir -p $(RESULTS_DIR)
	$(PY) -m bench.launch_comparison \
		--backend static_file \
		--limit-conversations 2 --limit-utterances 50 \
		--out $(RESULTS_DIR)/chart1_static_file.json --print-summary
	$(PY) -m bench.launch_comparison \
		--backend raw_stack --fake-embedder \
		--limit-conversations 2 --limit-utterances 50 \
		--out $(RESULTS_DIR)/chart1_raw_stack_fake.json --print-summary
	$(PY) -m bench.launch_comparison \
		--backend mnemoss --fake-embedder \
		--limit-conversations 2 --limit-utterances 50 \
		--out $(RESULTS_DIR)/chart1_mnemoss_fake.json --print-summary
	$(PY) -m bench.plots --chart 1 \
		--results $(RESULTS_DIR)/chart1_static_file.json \
		          $(RESULTS_DIR)/chart1_raw_stack_fake.json \
		          $(RESULTS_DIR)/chart1_mnemoss_fake.json \
		--out $(RESULTS_DIR)/chart1_smoke.svg
	@echo "smoke chart: $(RESULTS_DIR)/chart1_smoke.svg"

.PHONY: launch-bench
launch-bench: locomo-data ## full Chart 1 run (needs OPENAI_API_KEY, ~$$2)
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "ERROR: OPENAI_API_KEY must be set for the full benchmark."; exit 1; \
	fi
	@mkdir -p $(RESULTS_DIR)
	$(PY) -m bench.launch_comparison \
		--backend static_file \
		--out $(RESULTS_DIR)/chart1_static_file.json --print-summary
	$(PY) -m bench.launch_comparison \
		--backend raw_stack \
		--out $(RESULTS_DIR)/chart1_raw_stack.json --print-summary
	$(PY) -m bench.launch_comparison \
		--backend mnemoss \
		--out $(RESULTS_DIR)/chart1_mnemoss.json --print-summary
	$(PY) -m bench.plots --chart 1 \
		--results $(RESULTS_DIR)/chart1_static_file.json \
		          $(RESULTS_DIR)/chart1_raw_stack.json \
		          $(RESULTS_DIR)/chart1_mnemoss.json \
		--out $(RESULTS_DIR)/chart1.svg \
		--title "Recall@10 on LoCoMo 2024 — Mnemoss vs the stack you'd build"
	@echo "published chart: $(RESULTS_DIR)/chart1.svg"

.PHONY: chart1
chart1: ## re-render Chart 1 SVG from existing JSON
	@if [ -f $(RESULTS_DIR)/chart1_mnemoss.json ]; then \
		$(PY) -m bench.plots --chart 1 \
			--results $(RESULTS_DIR)/chart1_static_file.json \
			          $(RESULTS_DIR)/chart1_raw_stack.json \
			          $(RESULTS_DIR)/chart1_mnemoss.json \
			--out $(RESULTS_DIR)/chart1.svg; \
		echo "wrote $(RESULTS_DIR)/chart1.svg"; \
	else \
		$(PY) -m bench.plots --chart 1 \
			--results $(RESULTS_DIR)/chart1_static_file.json \
			          $(RESULTS_DIR)/chart1_raw_stack_fake.json \
			          $(RESULTS_DIR)/chart1_mnemoss_fake.json \
			--out $(RESULTS_DIR)/chart1_smoke.svg; \
		echo "wrote $(RESULTS_DIR)/chart1_smoke.svg (smoke — no full run available)"; \
	fi


# ─── dreaming-validation harness ───────────────────────────────────
#
# Per-phase ablation study + final comprehensive validation. Pre-
# registered KEEP/CUT/REBUILD verdicts live in docs/dreaming-decision.md.

.PHONY: ablate-dreaming-binary
ablate-dreaming-binary: ## binary gate: full pipeline vs dreaming-off (topology)
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set (check .env)"; exit 2; fi
	$(PY) -m bench.ablate_dreaming --binary
	$(PY) -m bench.plot_pareto

.PHONY: ablate-dreaming
ablate-dreaming: ## full topology ablation matrix (14 conditions)
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set (check .env)"; exit 2; fi
	$(PY) -m bench.ablate_dreaming --full
	$(PY) -m bench.plot_pareto

.PHONY: ablate-dreaming-pareto
ablate-dreaming-pareto: ## re-render Pareto chart from existing results
	$(PY) -m bench.plot_pareto

.PHONY: ablate-dreaming-pressure-binary
ablate-dreaming-pressure-binary: ## pressure binary gate (Dispose + Rebalance)
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set (check .env)"; exit 2; fi
	$(PY) -m bench.ablate_dreaming --pressure-binary
	$(PY) -m bench.plot_pressure

.PHONY: ablate-dreaming-pressure
ablate-dreaming-pressure: ## pressure full matrix (7 conditions)
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set (check .env)"; exit 2; fi
	$(PY) -m bench.ablate_dreaming --pressure-full
	$(PY) -m bench.plot_pressure

.PHONY: ablate-dreaming-pressure-plot
ablate-dreaming-pressure-plot: ## re-render pressure chart from existing results
	$(PY) -m bench.plot_pressure

.PHONY: pressure-corpus-gen
pressure-corpus-gen: ## (re)generate pressure corpus JSONL (seed 42)
	$(PY) -m bench.fixtures.pressure_corpus_gen --seed 42

.PHONY: gist-quality
gist-quality: ## pairwise LLM-as-judge for Consolidate's gist quality
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set (check .env)"; exit 2; fi
	$(PY) -m bench.gist_quality
	$(PY) -m bench.plot_gist

.PHONY: gist-quality-plot
gist-quality-plot: ## re-render gist-quality bar chart
	$(PY) -m bench.plot_gist

.PHONY: forgetting-curves
forgetting-curves: ## B_i vs age scatter on the pressure corpus
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	$(PY) -m bench.forgetting_curves --ablation dreaming_off

.PHONY: comprehensive-validation
comprehensive-validation: ## final speed+accuracy run (4 conds × 2 corpora × 3 reps)
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "error: OPENAI_API_KEY not set (check .env)"; exit 2; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "error: OPENROUTER_API_KEY not set (check .env)"; exit 2; fi
	$(PY) -m bench.comprehensive_validation

.PHONY: bench-tests
bench-tests: ## bench-harness unit tests (offline, safe in CI)
	pytest bench/tests/


# ─── static explainer figures ──────────────────────────────────────

.PHONY: figures
figures: ## render the three static explainer figures (A/B/C)
	@mkdir -p $(FIGURES_DIR)
	$(PY) -m docs.figures --out-dir $(FIGURES_DIR)


# ─── simulation recording ──────────────────────────────────────────

.PHONY: record-simulation
record-simulation: ## record Scene 1 with Gemini (needs GEMINI_API_KEY)
	@if [ -z "$$GEMINI_API_KEY" ] && [ -z "$$GOOGLE_API_KEY" ]; then \
		echo "ERROR: GEMINI_API_KEY or GOOGLE_API_KEY must be set."; exit 1; \
	fi
	@mkdir -p $(DEMO_OUT)
	$(PY) -m demo.simulate \
		--scene scene1_preference_recall \
		--backend mnemoss --llm gemini \
		--out $(DEMO_OUT)/trace-scene1.json
	$(PY) -m demo.render_trace $(DEMO_OUT)/trace-scene1.json \
		--out $(DEMO_OUT)/player.html \
		--title "Mnemoss · Scene 1 — Preference Recall"
	@echo "committed launch asset: $(DEMO_OUT)/player.html"

.PHONY: record-simulation-stub
record-simulation-stub: ## re-record Scene 1 with StubLLM (deterministic, no API)
	@mkdir -p $(DEMO_OUT)
	$(PY) -m demo.simulate \
		--scene scene1_preference_recall \
		--backend mnemoss --llm stub \
		--out $(DEMO_OUT)/trace-scene1.json
	$(PY) -m demo.render_trace $(DEMO_OUT)/trace-scene1.json \
		--out $(DEMO_OUT)/player.html \
		--title "Mnemoss · Scene 1 — Preference Recall (stub LLM)"


# ─── cleanup ───────────────────────────────────────────────────────

.PHONY: clean
clean: ## delete generated SVGs / JSONs (keeps committed demo/out/)
	rm -rf $(RESULTS_DIR)
	rm -rf $(FIGURES_DIR)
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
