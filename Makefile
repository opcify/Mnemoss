# Mnemoss launch work: common commands.
#
# The design doc (~/.gstack/projects/opcify-Mnemoss/...-design-*.md)
# cites `make launch-bench` as the one-command reproducibility path
# for Chart 1. That lives here.
#
# Everything else in the file is a thin convenience wrapper around
# `python -m ...` commands the rest of the project exposes. No magic.

# ─── .env auto-load ────────────────────────────────────────────────
#
# Copy .env.example to .env and fill in your keys. The `-include`
# (with the leading dash) makes it optional: `.env` missing is not
# an error. The `export` line propagates the listed vars to every
# recipe's subprocess (which is how Python sees them via os.environ).
# The Python CLIs also call load_dotenv() themselves as a backstop
# for direct invocation without `make`.

-include .env
export OPENAI_API_KEY GEMINI_API_KEY GOOGLE_API_KEY MNEMOSS_BENCH_BUDGET_USD

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
