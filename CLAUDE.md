# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Mnemoss Is

Mnemoss is an ACT-R based memory system for AI agents, designed from the 
first principles of human cognition. It's a **standalone Python library** 
plus a REST server, MCP wrapper, and framework adapters (Hermes, OpenClaw) 
that any agent stack can integrate.

(Pronunciation and naming — see the intro of `README.md`.)

## Repository State

The MVP is **shipped**. All six stages in `MNEMOSS_PROJECT_KNOWLEDGE.md` §9 
landed; the codebase is at `v0.1.0` on PyPI track. Active work is 
post-Stage-6: benchmarks, whitepaper, adapter polish, new capabilities.

Concretely, the repo ships:

- Python library (`src/mnemoss/`) — full ACT-R recall, 6-phase dreaming 
  pipeline, multi-tier index (HOT/WARM/COLD/DEEP) with cascade.
- REST server (`src/mnemoss/server/`) + Python SDK (`src/mnemoss/sdk/`) + 
  TypeScript SDK (`sdks/typescript/`).
- MCP wrapper (`src/mnemoss/mcp/`) for MCP-compatible agents.
- Scheduler (`src/mnemoss/scheduler/`) — nightly + idle dream automation.
- Observability (`status()`, structured logging, Prometheus metrics under 
  the `[observability]` extra).
- Framework adapters: `adapters/hermes-agent/` (Python Hermes 
  `MemoryProvider`) and `adapters/openclaw/` (TypeScript OpenClaw plugin).

## Quick Commands

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"
# Add extras as needed: [openai], [anthropic], [gemini], [server], [sdk], [mcp], [observability]

# Tests
pytest                          # unit + mocked
pytest -m "not integration"     # skip model-download tests (~470MB first run)
pytest -m integration           # only the multilingual integration tests
pytest tests/test_recall_unit.py::test_name  # single test

# Lint + types
ruff check src tests
ruff format src tests
mypy --strict src/mnemoss

# Run the REST server / MCP wrapper
mnemoss-server --host 0.0.0.0 --port 8000
mnemoss-mcp

# TypeScript SDK + OpenClaw adapter
cd sdks/typescript && npm install && npm test && npm run typecheck
cd adapters/openclaw && npm install && npm test && npm run typecheck
```

## Authoritative Documents

Read these in order before making non-trivial changes:

1. **MNEMOSS_PROJECT_KNOWLEDGE.md** — Design philosophy, cognitive 
   foundations, data model, architecture, stage plan, glossary.

2. **MNEMOSS_FORMULA_AND_ARCHITECTURE.md** — The authoritative 
   mathematical formula and architecture diagrams. The formula is the 
   heart of the system; every equation change goes here first.

3. **README.md** — The public face of the project.

If code and docs disagree, that's a bug. Fix whichever is wrong (usually 
the doc; flag if it's the code).

## Architecture at a Glance

The full design lives in the two MNEMOSS_*.md docs. This is the minimum 
needed to be productive without re-reading them cover-to-cover.

**Three paths, three budgets:**
- **Hot Path** (encoding, <50ms, zero LLM) — `observe()` appends to Raw 
  Log, updates Working Memory, does rule-based event segmentation, writes 
  a Memory row, updates indices.
- **Warm Path** (index maintenance, <1s, event-driven) — incremental 
  index updates on turn-end/session-end/idle.
- **Cold Path** (dreaming, opportunistic, offline) — five triggers 
  (`idle`, `session_end`, `surprise`, `cognitive_load`, `nightly`) 
  drive a 6-phase pipeline: Replay → Cluster → Consolidate → Relations → 
  Rebalance → Dispose. Consolidate is a single LLM call per cluster 
  that emits the summary, per-member refinements, and any intra-cluster 
  patterns (it replaces the former Extract / Refine / Generalize trio). 
  LLM is used only for content generation, never for system decisions.

**The formula (the heart):** a single ACT-R activation equation 
`A_i = B_i + Σ W_j·S_ji + MP·[w_F·s̃_F + w_S·s̃_S] + ε` drives 
retrieval ranking, index-tier migration (HOT/WARM/COLD/DEEP based on 
`idx_priority`), disposal (`max_A_i < τ - δ`), and reminiscence. 
**No LLM is allowed to make these decisions.**

**Data model:** one `Memory` table holds every memory type (episode, 
fact, entity, pattern), distinguished by `memory_type` and 
`abstraction_level`. Every row carries `agent_id: str | None` — a 
**workspace is a gateway** (OpenClaw term) that may hold multiple agents; 
non-null = private to that agent, null = workspace-shared (ambient, 
visible to every agent). Default recall for agent A is 
`WHERE agent_id = 'A' OR agent_id IS NULL`. The Raw Log is a separate 
append-only layer *and* a separate SQLite file (`raw_log.sqlite`); 
Memories are event-level derivations, not raw messages.

**Actual package layout:**

```
src/mnemoss/
  client.py          public Mnemoss class (+ AgentHandle via for_agent)
  core/              types.py, config.py (incl. SCHEMA_VERSION)
  formula/           activation, base_level, spreading, matching,
                     query_bias, noise, idx_priority — pure functions
  store/             sqlite_backend.py (memory.sqlite + raw_log.sqlite),
                     schema.py, paths.py
  encoder/           embedder.py (LocalEmbedder / OpenAIEmbedder /
                     FakeEmbedder), event_encoder.py
  working/           working_memory.py (per-agent FIFO active set)
  relations/         graph.py (co_occurs_in_session + dream-populated)
  recall/            engine.py (parallel vec+FTS → ACT-R scoring →
                     reconsolidation), history.py, expand.py, cascade.py
  dream/             runner.py (6-phase pipeline), consolidate.py
                     (merged P3 LLM phase), types.py (triggers)
  scheduler/         nightly + idle trigger automation
  index/             tier management (HOT/WARM/COLD/DEEP)
  export/            memory.md generator
  llm/               LLM client abstraction (Anthropic/OpenAI/Gemini)
  server/            FastAPI REST + Pydantic schemas
  sdk/               Python client for the REST server
  mcp/               MCP tool wrapper

sdks/typescript/     TypeScript SDK (@mnemoss/sdk)
adapters/
  hermes-agent/      Python Hermes MemoryProvider plugin
  openclaw/          TypeScript OpenClaw plugin (MemorySearchManager)
```

**Public API** is three core methods — `observe()`, `recall()`, `pin()` — 
plus extras: `expand()`, `dream()`, `export_markdown()`, `status()`, 
`explain_recall()`. Everything is `async`. `mem.for_agent(id)` returns 
a handle that binds `agent_id` on all calls.

## Non-Negotiable Principles

See `MNEMOSS_PROJECT_KNOWLEDGE.md` §3 for the authoritative statements. 
The eight, in brief:

1. **Formula drives everything** — no LLM in system decisions.
2. **One Memory table** holds all types (episode/fact/entity/pattern).
3. **Raw Log and Memory Store are separate layers** (and separate files).
4. **Hot Path is minimal** — zero LLM calls, <50ms.
5. **Everything lazy** — fields fill on demand.
6. **Dreaming is opportunistic** — five triggers, not just nightly.
7. **Multi-tier index, unified data** — tier migration = metadata only.
8. **Disposal is formula-derived** — `max_A_i < τ − δ`, zero LLM.

If a proposed change violates one, reject it or revise the principles 
explicitly (and update the doc).

## Architectural Invariants

Load-bearing design decisions baked into the codebase. Changing any of 
these is a schema-or-semantics change, not a refactor — bump 
`SCHEMA_VERSION` and write a migration or explicit breakage note.

- **Empty recall results** → return `[]`, not an exception.
- **Nonexistent workspace** → auto-create on first `observe()`.
- **FTS5 tokenizer** → `trigram` (CJK-capable; BM25 degenerates to zero 
  under `unicode61` for non-space-delimited scripts like Chinese, 
  Japanese, Thai). Multilingual is first-class, not a stretch goal.
- **Embedder default (zero-config)** → `LocalEmbedder` with 
  `paraphrase-multilingual-MiniLM-L12-v2` (384 dims, 50+ languages). 
  English-only models are rejected.
- **Cloud embedder (opt-in)** → `OpenAIEmbedder`, `text-embedding-3-small`, 
  1536 dims. Behind the `[openai]` extra. Reads `OPENAI_API_KEY` from env.
- **Embedding dim pinned at workspace-create time.** Stored in the schema 
  header; opening with a mismatched embedder raises 
  `SchemaMismatchError`. No migration, no on-the-fly MRL reduction.
- **Two SQLite files per workspace** — `memory.sqlite` holds Memory + 
  relations + tombstones + indices; `raw_log.sqlite` holds the append-only 
  message log. Both validated on open.
- **Encoding-grace bonus in `B_i`** — `η(t) = η₀ · exp(-(t − t_creation) / τ_η)`, 
  defaults `η₀ = 1.0`, `τ_η = 3600 s`. Configured via 
  `FormulaParams.eta_0` / `eta_tau_seconds`.
- **Encoder role filter** — `EncoderParams.encoded_roles` defaults to all 
  four roles (`user`, `assistant`, `tool_call`, `tool_result`); 
  user-configurable. Raw Log is still unfiltered (Principle 3).
- **Auto-expand on same-topic recall** — same-topic detection is purely 
  semantic (cosine ≥ `same_topic_cosine`, default 0.7); streak reset is 
  time-bound (`streak_reset_seconds`, default 600). Streak counter 
  escalates BFS hops `1 → 2 → 3` (`expand_hops_max`). The explicit 
  `mem.expand()` call exposes the same BFS.
- **Concurrent writes within one workspace** → serialized via 
  `asyncio.Lock` + SQLite WAL. Cross-process coordination is not 
  implemented; one writer process per workspace.
- **Schema version** — pinned constant in `src/mnemoss/core/config.py` 
  (`SCHEMA_VERSION`). Mismatch raises. No migration logic lives in the 
  library yet; bump when you break the format.
- **IDs are ULIDs** (time-ordered, lexicographically sortable). Datetimes 
  are timezone-aware.
- **Privacy is cooperative, not adversarial.** `for_agent` scopes queries 
  via SQL filter; workspace-level access and Dreaming can read across 
  agents. Adversarial isolation = separate workspaces, not separate 
  agents.

## Multi-Package Layout

This is a monorepo. Changes often touch more than one package:

| Path | Package | Language | Tests |
|---|---|---|---|
| `src/mnemoss/` | `mnemoss` (PyPI) | Python | `pytest` at root |
| `sdks/typescript/` | `@mnemoss/sdk` | TypeScript | `npm test` in dir |
| `adapters/hermes-agent/` | `mnemoss-hermes` | Python | `pytest` in dir |
| `adapters/openclaw/` | `@mnemoss/openclaw-plugin` | TypeScript | `npm test` in dir |

Python adapters depend on the core `mnemoss` package (source install during 
dev). TypeScript adapters depend on `@mnemoss/sdk` via `file:` ref during 
dev, npm during release.

When you change the core API, check that the SDK types, the REST schemas, 
the MCP tool list, and both adapters stay in sync. The eng-review checklist 
has a "touched-the-API" fan-out list.

## Code Style

- Python 3.10+, async-first public API
- Ruff for formatting and linting (rules: `E F W I B UP N SIM`)
- mypy strict on `src/mnemoss`
- NumPy-style docstrings for public APIs
- Small modules, single responsibility
- ULIDs for all IDs; timezone-aware datetime everywhere
- Pure functions in `formula/` — no side effects, easy to test

TypeScript side: strict `tsconfig`, ES2022, Bundler module resolution, 
Vitest for tests.

## When You're Not Sure

1. Re-read the relevant section of MNEMOSS_PROJECT_KNOWLEDGE.md or 
   MNEMOSS_FORMULA_AND_ARCHITECTURE.md.
2. Check existing tests — they pin a lot of behavior that isn't 
   doc-stated.
3. If still unclear, ask — don't guess.
4. Never violate the 8 non-negotiable principles. If a task seems to 
   require it, flag the principle that's at stake before proceeding.
