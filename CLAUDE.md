# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Mnemoss Is

Mnemoss is an ACT-R based memory system for AI agents, designed from the 
first principles of human cognition. It's a **standalone Python library** 
plus a REST server, MCP wrapper, and framework adapters (Hermes, OpenClaw, 
Claude Cowork / Claude Code) that any agent stack can integrate.

(Pronunciation and naming — see the intro of `README.md`.)

## Repository State

The MVP is **shipped** (all six stages in `MNEMOSS_PROJECT_KNOWLEDGE.md` 
§9). The codebase is at `v0.0.1` on PyPI track. A production-readiness 
pass then landed on top — cross-process workspace lock, schema 
migration framework, Dream cost governor, partial-failure recovery, 
retrying embedder wrapper, per-dataclass config validators, input 
hardening at REST/MCP boundaries, auto-split for long observes, plus 
operational tooling (`mnemoss-inspect` CLI and a `bench/` harness for 
recall latency + formula-parameter calibration). See §14 in the 
project knowledge doc for the resolved-open-questions trail.

NER is intentionally **not implemented** at any stage (query, encode, 
Dream) and is not on the roadmap — see §9.7 in the project knowledge 
doc for rationale and DIY hooks.

Concretely, the repo ships:

- Python library (`src/mnemoss/`) — full ACT-R recall, 6-phase dreaming 
  pipeline, multi-tier index (HOT/WARM/COLD/DEEP) with cascade.
- REST server (`src/mnemoss/server/`) + Python SDK (`src/mnemoss/sdk/`) + 
  TypeScript SDK (`sdks/typescript/`).
- MCP wrapper (`src/mnemoss/mcp/`) for MCP-compatible agents.
- Scheduler (`src/mnemoss/scheduler/`) — nightly + idle dream automation.
- `mnemoss-inspect` CLI (`src/mnemoss/cli/inspect.py`) — operator snapshot 
  of a live workspace (`mnemoss-inspect <workspace> [--json] [--tombstones]`).
- Cost governor (`src/mnemoss/dream/cost.py`) — `CostLimits` + 
  `CostLedger`, configurable ceilings on LLM calls per run / day / 
  month; counts persist in `workspace_meta`.
- Retry wrapper (`src/mnemoss/encoder/retrying.py`) — `RetryingEmbedder` 
  around any `Embedder` with bounded exponential backoff + jitter on 
  transient errors.
- Schema migration framework (`src/mnemoss/store/migrations.py`) — 
  registered per-version steps, applied automatically on open.
- Cross-process workspace lock (`src/mnemoss/store/_workspace_lock.py`) — 
  OS-level `fcntl` / `msvcrt` advisory lock so a second process can't 
  corrupt a live workspace.
- Long-content auto-split (`src/mnemoss/encoder/chunking.py`) — opt-in 
  via `EncoderParams.max_memory_chars`; splits at paragraph → line → 
  sentence → hard-cut boundaries.
- Observability (`status()` with cost + recent-dream summary, structured 
  logging, Prometheus metrics under the `[observability]` extra).
- Benchmark harness (`bench/bench_recall.py`, `bench/calibrate.py`) — 
  standalone scripts for recall-latency and formula-parameter sweeps.
- Framework adapters: `adapters/hermes-agent/` (Python Hermes 
  `MemoryProvider`), `adapters/openclaw/` (TypeScript OpenClaw plugin),
  and `adapters/claude-cowork/` (Claude Cowork / Claude Code plugin
  bundling the existing `mnemoss-mcp` server plus four skills).

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
mnemoss-server                             # reads MNEMOSS_* env vars
mnemoss-mcp                                # MCP stdio transport

# Inspect a live workspace
mnemoss-inspect <workspace>                # human-readable table
mnemoss-inspect <workspace> --json         # JSON for scripting
mnemoss-inspect <workspace> --tombstones   # include disposal audit trail

# Benchmarks (standalone, outside pytest)
python -m bench.bench_recall --sizes 100 1000 5000 --queries 50
python -m bench.calibrate --demo           # or: python -m bench.calibrate corpus.json

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

**The async-ACT-R bet (recall mode = fast_index).** Mnemoss's defining 
architectural choice: all expensive cognition — activation formula 
evaluation, spreading, relationship propagation, tier migration — runs 
**off the read path**. Recall itself is: (1) ANN top-K via HNSW, (2) 
batch-fetch cached `idx_priority`, (3) linear combine + sort. No FTS, 
no per-candidate formula, no noise, no tier cascade. The cost that 
grows with N (O(N) vector/FTS scan) happens during Dream, not during 
the user-facing `recall()` call. Enable with 
`FormulaParams(use_fast_index_recall=True)` — off by default to 
preserve the full ACT-R read path for agent workloads that want 
recency priors and query-dependent matching weights.

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
  core/              types.py, config.py (incl. SCHEMA_VERSION + param
                     dataclasses w/ __post_init__ validators),
                     config_file.py (TOML loader)
  formula/           activation (w/ ActivationBreakdown.to_dict),
                     base_level, spreading, matching, query_bias,
                     noise, idx_priority — pure functions
  store/             sqlite_backend.py (façade) + _memory_ops.py /
                     _graph_ops.py / _raw_log_ops.py / _sql_helpers.py
                     (sync SQL split), schema.py, paths.py,
                     migrations.py (versioned chain),
                     _workspace_lock.py (cross-process advisory lock)
  encoder/           embedder.py (LocalEmbedder / OpenAIEmbedder /
                     GeminiEmbedder / FakeEmbedder),
                     retrying.py (RetryingEmbedder wrapper),
                     chunking.py (long-content splitter),
                     event_encoder.py, event_segmentation.py,
                     extraction.py (gist + time only),
                     salience.py (encoder-side signal mixing)
  working/           working_memory.py (per-agent FIFO active set)
  relations/         graph.py (co_occurs_in_session + dream-populated)
  recall/            engine.py (parallel vec+FTS → ACT-R scoring →
                     reconsolidation), history.py, expand.py, cascade.py
  dream/             runner.py (6-phase pipeline + try/except per
                     phase), consolidate.py (merged P3 LLM phase),
                     cost.py (CostLimits + CostLedger), dispose.py,
                     replay.py, cluster.py, relations.py, diary.py,
                     types.py (trigger / phase enums, DreamReport)
  scheduler/         nightly + idle trigger automation
  index/             tier management (HOT/WARM/COLD/DEEP) + rebalance
  export/            memory.md generator
  llm/               client.py (LLMClient Protocol + OpenAI / Anthropic
                     / Gemini implementations), mock.py
  server/            FastAPI REST app (w/ input hardening caps),
                     auth, pool, schemas, metrics, config, cli
  sdk/               Python client for the REST server
  mcp/               MCP tool wrapper (server, backend, tools, cli)
  cli/               inspect.py (mnemoss-inspect operator CLI)

bench/               Standalone benchmark harnesses — run with
                     python -m bench.<name>. bench_recall.py sweeps
                     recall latency; calibrate.py sweeps FormulaParams
                     against a labeled corpus.

sdks/typescript/     TypeScript SDK (@mnemoss/sdk)
adapters/
  hermes-agent/      Python Hermes MemoryProvider plugin
  openclaw/          TypeScript OpenClaw plugin (MemorySearchManager)
  claude-cowork/     Claude Cowork / Claude Code plugin (.mcp.json + skills)
```

**Public API** is three core methods — `observe()`, `recall()`, `pin()` — 
plus extras: `expand()`, `dream()`, `export_markdown()`, `status()`, 
`explain_recall()`, `rebalance()`, `dispose()`, `tombstones()`, 
`flush_session()`. Everything is `async`. `mem.for_agent(id)` returns 
a handle that binds `agent_id` on all calls.

Construction knobs worth knowing:
- `Mnemoss(cost_limits=CostLimits(max_llm_calls_per_run=50, ...))` — 
  caps dreaming LLM spend; counts persist via `CostLedger` in 
  `workspace_meta` and surface through `status().llm_cost`.
- `Mnemoss(embedding_model=RetryingEmbedder(OpenAIEmbedder(), max_retries=3))` — 
  composition pattern for flaky providers. The wrapper passes `dim` 
  / `embedder_id` through transparently so the workspace schema pin 
  still matches.
- `EncoderParams(max_memory_chars=2000)` — opt-in long-observe 
  auto-split (see invariant below).

Errors to catch at boundaries: `SchemaMismatchError`, 
`WorkspaceLockError`, `MigrationError`, `CostExceededError` (all from 
`mnemoss.store` / `mnemoss.dream`). `ValueError` comes out of param 
dataclass validators at construction time.

`explain_recall(query, memory_id)` returns `ActivationBreakdown | None` 
with a `.to_dict()` for wire-safe export. `dream(...)` returns a 
`DreamReport` with `degraded_mode: bool` + `errors(): list[PhaseOutcome]` 
so callers can distinguish "worked" from "partially crashed."

## Non-Negotiable Principles

See `MNEMOSS_PROJECT_KNOWLEDGE.md` §3 for the authoritative statements. 
The nine, in brief:

1. **Formula drives everything** — no LLM in system decisions.
2. **One Memory table** holds all types (episode/fact/entity/pattern).
3. **Raw Log and Memory Store are separate layers** (and separate files).
4. **Hot Path is minimal** — zero LLM calls, <50ms.
5. **Everything lazy** — fields fill on demand.
6. **Dreaming is opportunistic** — five triggers, not just nightly.
7. **Multi-tier index, unified data** — tier migration = metadata only.
8. **Disposal is formula-derived** — `max_A_i < τ − δ`, zero LLM.
9. **`idx_priority` is for ranking, not search** — drives tier
   membership, disposal, export filtering. Search uses pure cosine
   within tiers. Mixing the two collapses recall on aged corpora.

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
- **`memory_fts` is single-column** — indexes `content` only (trigram
  tokenizer). Do not add an entities column or any other secondary
  FTS field without explicit design discussion; NER is intentionally
  out of scope (§9.7 in PROJECT_KNOWLEDGE).
- **Query bias `b_F(q)` is structural only** — `{1.0, 1.2, 1.3, 1.4, 
  1.5}` from quotes/backticks (1.5), URL/email/path (1.4), 
  time/date/number/hashtag/@mention/CamelCase/snake_case/kebab/version 
  (1.3), ALL-CAPS acronym (1.2), neutral (1.0). Every rule is a regex 
  on typographic markers; no NER, no vocabulary, no language detection. 
  See `src/mnemoss/formula/query_bias.py`.
- **No automatic entity extraction anywhere.** Level-1 heuristic
  fills `gist` + `time`; Dream P3 refines `gist` + `time` at level=2.
  `extracted_entities`, `extracted_location`, `extracted_participants`
  stay `None` unless a caller writes them manually. See §9.7 for
  DIY guidance if you want entity features in your own fork.
- **Embedder default (zero-config)** → `LocalEmbedder` with 
  `paraphrase-multilingual-MiniLM-L12-v2` (384 dims, 50+ languages). 
  English-only models are rejected.
- **Cloud embedder (opt-in)** → `OpenAIEmbedder`, `text-embedding-3-small`, 
  1536 dims. Behind the `[openai]` extra. Reads `OPENAI_API_KEY` from env.
- **Cloud embedder (opt-in, Google)** → `GeminiEmbedder`, 
  `gemini-embedding-001`, 3072 dims native (768/1536 via MRL `dim=`, 
  renormalized on return). Behind the `[gemini]` extra. Reads 
  `GEMINI_API_KEY` or `GOOGLE_API_KEY` from env.
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
- **Auto-split on long observes** — `EncoderParams.max_memory_chars` 
  (default `None`, meaning no split — backward compatible) caps the 
  `content` length of any single Memory row. When an `observe()` 
  produces a Memory whose content exceeds the cap, the encoder splits 
  at the nearest natural boundary — paragraph (`\n\n`) → line (`\n`) → 
  sentence (CJK-aware: `.!?…。！？`) → hard char cut — and emits **N 
  Memory rows** for one Raw Log row. The Raw Log is still 1-to-1 with 
  the `observe()` call (Principle 3 preserved); only the Memory table 
  fans out. Every chunk carries 
  `source_context.split_part = {"index": i, "total": n, "group_id": <first_chunk_id>}` 
  so callers can de-duplicate in recall if they want. Recommended 
  values: `2000` for `LocalEmbedder` (MiniLM truncates past ~512 
  tokens); `30000` for OpenAI `text-embedding-3-small`. Rationale: 
  without a cap, embedders silently drop tokens past their max and 
  semantic recall degrades invisibly; see `encoder/chunking.py`.
- **Auto-expand on same-topic recall** — same-topic detection is purely 
  semantic (cosine ≥ `same_topic_cosine`, default 0.7); streak reset is 
  time-bound (`streak_reset_seconds`, default 600). Streak counter 
  escalates BFS hops `1 → 2 → 3` (`expand_hops_max`). The explicit 
  `mem.expand()` call exposes the same BFS.
- **Concurrent writes within one workspace** → serialized via 
  `asyncio.Lock` + SQLite WAL **within a process**, and by an OS-level 
  advisory lock (`fcntl.flock` on unix, `msvcrt.locking` on windows) 
  on `{workspace_dir}/.mnemoss.lock` **across processes**. A second 
  process opening the same workspace raises `WorkspaceLockError`. The 
  lock file is created on first open and released on `close()` or 
  process exit. Also `PRAGMA busy_timeout=5000` on both DBs so intra-
  process WAL contention retries instead of raising SQLITE_BUSY.
- **Schema migrations** — `src/mnemoss/store/migrations.py` owns a 
  registered chain of single-version-step migrations. On open, if the 
  stored `schema_version` is older than code, the chain applies inside 
  one transaction. Newer-than-code raises `MigrationError`. Each 
  migration bumps by exactly one version and its closure takes an 
  `apsw.Connection`; the framework owns the version-marker update. 
  Adding a schema change: bump `SCHEMA_VERSION`, append a 
  `Migration(from_version=N, to_version=N+1, description=…, fn=…)` 
  to `MIGRATIONS`, and cover in `tests/test_migrations.py`.
- **Dream cost governor** — `dream/cost.py` ships `CostLimits` 
  (`max_llm_calls_per_run` / `_per_day` / `_per_month`, `None` = 
  unlimited) and `CostLedger` (persists per-day + all-time counts in 
  `workspace_meta`). `Mnemoss(cost_limits=...)` plumbs them through 
  `dream()` → `DreamRunner._phase_consolidate` which checks budget 
  before each cluster's LLM call and records skip reasons in 
  `PhaseOutcome.details.budget_skips`. Ledger counts survive close/
  reopen; they're visible in `status().llm_cost`.
- **Partial-failure recovery** — `PhaseOutcome.status` ∈ `{"ok", 
  "skipped", "error"}`, with typed `skip_reason` and `error` fields. 
  The runner wraps every phase in try/except; a raise never kills 
  the dream run, it's recorded as `status="error"` and downstream 
  phases still attempt. `DreamReport.degraded_mode` is `True` iff any 
  phase errored. The last N dream runs (bounded in-memory at 10) 
  show up in `status().dreams.recent` as lightweight summaries.
- **Schema version** — pinned constant in `src/mnemoss/core/config.py` 
  (`SCHEMA_VERSION`, currently 8). On version drift, the migration 
  framework upgrades the DB; on a newer DB than code, it raises. In 
  the old no-migration world bumps required manual workspace rebuilds; 
  now the chain handles older DBs automatically.
- **Config parameters validated at construction** — `FormulaParams`, 
  `EncoderParams`, `SegmentationParams`, `MnemossConfig`, and 
  `CostLimits` all run `__post_init__` validators. Negative scalars, 
  zero where forbidden, out-of-order tier offsets, empty workspace 
  strings, and bool-as-int for cost caps raise `ValueError` at 
  construction rather than at first recall. See `tests/test_config_validation.py` 
  for the rejection surface.
- **Retry wrapper for flaky embedders** — `RetryingEmbedder` wraps 
  any `Embedder` with bounded exponential-backoff + jitter retries 
  on `ConnectionError` / `TimeoutError` / `OSError` + any extra types 
  passed via `retry_on=`. `ValueError` is never retried. `dim` and 
  `embedder_id` pass through transparently so the schema pin matches 
  the underlying embedder. Opt-in; users compose:
  `RetryingEmbedder(OpenAIEmbedder(...), max_retries=3)`.
- **`status()` shape** — returns a `json.dumps`-able dict with 
  `workspace`, `schema_version`, `embedder`, `memory_count`, 
  `tier_counts`, `tombstone_count`, `last_observe_at` / `last_dream_at` 
  / `last_rebalance_at` / `last_dispose_at`, `last_dream_trigger`, and 
  two new blocks: `llm_cost` (today/month/total calls + configured 
  limits) and `dreams` (bounded list of recent run summaries + 
  degraded count). Every value is primitive / list / dict — no 
  datetimes or dataclasses leak in.
- **`ActivationBreakdown.to_dict()`** — JSON-safe view of the ACT-R 
  scoring decomposition. `explain_recall` returns 
  `ActivationBreakdown | None`; callers can ship the decomposition 
  over REST / MCP / logs via `breakdown.to_dict()` without custom 
  encoders.
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
| `adapters/claude-cowork/` | Claude Cowork / Code plugin | JSON + Markdown | `claude plugin validate` |

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
