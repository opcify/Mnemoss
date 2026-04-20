# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Mnemoss Is

Mnemoss is an ACT-R based memory system for AI agents, designed from the 
first principles of human cognition. It's a **standalone Python library** 
that any agent framework (OpenClaw, Hermes, Claude Code, Cursor, custom) 
can integrate.

(Pronunciation and naming — see `MNEMOSS_PROJECT_KNOWLEDGE.md` §0.)

## Repository State

This repo currently contains **design documents only** — no source code, 
no `pyproject.toml`, no tests yet. Stage 1 implementation has not started. 
Do not invent build/test/lint commands; once scaffolding exists, the 
planned stack is `pytest` / `ruff` / `mypy` (see the Dependencies section 
further down). Until then, "run the tests" is not a thing that works.

## Authoritative Documents

Read these in order before writing any code:

1. **MNEMOSS_PROJECT_KNOWLEDGE.md** — Complete project knowledge base:
   design philosophy, cognitive foundations, data model, architecture,
   API, stages, dependencies, conventions.

2. **MNEMOSS_FORMULA_AND_ARCHITECTURE.md** — The authoritative mathematical 
   formula and architecture diagrams. The formula is the heart of the system.

3. **README.md** — The public face of the project.

## Architecture at a Glance

The full design lives in the two MNEMOSS_*.md docs. This is the minimum 
needed to be productive without reading 75KB of design material first.

**Three paths, three budgets:**
- **Hot Path** (encoding, <50ms, zero LLM) — `observe()` appends to Raw 
  Log, updates Working Memory, does rule-based event segmentation, writes 
  a Memory row, updates indices.
- **Warm Path** (index maintenance, <1s, event-driven) — incremental 
  index updates on turn-end/session-end/task-done/idle.
- **Cold Path** (dreaming, opportunistic, offline) — six triggers 
  (`idle`, `session_end`, `task_completion`, `surprise`, `cognitive_load`, 
  `nightly`) drive an 8-phase pipeline: Replay → Cluster → Extract → 
  Refine → Relations → Generalize → Rebalance → Dispose. LLM is used 
  only for content generation here, never for system decisions.

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
append-only layer; Memories are event-level derivations, not raw messages.

**Planned package layout** (Stage 1 targets):
```
src/mnemoss/
  core/       types.py, config.py
  formula/    activation.py, base_level.py, matching.py, query_bias.py
  store/      SQLite + sqlite-vec + FTS5; Raw Log and Memory Store
  encoder/    embedder.py (sentence-transformers), simplified encoder
  recall/     engine.py (parallel vec+FTS → ACT-R scoring → reconsolidation)
  client.py   public Mnemoss class
```

**Public API is three methods** — `observe()`, `recall()`, `pin()` — 
plus advanced extras (`dream()`, `export_markdown()`, `status()`, 
`explain_recall()`). Everything is `async`.

## Non-Negotiable Principles

See `MNEMOSS_PROJECT_KNOWLEDGE.md` §3 for the authoritative statements. 
The eight, in brief:

1. **Formula drives everything** — no LLM in system decisions.
2. **One Memory table** holds all types (episode/fact/entity/pattern).
3. **Raw Log and Memory Store are separate layers.**
4. **Hot Path is minimal** — zero LLM calls, <50ms.
5. **Everything lazy** — fields fill on demand.
6. **Dreaming is opportunistic** — six triggers, not just nightly.
7. **Multi-tier index, unified data** — tier migration = metadata only.
8. **Disposal is formula-derived** — `max_A_i < τ − δ`, zero LLM.

## Current Development Stage

**Stage 1 — MVP Foundation**

See MNEMOSS_PROJECT_KNOWLEDGE.md Section 9 for the full stage plan.

### Stage 1 Scope (strict — do not expand)

Build ONLY these components:

- Project scaffolding (pyproject.toml, directory structure, LICENSE, .gitignore)
- Core types (`src/mnemoss/core/types.py`): Memory, RawMessage, Event, 
  Relation, enums — all carry `agent_id: str | None` for multi-agent 
  workspaces (Tombstone deferred to Stage 5)
- Config (`src/mnemoss/core/config.py`): FormulaParams, StorageParams, 
  EncoderParams (with `encoded_roles: set[str]`), MnemossConfig — 
  dataclasses only, no YAML loading
- Storage (`src/mnemoss/store/`): SQLite backend with sqlite-vec + FTS5, 
  Raw Log, Memory Store
- Embedder (`src/mnemoss/encoder/embedder.py`): abstract `Embedder` 
  protocol + two impls:
  - `LocalEmbedder` (default, zero-config) — sentence-transformers, 
    `"paraphrase-multilingual-MiniLM-L12-v2"` (384-dim, multilingual; 
    required for the Chinese success criterion).
  - `OpenAIEmbedder` (opt-in) — `openai>=1.0` (optional extra), default 
    model `"text-embedding-3-small"` at native 1536 dims. Requires 
    `OPENAI_API_KEY` in env (or explicit `api_key=`).
  
  Factory at the `Mnemoss(...)` level selects via 
  `embedding_model="local"` (default) / `"openai"` / 
  `"openai:text-embedding-3-small"` / a user-supplied `Embedder` 
  instance. Embedding dim is pinned in the workspace schema at DB 
  creation time (384 for local, 1536 for OpenAI native); opening a 
  workspace with a mismatched embedder raises.
- Simplified encoder: each `observe()` = one event = one memory (no 
  complex segmentation yet). Which roles get encoded is controlled by 
  `EncoderParams.encoded_roles` (default: all four — `user`, `assistant`, 
  `tool_call`, `tool_result`). The Raw Log still receives every message 
  unconditionally (Principle 3).
- Formula module (`src/mnemoss/formula/`):
  - `activation.py`: full equation but with spreading=0.0 placeholder
  - `base_level.py`: B_i computation
  - `matching.py`: dynamic hybrid matching
  - `query_bias.py`: simple rule-based query analysis
- Recall engine (`src/mnemoss/recall/engine.py`): parallel vec+FTS 
  retrieval, ACT-R scoring, reconsolidation, single-tier (HOT only)
- Main client (`src/mnemoss/client.py`): Mnemoss class with observe, 
  recall, pin, for_agent, status, export_markdown (stubs where 
  appropriate). `for_agent(id)` returns a thin handle that forwards 
  observe/recall/pin with `agent_id` bound.
- Top-level `__init__.py`: export public API
- Tests: unit tests for formula, storage; integration test matching 
  the success criterion below
- One example: `examples/basic.py`

### Stage 1 Out of Scope

These are explicitly NOT in Stage 1:

- Dreaming (entire cold path) — stub `dream()` to raise NotImplementedError
- Multi-tier indices (HOT/WARM/COLD/DEEP) — single HOT only
- Spreading activation — pass 0.0 placeholder
- Lazy extraction — extracted_* fields stay None
- Event segmentation complexity — one message = one event
- Tombstones — skip entirely
- memory.md generation — stub
- YAML config loading — stub
- LLM integrations — none
- Cross-agent memory promotion (Dreaming moving per-agent memories to 
  `agent_id=None`) — deferred to Stage 4+; Stage 1 keeps whatever 
  `agent_id` was written at `observe()` time

### Stage 1 Success Criterion

This code must run and produce sensible output:

```python
import asyncio
from mnemoss import Mnemoss

async def main():
    mem = Mnemoss(workspace="test")
    await mem.observe(role="user", content="我明天下午 4:20 和 Alice 见面")
    await mem.observe(role="user", content="见面地点在悉尼歌剧院旁边")
    results = await mem.recall("什么时候见 Alice?", k=3)
    for r in results:
        print(r.content)

asyncio.run(main())
```

Expected: the first message (containing "4:20") should rank highest.

## Critical Constraints (Enforced)

These are "break glass in case of temptation" rules:

1. **ZERO LLM *decision-making* in Stage 1.** No LLM invocation for 
   ranking, disposal, extraction, refinement, generalization, content 
   generation, or any system control flow. Embedder calls (local or 
   cloud OpenAI) are explicitly allowed — they produce vectors, not 
   decisions, and they're feature extraction, not judgment.

2. **No premature optimization.** Don't build for Stage 2+ capabilities 
   that aren't asked for. Leave `# TODO(Stage N):` comments instead.

3. **Pure functions in formula module.** No side effects. Easy to test.

4. **Type hints on everything public.** Use mypy.

5. **Async for all I/O.** Public API is async by default.

6. **Tests for the formula.** The formula is the heart. It must be 
   covered by unit tests.

## Working Style

- **One component at a time.** Build in dependency order:
  1. Scaffolding
  2. core/types.py + core/config.py
  3. formula/ module with tests
  4. store/ layer
  5. encoder/embedder.py
  6. encoder/ (simplified event handling)
  7. recall/engine.py
  8. client.py
  9. Integration test
  10. README + example

- **After each component**: summarize what was built, run tests, 
  wait for user confirmation before proceeding.

- **Explain trade-offs inline** when Stage 1 simplifications differ 
  from what MNEMOSS_PROJECT_KNOWLEDGE.md specifies. Add clear 
  `# TODO(Stage N)` comments.

- **If uncertain**, ask before expanding scope. When in doubt, 
  choose the simpler option.

## Settled Stage-1 Decisions

These were open questions; answers are locked in for Stage 1:

- **Empty recall results** → return `[]`, not an exception.
- **Nonexistent workspace** → auto-create on first `observe()`.
- **FTS5 tokenizer** → `trigram` (CJK-capable, bundled with modern 
  SQLite; BM25 degenerates to zero under `unicode61` for Chinese text).
- **Embedder default (zero-config)** → `LocalEmbedder` with 
  `paraphrase-multilingual-MiniLM-L12-v2` (384 dims, 50+ languages). 
  English-only `all-MiniLM-L6-v2` cannot embed the Chinese 
  success-criterion text.
- **Cloud embedder (opt-in)** → `OpenAIEmbedder` with 
  `text-embedding-3-small`, 1536 dims native. Lives behind the 
  `[openai]` extra in `pyproject.toml`; reads `OPENAI_API_KEY` from env.
- **Embedding dim** → pinned at workspace creation time to the embedder's 
  native dim (384 local, 1536 OpenAI). Stored in the schema header; 
  open-time mismatch raises. No migration, no on-the-fly MRL reduction.
- **`idx_priority` in Stage 1** → recomputed on the fly during recall 
  from latest `B_i` + protections; not persisted as authoritative. 
  Persistence + Dreaming P7 (Rebalance) lands in Stage 2+.
- **Encoder role filter** → `EncoderParams.encoded_roles` defaults to 
  all four roles (`user`, `assistant`, `tool_call`, `tool_result`); 
  user-configurable. Raw Log is still unfiltered (Principle 3).
- **Concurrent writes within one workspace** → serialize via 
  `asyncio.Lock` + SQLite WAL. Cross-process coordination deferred 
  to Stage 2.
- **Schema version** → pinned constant; mismatch raises. No migration 
  logic in Stage 1.

## Dependencies

The canonical dependency list lives in **`pyproject.toml`**. Do not mirror 
it here.

**Avoided deliberately**: LangChain, pgvector, FAISS, LlamaIndex.

## Code Style

- Python 3.10+
- Ruff for formatting and linting
- mypy strict mode on public API
- NumPy-style docstrings for public APIs
- Small modules, single responsibility
- ULIDs for all IDs (time-ordered)
- Timezone-aware datetime everywhere

## When You're Not Sure

1. First, re-read the relevant section of MNEMOSS_PROJECT_KNOWLEDGE.md 
   or MNEMOSS_FORMULA_AND_ARCHITECTURE.md.
2. If still unclear, ask — don't guess.
3. If you must guess, choose the simplest option and document it clearly.
4. Never violate the 8 non-negotiable principles.

## First Session Protocol

On the first session:

1. Read MNEMOSS_PROJECT_KNOWLEDGE.md in full.
2. Read MNEMOSS_FORMULA_AND_ARCHITECTURE.md in full.
3. Summarize back to the user (3-5 bullets):
   - What Mnemoss is
   - What Stage 1 specifically builds
   - The top three Stage-1 non-negotiables: zero LLM calls, strict Stage 1 
     scope (no premature optimization), pure + tested formula module
4. Propose a concrete plan for the first coding session.
5. Show the proposed pyproject.toml before running any install commands.

Then proceed step by step, confirming after each component.