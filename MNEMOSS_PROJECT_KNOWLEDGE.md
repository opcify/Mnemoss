# Mnemoss Project Knowledge Base

> Single source of truth for Mnemoss development.
> This document is read by humans, Claude Code, and any future collaborator.
> Keep it up to date as design evolves.

---

## 0. TL;DR

Mnemoss is an **ACT-R based memory system for AI agents**, designed from the 
first principles of human cognition. It is a **standalone library** that can 
be integrated into any agent framework (OpenClaw, Hermes, Claude Code, Cursor, 
or custom agents) via three integration layers: Python API, MCP Server, and 
Framework Plugins.

**One-sentence pitch:**
> Memory, the way your mind actually works. Recall what matters, when it matters.

---

## 1. Why Mnemoss Exists

### 1.1 The Problem

Most "agent memory" systems today (mem0, Zep, Letta, built-in systems in 
OpenClaw/Hermes/Claude Code) are variations of **RAG + vector database**. 
They search for similar text but have no notion of:

- Time dynamics (when was this memory last accessed?)
- Frequency effects (how often has it been useful?)
- Context sensitivity (what's active in working memory right now?)
- Abstraction hierarchy (is this an episode, a fact, or a pattern?)
- Cue-driven reminiscence (can a specific cue recover a seemingly forgotten memory?)

The result: agents "remember" in a shallow way. They retrieve similar text 
but can't reason about relationships, miss context, and lose specific 
decisions to vague summaries over time.

### 1.2 The Insight

Human memory is not about "storing vs. forgetting." It's about:

1. **Retrieval priority distribution** — not everything is equally accessible
2. **Opportunistic consolidation** — integration happens during idle moments, 
   not just at night
3. **Cue-driven recall** — the right cue can activate even "forgotten" memories
4. **Storage is cheap, retrieval is expensive** — optimize the latter

### 1.3 The Solution

Mnemoss implements **ACT-R** (Adaptive Control of Thought—Rational), the 
most empirically validated cognitive architecture, as the single mathematical 
core of the system. Every behavior — retrieval, indexing, disposal, 
reminiscence — emerges from one activation formula. No ad-hoc decision logic, 
no LLM-based judgment for system behaviors.

---

## 2. Cognitive Foundations

Every design decision in Mnemoss maps to a published cognitive theory:

| Theory | Author | Year | Role in Mnemoss |
|---|---|---|---|
| Memory Unified Representation | Eichenbaum | 2000s | Single Memory table, no Episodic/Semantic physical split |
| ACT-R Activation Equation | Anderson & Schooler | 1991 | Core retrieval scoring formula |
| Storage ≠ Accessibility ≠ Availability | Tulving | 1974 | Formula outputs accessibility; data is always stored |
| Encoding Specificity | Tulving | 1973 | Lazy extraction at query time, not encoding time |
| Context-Dependent Memory | Tulving | 1973 | DEEP index triggered by strong cues |
| Dual-Process Theory | Yonelinas | 2002 | FTS/Semantic dynamic weighting |
| Transfer-Appropriate Processing | Morris & Bransford | 1977 | Encoding-retrieval mode alignment |
| Awake Replay | Foster & Wilson | 2006 | Opportunistic dreaming triggers |
| Consolidation Theory | McClelland & O'Reilly | 1995 | Dreaming pipeline |
| Event Segmentation Theory | Zacks | 2007 | Raw Log → Memory boundary detection |
| Retrieval-Induced Forgetting | Anderson | 2003 | Index competition model, not data decay |

---

## 3. Core Design Principles (Non-Negotiable)

These eight principles govern every decision. If a proposed change violates one, 
reject it or revise the principles explicitly.

**Principle 1: Formula drives everything.** 
Retrieval ranking, index migration, disposal judgment, reminiscence — all 
emerge from the ACT-R activation formula. No LLM-based system decisions.

**Principle 2: One table holds all memory types, scoped by agent.**
Episodes, facts, entities, patterns all live in a single Memory table, 
distinguished by `memory_type` and `abstraction_level`. Each row carries 
`agent_id: str | None` — non-null is private to that agent, null is 
workspace-shared (visible to every agent in the workspace/gateway).

**Principle 3: Raw Log and Memory Store are separate layers.**
All raw messages go into an append-only log (audit, replay). Memory objects 
are event-level units derived via event segmentation.

**Principle 4: Hot Path is minimal.**
Encoding path uses zero LLM calls and completes in <50ms. All LLM work is 
deferred to Warm/Cold paths.

**Principle 5: Everything lazy.**
Don't pre-structure. Fields fill on query demand, relations build during 
dreaming, facts emerge from repetition clustering.

**Principle 6: Dreaming is opportunistic.**
Five triggers (idle, session-end, surprise, cognitive-load, nightly) drive 
a 6-phase pipeline. Integration happens when signals indicate it should, 
not on a fixed schedule.

**Principle 7: Multi-tier index, unified data.**
Four index tiers (HOT/WARM/COLD/DEEP) with latency gradient from <10ms to 
<500ms. Data exists in one place; "migration" only updates metadata.

**Principle 8: Disposal is formula-derived.**
A memory is dropped only when `max_A_i < τ - δ` — meaning it cannot be 
retrieved even under the most favorable conditions. Plus geometric methods 
(clustering, cosine coverage) for redundancy. Zero LLM decisions.

---

## 4. The Core Formula

The mathematical specification — the ACT-R activation equation, all four 
components (base level, spreading, dynamic hybrid matching, noise), the 
derived index-priority, retrieval threshold, disposal criterion, 
reconsolidation feedback, and all parameter defaults — lives in 
**`MNEMOSS_FORMULA_AND_ARCHITECTURE.md` Part I**. That document is the 
single source of truth. Do not duplicate equations here — they drift.

In one paragraph: a single scalar activation `A_i` combines base-level 
history (`B_i` — power-law decay over the memory's access history, plus 
a short-lived encoding-grace bonus that lifts fresh memories into HOT 
and fades over ~1 hour), spreading activation (context from Working 
Memory via the relation graph), dynamic hybrid matching (BM25 literal 
score + semantic cosine, adaptively weighted per-memory by `idx_priority` 
and per-query by a bias term), and a small Logistic noise. Every system 
behavior — retrieval ranking, index tier placement, disposal, 
reminiscence — is derived from this one number.

---

## 5. Data Model

### 5.1 Core Memory Object

```python
@dataclass
class Memory:
    # Identity
    id: str                              # ULID
    workspace_id: str
    agent_id: str | None                 # Non-null = private to agent; null = workspace-shared
    session_id: str | None
    created_at: datetime
    
    # Content
    content: str                         # Synthesized event text / fact / entity name
    content_embedding: np.ndarray
    role: str | None                     # user | assistant | tool (for episodes)
    
    # Type & abstraction
    memory_type: MemoryType              # "episode" | "fact" | "entity" | "pattern"
    abstraction_level: float             # 0.0 (concrete) → 1.0 (abstract)
    
    # Dynamics (formula-driven)
    access_history: list[datetime]
    rehearsal_count: int
    salience: float
    emotional_weight: float
    reminisced_count: int
    
    # Index (formula-derived)
    idx_priority: float
    index_tier: IndexTier                # "hot" | "warm" | "cold" | "deep"
    last_reindexed_at: datetime | None
    
    # Graph structure
    relations: list[Relation]
    derived_from: list[str]              # Abstractions derived from these more concrete memories
    derived_to: list[str]                # More abstract memories derived from me
    
    # Traceability to Raw Log
    source_message_ids: list[str]
    
    # Lazy fields (filled on demand, never nulled)
    extracted_gist: str | None
    extracted_entities: list[str] | None
    extracted_time: datetime | None
    extracted_location: str | None
    extracted_participants: list[str] | None
    extraction_level: int                # 0=raw, 1=heuristic (gist+time),
                                         # 2=Dream P3 LLM (gist+time only;
                                         # NER is intentionally not wired —
                                         # see §9.7)
    
    # Type-specific (entity only)
    aliases: list[str] | None
    entity_type: str | None
    
    # Clustering metadata (from Dreaming P2)
    cluster_id: str | None
    cluster_similarity: float | None
    is_cluster_representative: bool
    
    # User control
    pinned_fields: set[str]
    manually_marked: str | None          # "important" | "always_inject" | None
    
    source_context: dict
```

### 5.2 Supporting Types

```python
@dataclass
class Relation:
    predicate: str                       # "manages" | "supersedes" | "part_of" | ...
    target_id: str
    confidence: float
    created_at: datetime


@dataclass
class RawMessage:
    id: str
    workspace_id: str
    agent_id: str | None                 # None = workspace-ambient observation
    session_id: str
    turn_id: str
    parent_id: str | None
    timestamp: datetime
    role: str                            # user | assistant | tool_call | tool_result
    content: str                         # Original, never modified
    metadata: dict


@dataclass
class Event:
    """A segmented event — a group of messages forming one episode"""
    id: str
    agent_id: str | None                 # Inherited from constituent RawMessages
    session_id: str
    messages: list[RawMessage]
    started_at: datetime
    ended_at: datetime
    closed_by: str                       # "turn_complete" | "topic_shift" | "time_gap" | ...


@dataclass
class Tombstone:
    """Memento of a disposed memory"""
    original_id: str
    dropped_at: datetime
    reason: str                          # "activation_dead" | "redundant" | "fact_covered"
    gist_snapshot: str
    B_at_drop: float
    source_message_ids: list[str]        # Points back to Raw Log for reconstruction
```

### 5.3 Multi-Agent Scoping

A **workspace** is a single memory namespace equivalent to one OpenClaw 
gateway. Multiple **agents** coexist within a workspace.

**`agent_id` rules:**

- `Memory.agent_id: str | None` — non-null is private to that agent; null 
  is workspace-shared (visible to every agent in the gateway).
- `RawMessage.agent_id` and `Event.agent_id` follow the same rule. 
  `agent_id=None` is a legitimate workspace-ambient observation.
- Default recall for agent `A` returns `WHERE agent_id = 'A' OR 
  agent_id IS NULL`. An agent sees its own memories plus all 
  workspace-shared ones, never another agent's private memories.
- `access_history` on a shared memory is a single merged list; `B_i` 
  reflects "hot across the gateway" rather than per-agent heat.
- Working Memory is per-agent (spreading activation does not cross 
  agents).
- `pin()` is per-agent.
- One SQLite DB per workspace (do not shard by agent — joins are needed 
  for the shared-memory query path).

**Write-side behavior:** all memories written via 
`mem.for_agent("A").observe(...)` get `agent_id="A"`; all memories 
written via the top-level `mem.observe(...)` get `agent_id=None`. 
Cross-agent promotion (Dreaming's Consolidate phase emitting a summary 
derived from a cluster that spans multiple agents as `agent_id=None`) 
shipped in Stage 4. After the P3/P4/P6 merger it's still the same rule 
— the Consolidate call looks at the cluster's agent set and promotes 
to ambient when >1 agent is represented.

**Privacy model — cooperative, not adversarial.** The `for_agent` handle 
enforces agent isolation at the recall/pin surface (via the SQL filter 
above), so agent A's query cannot return agent B's private memories. But 
the workspace-level `Mnemoss` object, Dreaming (which must read across 
agents to extract shared patterns), and any direct SQLite access can read 
every agent's memories. "Private" here means *contextually scoped*, not 
*access-controlled*. For adversarial isolation between mutually-distrusting 
actors, use separate **workspaces** (separate SQLite DBs), not separate 
**agents** within one workspace. One gateway's agents trust the gateway.

---

## 6. Architecture Overview

### 6.1 Three Paths

**Hot Path (<50ms, no LLM):** Encoding
- Raw Log append (all messages, unconditional)
- Working Memory update
- Event segmentation (rule-based)
- On event closure: encode to Memory, store, index

**Warm Path (<1s, event-driven):** Index maintenance
- Triggered on: turn-end, session-end, idle
- Incremental index updates
- Async LLM refinement queue

**Cold Path (opportunistic, offline):** Dreaming
- Five triggers drive a 6-phase pipeline
- LLM used only for content generation (not system decisions)
- Consolidate is one LLM call per cluster — merged from the former 
  Extract / Refine / Generalize trio

### 6.2 Dreaming Pipeline

Six phases, selected per trigger:

| Phase | Purpose |
|---|---|
| P1: Replay | Select memories by ACT-R B_i |
| P2: Cluster | HDBSCAN on embeddings, set cluster metadata |
| P3: Consolidate | **One LLM call per cluster** emits: (a) summary memory (fact/entity/pattern), (b) refined `extracted_*` fields for every member, (c) intra-cluster patterns. Replaces the former Extract + Refine + Generalize phases. |
| P4: Relations | Update memory-to-memory relations from cluster co-membership and `derived_from` edges; handle conflicts (supersedes) |
| P5: Rebalance | Recompute `idx_priority` for all memories, migrate between tiers |
| P6: Dispose | Formula-driven disposal, write tombstones |

Cross-cluster generalization (the former P6 scanning all newly-
extracted facts run-wide) is intentionally dropped: clusters are
already the semantic boundary, and patterns that truly span multiple
clusters are rare. Intra-cluster patterns remain, emitted inside the
same Consolidate call.

### 6.3 Trigger → Phase Mapping

```python
PHASES_BY_TRIGGER = {
    "idle":            ["replay", "cluster", "consolidate", "relations"],
    "session_end":     ["replay", "cluster", "consolidate", "relations"],
    "surprise":        ["consolidate", "relations"],
    "cognitive_load":  ["consolidate"],
    "nightly":         ["replay", "cluster", "consolidate", "relations",
                        "rebalance", "dispose"],
}
```

### 6.4 Integration Layers

Mnemoss provides three integration surfaces:

1. **Python API** — `import mnemoss; mem = Mnemoss(workspace="x")`
2. **MCP Server** — for any MCP-compatible agent (Claude Code, Cursor, etc.)
3. **Framework Plugins** — deep integration (OpenClaw plugin, Hermes provider)

### 6.5 memory.md — User-Facing View

Generated from memory store by Dreaming (not independent data):
- High idx_priority + high abstraction + pinned + frequently accessed
- Rendered deterministically to Markdown (no LLM)
- Injected into system prompt at session start
- User edits via `memory_overrides.md` (no bidirectional sync complexity)

---

## 7. Public API (Minimal)

### 7.1 Core API

A workspace is a gateway-level namespace that may hold multiple agents. 
The public surface has two entry points: the workspace-level `Mnemoss` 
object (writes/reads ambient, workspace-shared memory) and per-agent 
handles obtained via `mem.for_agent(id)`.

```python
from mnemoss import Mnemoss

mem = Mnemoss(workspace="gateway_a")

# ── Workspace-level (ambient) ─────────────────────────────
# These memories are visible to every agent in the gateway.

await mem.observe(role="user", content="...")          # agent_id=None
results = await mem.recall(query="...", k=5)           # union across all agents + ambient

# ── Per-agent (private) ───────────────────────────────────
# Bind once, call many. The handle is sugar over an internal
# agent_id= parameter.

alice = mem.for_agent("alice")
await alice.observe(role="user", content="...")        # agent_id="alice"
results = await alice.recall(query="...", k=5)         # WHERE agent_id="alice" OR IS NULL
await alice.pin(memory_id)                             # pin is per-agent
```

### 7.2 Advanced API

```python
await mem.dream(scope="session")            # Force dreaming (always cross-agent by design)
await mem.export_markdown()                 # Workspace-level memory.md (all agents + ambient)
await alice.export_markdown()               # Per-agent memory.md (agent_id="alice" OR NULL)
await mem.status()                          # System state
await mem.explain_recall(query, memory_id)  # Debug: why was this returned?
```

`export_markdown()` follows the same scoping rule as `recall()`: called on 
the root `Mnemoss`, it produces a workspace-wide view; called on a 
`for_agent(id)` handle, it produces that agent's view (private + ambient).

### 7.3 Events (for integrations)

```python
mem.on("memory_encoded", callback)
mem.on("dream_completed", callback)
mem.on("memory_disposed", callback)
```

### 7.4 Configuration

```python
Mnemoss(
    workspace="gateway_a",               # gateway-level namespace; agents via mem.for_agent(id)
    storage_path=None,                   # Default: ~/.mnemoss/workspaces/{workspace}
    embedding_model="local",             # Shipped options:
                                         #   "local"                         → paraphrase-multilingual-MiniLM-L12-v2 (384d, zero-config)
                                         #   "openai"                        → text-embedding-3-small (1536d; needs OPENAI_API_KEY + [openai] extra)
                                         #   "openai:<model-id>"             → explicit OpenAI model pick
                                         #   Embedder instance               → user-supplied, implements the Embedder protocol
                                         # Dim is pinned in the workspace schema at create time; switching later raises.
    formula_params=FormulaParams(...),
    encoder_params=EncoderParams(
        encoded_roles={"user", "assistant", "tool_call", "tool_result"},
    ),                                   # which Raw Log roles become Memory rows
    dreaming=DreamingConfig(...),
    disposal=DisposalConfig(...),
)
```

---
## 8. Package Structure

See [`CLAUDE.md`](./CLAUDE.md) §"Architecture at a Glance" for the 
current `src/mnemoss/` layout, plus the monorepo's TypeScript SDK and 
framework adapters. Don't mirror the tree here — it drifts. The 
authoritative view is `tree src/mnemoss` against a clean checkout.

---

## 9. Stage-Based Development Plan

> **Status — all six stages shipped.** The sections below are kept as a 
> record of how the system was built and why each layer exists. They are 
> not a forward roadmap. For current state and active invariants, see 
> `CLAUDE.md`. For the next wave of work (benchmarks, whitepaper, 
> additional adapters, performance tuning), see the issue tracker.

### Stage 1 — MVP Foundation ✅

**Goal:** End-to-end observe → recall works with the **full activation 
formula** engaged. API is clean.

**Scoping principle — the formula is not progressive.** The four terms of 
the activation equation ($B_i$, spreading, matching, noise) interact: 
a memory's rank at retrieval is a sum across all of them, and partial 
implementations (spreading=0 placeholder, etc.) produce misleading 
behavior that doesn't validate the cognitive design. Stage 1 therefore 
ships the formula as **one integrated unit**. The *architecture* around 
the formula (dreaming, multi-tier indices, lazy extraction) remains 
progressive.

**Scope:**
- Core types (Memory, RawMessage, Event, Relation) — all carry 
  `agent_id: str | None`
- SQLite backend + Raw Log + Memory Store
- Local embedding (sentence-transformers, multilingual)
- Simplified encoder (each message = one event, no complex segmentation)
- **Full activation formula**:
  - $B_i$ with encoding-grace $\eta(t)$
  - Spreading activation $\sum W_j \cdot S_{ji}$ with fan effect
  - Dynamic hybrid matching with normalized softmax-style weights
  - Logistic noise
- **Supporting state for the formula:**
  - Working Memory: per-agent FIFO active set $\mathcal{C}$ (default 
    capacity 10), populated by `observe()` and by returned `recall()` 
    results. Drives spreading.
  - Relation graph: basic co-occurrence edges (`co_occurs_in_session`) 
    written at encode time. `fan_j` recomputed on the fly. Richer 
    relation types (supersedes, part_of, etc.) come from Dreaming P5 
    in Stage 4+.
- Single-tier index (HOT only, sqlite-vec + FTS5; multi-tier came in Stage 2, still on sqlite-vec rather than a separate HNSW binary)
- `idx_priority` recomputed on the fly from latest $B_i$ (persistence 
  + tier migration = Stage 2)
- Core API: `observe`, `recall`, `pin`, plus `mem.for_agent(id)` handle 
  for per-agent scoping. Workspace-level calls write ambient memories 
  (`agent_id=None`).
- Multilingual from day one: trigram FTS5 tokenizer, multilingual 
  embedder, no English-specific heuristics in query bias or scoring
- Basic tests, minimal README, PyPI package

**Out of scope:**
- Dreaming (entire cold path)
- Multi-tier indices (HOT/WARM/COLD/DEEP)
- `idx_priority` persistence + P7 tier migration
- Rich relation types (supersedes, part_of, etc.)
- Lazy extraction
- Proper event segmentation
- Tombstones
- memory.md generation

**Success criterion:** This code runs and returns the first message as 
the top hit. The test exercises the multilingual stress path (non-ASCII 
FTS5, multilingual embedder); English and other languages pass a 
fortiori. CI includes at least one additional non-Latin-script language.

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

### Stage 2 — Multi-Tier Architecture ✅

The formula already works from Stage 1; Stage 2 built the index 
architecture around it.

- Multi-tier indices (HOT/WARM/COLD/DEEP) — per-tier index structures
- `idx_priority` persistence + P7 (Rebalance) migration between tiers
- Cascade retrieval with early stopping and confidence thresholds
- Richer query bias analysis (multilingual: non-English proper-noun 
  detection, CJK punctuation patterns)
- Async embedding path (so cloud embedders don't block the Hot Path 
  budget)

### Stage 3 — Encoding Completeness ✅

- Proper event segmentation (rule-based)
- Multi-dimensional salience scoring
- Lazy extraction, level-1 heuristics only (dateparser for time;
  gist via first-sentence split). Entities / location / participants
  are never auto-populated in any level — NER is intentionally not
  wired into Mnemoss. See §9.7 for the rationale.
- Complete Working Memory

### Stage 4 — Light Dreaming ✅

- Phases P1–P5
- Session-end, idle, task-completion triggers
- LLM client abstraction
- Memory.md generation
- Dream Diary
- Cross-agent memory promotion: when Consolidate creates a summary from 
  a cluster that spans multiple agents, emit it with `agent_id=None` 
  (workspace-shared)

### Stage 5 — Deep Dreaming + Disposal ✅

- Phases P6–P8
- Nightly trigger
- Tombstone system
- DEEP index + reminiscence
- Surprise and cognitive-load triggers

### Stage 6 — Integration ✅ (shipped surfaces; post-ship work continues)

Shipped:
- REST server (`mnemoss-server`) + Python SDK (`mnemoss.sdk`)
- TypeScript SDK (`@mnemoss/sdk`)
- MCP server (`mnemoss-mcp`)
- Hermes `MemoryProvider` adapter (`adapters/hermes-agent/`)
- OpenClaw plugin (`adapters/openclaw/`, TypeScript)

Post-ship (tracked separately):
- Whitepaper
- Benchmarks vs mem0 / Zep / Letta
- Adapter polish as host frameworks evolve

### 9.7  NER is intentionally not implemented

NER has been removed from Mnemoss end-to-end and is **not on the
roadmap**. No automatic entity extraction runs anywhere — not at
encode time, not in Dream P3, not at query time. `b_F(q)` is
structural-only (quotes / URL / digits / acronyms); Dream P3 refines
only `gist` and `time`. The `extracted_entities`, `extracted_location`,
and `extracted_participants` fields stay in the data model but are
never populated by the library — they exist as a generic surface a
caller can write to manually if they already have entity information.

**Why no NER:**

- Any cheap extractor (Title-Case regex, gazetteer) is structurally
  language-biased — silent on CJK, Arabic, Thai — and Mnemoss is
  multilingual-first.
- LLM-driven NER in Dream P3 is expensive per cluster for a feature
  whose end value (marginal recall lift over the embedding +
  trigram-FTS hybrid) is unproven.
- Query-side NER contradicts the "memory system is query-agnostic"
  principle.
- Structural `b_F` cues (quotes, digits, URLs, code identifiers,
  acronyms) already handle the literal-query case.

**If you want NER in your workspace, DIY surfaces:**

- `Memory.extracted_entities` — set directly on memories you write
  via `observe()` (extend with a wrapper, or post-process after).
- `memory_type=entity` — the enum still exists; nothing prevents you
  from writing entity-typed Memory rows yourself.
- `relation` table — write custom predicates (`shares_entity`,
  `mentions`, whatever) with `store.write_relation()`; spreading
  activation is predicate-agnostic and will pick them up.
- `memory_fts` is single-column (`content`) in v8 — if you want
  BM25 over a secondary entity column, fork `store/schema.py` and
  bump your own local schema version.

A prior draft of this section proposed a full NER feature (phases
α–η: canonical cross-run identity, entity Memory-row promotion,
typed kinds, scoped recall API, disambiguation). It was removed
deliberately. If we ever revisit the decision, start from the git
history before commit `e2f8654` was reverted — the design thinking
is preserved there.

---

## 10. Technical Decisions & Dependencies

### 10.1 Language & Runtime

- **Python 3.10+** (modern type hints, match statements)
- Async-first API (`async def` everywhere public)
- Type-checked with mypy

### 10.2 Core Dependencies

The canonical dependency list lives in **`pyproject.toml`**. Do not mirror 
it here — it drifts. Each stage adds only what its scope requires.

**Deliberately avoided** (not in `pyproject.toml`, and won't be added):
- LangChain — dependency hell, over-abstraction
- pgvector — too heavy for local-first default
- FAISS — hard to install across platforms
- LlamaIndex — conflates retrieval with RAG

### 10.3 Storage: SQLite First

SQLite is the default backend because:
- Zero deployment (single file)
- Well-supported extensions (FTS5, sqlite-vec)
- Multi-workspace isolation = multiple .sqlite files
- Good enough for >1M memories per workspace

Storage layout per workspace:
~/.mnemoss/workspaces/{workspace_id}/
├── memory.sqlite           # Memory + Relations + Tombstones (+ sqlite-vec vectors, FTS5 index)
├── raw_log.sqlite          # Append-only message log
├── memory.md               # Generated view
├── memory_overrides.md     # User-edited
└── dreams/
├── 2026-04-21.md
└── ...

Vectors live in a `vec0` virtual table inside `memory.sqlite` (via 
`sqlite-vec`), not in a separate HNSW file. FTS5 trigram index lives 
in the same file.

### 10.4 Performance Targets

- `observe()`: < 50ms (median), < 100ms (p99)
- `recall()`: < 100ms for HOT-only hits, < 500ms for full cascade
- Dreaming: happens in background, never blocks user-facing operations
- Storage: ~2KB per memory average, ~2GB per year for heavy use

---

## 11. What Mnemoss Is NOT

Defining boundaries is as important as defining scope:

- **Not a vector DB.** We use one, but Mnemoss ≠ vector DB.
- **Not a RAG framework.** Mnemoss is memory; RAG is a usage pattern.
- **Not an LLM wrapper.** LLM is used inside Dreaming, not the user interface.
- **Not a conversation history store.** Raw Log is an implementation detail; 
  Memory is not a log.
- **Not tied to any agent framework.** Framework plugins are optional 
  integration layers.
- **Not a cloud service (yet).** Local-first is a core value.
- **Not magic.** Every behavior is formula-derived and traceable.

---

## 12. Naming & Terminology Glossary

Consistent terminology across code, docs, and conversations:

| Term | Meaning |
|---|---|
| Workspace | A memory namespace equivalent to one OpenClaw gateway. One SQLite DB. Holds memories from one or more agents plus workspace-shared (ambient) memories. |
| Agent | A persistent actor within a workspace. Each Memory/RawMessage/Event carries `agent_id: str \| None`; null means workspace-shared. |
| Ambient memory | A Memory with `agent_id=None`, visible to every agent in the workspace. |
| Memory | A single unit in the Memory Store (any type) |
| RawMessage | A single message in the Raw Log |
| Event | A group of RawMessages forming one episode |
| Observation | The act of calling `observe()` — doesn't guarantee Memory creation |
| Encoding | The process of turning an Event into a Memory |
| Recall | Retrieving Memories relevant to a query |
| Activation (A_i) | The formula output for a given (memory, query, time) |
| idx_priority | Derived value governing index tier |
| Tier | One of HOT/WARM/COLD/DEEP |
| Cascade | Multi-tier sequential retrieval with early stopping |
| Reconsolidation | Metadata updates after retrieval |
| Dreaming | The offline integration pipeline |
| Phase (P1-P8) | Individual steps in Dreaming |
| Trigger | An event that invokes Dreaming (idle, session-end, etc.) |
| Reminiscence | Cue-driven recall from DEEP tier |
| Tombstone | Record of a disposed memory |
| Fact | A Memory with `memory_type="fact"`, higher abstraction |
| Pattern | A Memory with `memory_type="pattern"`, highest abstraction |
| Entity | A Memory with `memory_type="entity"`, a concept anchor |
| memory.md | Auto-generated Markdown view of high-priority memories |
| memory_overrides.md | User-edited additions to memory.md |

---

## 13. Coding Conventions

- **Type hints everywhere.** Public APIs must be fully typed.
- **Async by default** for anything touching I/O.
- **Dataclasses for types**, Pydantic for config.
- **Small modules.** Each file has one clear responsibility.
- **Pure functions in `formula/`.** No side effects; easy to test.
- **No LLM calls in Hot Path.** Enforced by code review.
- **Errors fail loudly in dev, gracefully in prod.** Log, don't crash.
- **Tests for every formula component.** The formula is the heart; it must be covered.
- **Docstrings in NumPy style** for public APIs.
- **One responsibility per commit.** Easy review, easy revert.

---

## 14. Open Questions (To Resolve As We Build)

These are known unknowns. Document answers here as they're settled.

1. **Formula parameter calibration**: What are good defaults for d, τ, MP, α, β, γ 
   given actual Mnemoss usage patterns (vs. the original ACT-R experiments)?
   → *Still open — calibration against real workloads (see benchmarks 
   work) is the remaining task. Current defaults come from Anderson & 
   Schooler's original ACT-R values.*

2. **Event segmentation thresholds**: What time_gap, topic_shift_distance, 
   token_budget values produce natural-feeling events?
   → *Resolved in Stage 3. Defaults live in `FormulaParams` / 
   `EncoderParams`; tunable per workspace.*

3. **Relation graph construction**: How are relations first populated before 
   any dreaming runs?
   → *Resolved. Co-occurrence edges (`co_occurs_in_session`) written at 
   encode-time (Stage 1); richer predicates (`similar_to`, `derived_from`, 
   `supersedes`, `part_of`) populated by Dreaming P5 (Stage 4).*

4. **LLM cost envelope**: What's a reasonable daily LLM budget for dreaming 
   in typical use? How to enforce it?
   → *Partially resolved. Stage 4 shipped per-phase token budgeting + 
   retry caps in the LLM client abstraction. A full cost-governor 
   (across runs, across phases) remains open.*

5. **memory.md size ceiling**: What happens when pinned + auto-promoted 
   memories exceed comfortable system prompt size?
   → *Resolved in Stage 4. `export_markdown()` accepts 
   `min_idx_priority` and emits Facts → Entities → Patterns → Episodes 
   in priority order; callers truncate to fit their budget.*

6. **Concurrent access**: How should Mnemoss handle multiple processes 
   writing to the same workspace?
   → *Still open. Within-process writes serialize via `asyncio.Lock` + 
   SQLite WAL (single writer process per workspace). Cross-process 
   coordination is not implemented.*

---

## 15. References for Implementation

### 15.1 Must-Read Papers (core theory)

- Anderson, J. R., & Schooler, L. J. (1991). "Reflections of the environment 
  in memory." *Psychological Science*, 2(6), 396-408.
- Tulving, E. (1972). "Episodic and semantic memory." In *Organization of 
  Memory*.
- Tulving, E., & Thomson, D. M. (1973). "Encoding specificity and retrieval 
  processes in episodic memory." *Psychological Review*, 80(5), 352.
- Yonelinas, A. P. (2002). "The nature of recollection and familiarity." 
  *Journal of Memory and Language*, 46(3), 441-517.
- Zacks, J. M., et al. (2007). "Event perception: a mind-brain perspective." 
  *Psychological Bulletin*, 133(2), 273.

### 15.2 Useful Systems to Study (but not to imitate blindly)

- **mem0, Zep, Letta** — existing agent memory systems (know the competition)
- **OpenClaw, Hermes** — main integration targets
- **ACT-R 7 reference implementation** — for formula correctness
- **SuperMemo / Anki SM-2 algorithm** — practical spaced repetition

---

## 16. Project Metadata

- **Project name:** Mnemoss
- **License:** MIT
- **Homepage:** https://github.com/opcify/mnemoss
- **Repository:** https://github.com/opcify/mnemoss
- **PyPI package:** `mnemoss`
- **Current version:** 0.1.0 (Alpha — MVP feature-complete across Stages 1–6)
- **Maintainer:** Guangyang Qi ([@opcify](https://github.com/opcify))
- **Started:** 2026
- **Built on top of:** Opcify's internal memory needs + broader agent ecosystem

---

*End of MNEMOSS_PROJECT_KNOWLEDGE.md. Keep this updated as the project evolves.*