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

These nine principles govern every decision. If a proposed change violates one, 
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

**Principle 9: `idx_priority` is for ranking, not search.**
`idx_priority` is the ACT-R activation formula compressed into a single
scalar — a memory's intrinsic *importance* / *lifecycle classification*.
It drives **tier membership** (HOT/WARM/COLD/DEEP), **disposal**,
**export filtering**, and **rebalance ordering**. It is **not** a
relevance signal for "does this memory answer the query I'm asking
right now?" — that's what cosine similarity is for. The two axes are
orthogonal: a memory can be high-priority (recently active, pinned)
but irrelevant to the current query, or low-priority (dormant, old)
but the only correct answer.

The default recall path (`use_tier_cascade_recall=True`) reads the
tier classification (which idx_priority drove at the last Rebalance)
to decide *which subset* to scan first, then ranks **purely by
cosine** within each tier. Mixing idx_priority into the per-candidate
score at recall time creates the failure modes documented in §17:
high-cosine-low-priority gold answers get filtered out by the activation
gate, and the system underperforms raw cosine baselines on aged corpora.

The cognitive analog: working memory's "this is currently primed" is
separate from "this answers the question I'm asking." Humans don't
filter retrieval by intrinsic importance — they retrieve by relevance,
and importance only governs *what stays primed* between retrievals.

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

### Post-MVP — Production readiness ✅

Landed after Stage 6, before anyone declared the MVP "done":

- **Structural refactor.** `sqlite_backend.py` split into a thin
  async façade plus `_memory_ops.py` / `_graph_ops.py` /
  `_raw_log_ops.py` / `_sql_helpers.py` for sync SQL, each
  unit-testable against a plain `apsw.Connection`.
- **Schema migration framework** (`store/migrations.py`). Registered
  chain applies on open; newer-DB-than-code raises. Answers §14 Q8.
- **Cross-process workspace lock** (`store/_workspace_lock.py`).
  `fcntl`/`msvcrt` advisory lock on `{workspace}/.mnemoss.lock` so a
  second opener fails fast with `WorkspaceLockError`. Answers §14 Q6.
- **Dream cost governor** (`dream/cost.py`). `CostLimits` ceilings on
  LLM calls per run / day / month, `CostLedger` persisted in
  `workspace_meta`. Counts surface via `status().llm_cost`. Answers §14 Q4.
- **Partial-failure recovery** in the Dream runner. Per-phase
  try/except; `DreamReport.degraded_mode` flag; phase errors don't
  kill the run. Answers §14 Q9.
- **Retry wrapper for flaky embedders** (`encoder/retrying.py`).
  Opt-in `RetryingEmbedder` with bounded exponential backoff. Answers §14 Q7.
- **Param validators**. `FormulaParams`, `EncoderParams`,
  `SegmentationParams`, `MnemossConfig`, `CostLimits` all reject
  invalid values at construction with clear error messages.
- **Input hardening at REST/MCP**. Configurable caps on observe
  content size, recall k, and metadata size (`ServerConfig.*`).
- **Auto-split for long observes** (`encoder/chunking.py`). Opt-in
  `EncoderParams.max_memory_chars` produces N Memory rows for one
  Raw Log row when content exceeds the cap. Answers §14 Q10.
- **Operator CLI**: `mnemoss-inspect <workspace>` prints a snapshot
  of `status()` + tombstones.
- **Benchmark harnesses**: `bench/bench_recall.py` (latency sweep),
  `bench/calibrate.py` (FormulaParams sweep vs labeled corpus).
- **Code quality**: mypy strict clean across 78 source files; 667+
  tests passing at ~93% coverage.

Post-ship backlog (tracked separately):
- Whitepaper
- Comparative benchmarks vs mem0 / Zep / Letta
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
   across runs landed in the production-readiness pass — see 
   `dream/cost.py` (`CostLimits` + `CostLedger`). Per-run / per-day 
   / per-month ceilings are configurable on `Mnemoss(cost_limits=…)`; 
   the persisted ledger survives restarts and is surfaced through 
   `status().llm_cost`.*

5. **memory.md size ceiling**: What happens when pinned + auto-promoted 
   memories exceed comfortable system prompt size?
   → *Resolved in Stage 4. `export_markdown()` accepts 
   `min_idx_priority` and emits Facts → Entities → Patterns → Episodes 
   in priority order; callers truncate to fit their budget.*

6. **Concurrent access**: How should Mnemoss handle multiple processes 
   writing to the same workspace?
   → *Answered in the production-readiness pass. Within-process writes 
   still serialize via `asyncio.Lock` + SQLite WAL. Cross-process 
   coordination is a stdlib advisory lock on `{workspace_dir}/.mnemoss.lock` 
   (`fcntl.flock` on unix, `msvcrt.locking` on windows) acquired during 
   `SQLiteBackend.open()`. A second process attempting to open the same 
   workspace raises `WorkspaceLockError` fast; the lock releases on 
   `close()` or process exit. See `store/_workspace_lock.py`.*

7. **Flaky embedder providers**: OpenAI / Gemini embedding endpoints 
   return transient 429 / 5xx / timeout errors; `observe()` shouldn't 
   die on them.
   → *Resolved. `RetryingEmbedder` wraps any `Embedder` with bounded 
   exponential-backoff + jitter retries on transient exception classes 
   (`ConnectionError`, `TimeoutError`, `OSError`, plus caller-provided 
   `retry_on=` for provider-specific errors like 
   `openai.RateLimitError`). Opt-in composition; `dim` / `embedder_id` 
   pass through transparently so the schema pin matches.*

8. **Schema evolution without workspace rebuild**: Users can't afford to 
   rebuild their workspace every time a `SCHEMA_VERSION` bump ships.
   → *Resolved. `store/migrations.py` owns a registered chain; on 
   open, any DB older than code runs the chain in one transaction. 
   Newer-than-code raises. Adding a migration = bump `SCHEMA_VERSION`, 
   append a `Migration(from_version=N, to_version=N+1, …)` with a 
   closure that takes an `apsw.Connection`, cover in 
   `tests/test_migrations.py`.*

9. **Dream partial failures**: If one phase crashes, should the whole 
   run abort or should downstream phases continue on partial state?
   → *Resolved. `DreamRunner` wraps every phase in try/except; an 
   exception becomes `PhaseOutcome(status="error", error=…)` and 
   downstream phases still attempt on whatever state survived. 
   `DreamReport.degraded_mode` / `errors()` expose the failure surface. 
   Integration test: `tests/test_dream_failure_recovery.py`.*

10. **Long-content observes**: What happens when a caller passes a 
    super-long message to `observe()` — a 32KB tool output, a 
    multi-paragraph agent memo, a minified JSON blob?
    → *Resolved. Opt-in auto-split via `EncoderParams.max_memory_chars` 
    (default `None`, backward compatible). When set and the encoded 
    content exceeds the cap, the encoder produces **N Memory rows** 
    for one `observe()` — the Raw Log still writes exactly one 
    `raw_message` row, preserving Principle 3. Split boundaries, in 
    order of preference:*
    
    1. *paragraph (`\n\n`)*
    2. *line (`\n`)*
    3. *sentence — CJK-aware, matches `.!?…。！？` followed by whitespace*
    4. *hard char cut (last-resort fallback for e.g. minified JSON with 
       no whitespace or terminators)*
    
    *Each chunk carries `source_context.split_part = {"index": i, 
    "total": n, "group_id": <first_chunk_id>}`. Chunks share their 
    `source_message_ids` so any chunk can trace back to the original 
    RawMessage. Recommended caps: `2000` for LocalEmbedder (MiniLM 
    tokenizer truncates past ~512 tokens), `30000` for OpenAI's 
    text-embedding-3-small. Without a cap, embedders silently drop 
    tokens past their context window and semantic recall on the 
    dropped tail is broken without any warning. See 
    `src/mnemoss/encoder/chunking.py` and 
    `tests/test_observe_chunking.py`.*
    
    *Note: this is distinct from `SegmentationParams.max_event_characters` 
    (default 8000), which caps how many sibling messages accumulate 
    in a per-`(agent, session, turn)` buffer before the segmenter 
    closes it. That rule fires on the buffer's cumulative length, 
    not on one message's length. Both limits coexist: a segmenter 
    buffer closes at 8000 cumulative chars across N messages; if the 
    resulting Memory still exceeds `max_memory_chars`, the encoder 
    then splits it into chunks.*

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
- **Current version:** 0.0.1 (First formal alpha — capacity-based tiers + tier-cascade-pure-cosine recall + supersede-on-observe by default; beats raw_stack on recall and latency on realistic aged corpora)
- **Maintainer:** Guangyang Qi ([@opcify](https://github.com/opcify))
- **Started:** 2026
- **Built on top of:** Opcify's internal memory needs + broader agent ecosystem

---

## 17. Q&A — Implementation Verification

Concrete answers to common questions about how the documented architecture
maps to actual code. Each answer is sourced from a code audit; line numbers
are stable as of the post-MVP production-readiness pass. Treat them as a
shortcut, not a source of truth — the code itself wins if a citation rots.

### Architecture update — April 2026

Five default changes shipped together based on empirical findings from
`bench/bench_tier_lifecycle.py`, `bench/bench_rebalance_lift.py`,
`bench/bench_tier_oracle.py`, and `bench/bench_multi_step.py`:

1. **Capacity-based tier bucketing** (replaces threshold-based).
   `Rebalance` ranks memories by ``idx_priority`` and fills HOT
   (200 cap), WARM (2000), COLD (20000), DEEP (rest) top-down.
   The original `idx_priority > 0.7 → HOT` threshold rule
   degenerated under any aged corpus — 99%+ of memories collapsed
   into DEEP and the cascade became useless. Capacity caps are
   structurally bounded and self-calibrate to any corpus age.
   Cognitively grounded: working memory is hard-capped (Miller
   1956, Cowan 2001), not threshold-gated.

2. **Tier-cascade-pure-cosine** is the new default recall path
   (`use_tier_cascade_recall=True`). Recall reads tier classifications
   that Dream/Rebalance computed off the read path; ranking within a
   tier is pure cosine. No per-candidate ``B_i``, no spreading
   activation, no matching ``idx_priority`` gate, no ``τ`` floor at
   recall. The legacy ACT-R recall remains opt-in via
   ``use_tier_cascade_recall=False``.

3. **Cascade short-circuit disabled** (`cascade_min_cosine=0.99`).
   Real-world cosines rarely reach 0.99, so the cascade exhausts every
   populated tier on each query. The earlier 0.5 default caused a 4.7pp
   recall regression because realistic Rebalance can't reliably put
   every gold answer in HOT. With short-circuit off, recall matches
   raw_stack while latency stays 2× faster.

4. **Reconsolidation gated on cosine** (`reconsolidate_min_cosine=0.7`).
   Only memories whose query-time cosine clears the threshold get
   ``access_history`` bumps. Reduces "popular distractor" promotion
   in Rebalance — at threshold 0.7 the test-phase recall lifts from
   0.3737 (ungated) to 0.3882 (+1.45pp), at +7ms p50 cost.

5. **`supersede_on_observe=True`** is the new default (was opt-in).
   At the 0.85 cosine threshold, the mechanism filters near-duplicate
   memories from recall. Empirically: combined with the four other
   changes above, this is the first config under which mnemoss
   *cleanly beats raw_stack* on a realistic aged corpus —
   recall@10 = 0.4622 vs raw_stack 0.4205 (+4.17pp) on N=20K MiniLM.

Empirical headline (N=20K LoCoMo + 600 chain memories, MiniLM,
all five defaults shipped, `bench_multi_step` with rebalance after
each phase):

| Configuration | recall@10 | p50 latency |
|---|---:|---:|
| **mnemoss (all five defaults)** | **0.4622** | ~25 ms |
| raw_stack baseline | 0.4205 | 54 ms |
| Oracle ceiling (gold in HOT) | 0.7122 | 18 ms |

mnemoss now beats raw_stack on **both axes** simultaneously:
**+4.17pp recall, 2× faster latency**. The 25pp gap to the oracle
ceiling is the headroom available with better classification signal
(selective reconsolidation API, query-aware classification — both
on the post-MVP roadmap).

After Rebalance, mnemoss is **3.3× faster than raw_stack at recall**
because the cascade short-circuits at HOT (200 memories) for most
queries. The 4.7pp recall trade comes from "popular distractors"
getting promoted into HOT alongside genuine gold answers — a
reconsolidation-discipline issue, not an architectural one.

What this trades away: fresh-ingest supersession (the
``bench_multi_step`` win) **collapses to raw_stack-level** under
pure-cosine recall, and **Rebalance does not recover it**. We
verified this with a follow-up run of `bench_multi_step` that
explicitly calls `mem.rebalance()` between observing all chain
versions and scoring the queries. Both axes of the bench (latest@1,
older@1) match the no-rebalance numbers exactly across all five
arms. Two compounding reasons:

1. **Tier differentiation is too coarse for fresh chains.** With
   `d_storage=0.5` and 60s observe gaps, the latest version's
   ``idx_priority`` is only ~0.045 above older versions. With
   capacity caps 200/2000/20000 and ~900 total memories in the
   bench, all chain versions land in the same tier most of the
   time — Rebalance can't separate them.

2. **Within-tier ranking is pure cosine.** Even when the latest
   version is the highest-priority entry in HOT, generic queries
   like "where do you work now?" produce similar cosines for all
   chain versions (and for some unrelated distractors). The ACT-R
   recall path's per-candidate ``B_i`` term acted as a recency
   tiebreak; tier-cascade-pure-cosine doesn't have it.

So the win actually came from the per-candidate `B_i` math at
recall, not from the tier classification. Tier cascade at recall
is the right read-side architecture for warm-cache historical
recall, but not for fresh-ingest within-session supersession.

Supersession instead has to be handled by orthogonal mechanisms
that don't depend on read-time activation math:

- ``supersede_on_observe`` — immediate within-session, fires when
  cosine to an existing memory exceeds the threshold (default 0.85).
  Filters the old fact's `superseded_by` at SQL level so recall
  never sees it. Works regardless of Dream cadence or recall path.
- Disposal — old facts below the activation floor get tombstoned
  by Dream. Slow, depends on Dream cadence; bulk hygiene rather
  than session-level supersession.
- Legacy ACT-R recall path — set `use_tier_cascade_recall=False`
  to opt back into per-candidate `B_i` scoring. Trades the 3.3×
  latency win for the supersession win. Right call for workloads
  where within-session contradictions matter more than warm-cache
  speed.

Where the trade points the architecture: optimize for **the
warm-cache regime** (long-running agent with regular Dream cadence
and accumulated history). The fresh-ingest within-session
supersession case is a known weakness of the read path; either run
Dream more aggressively, rely on `supersede_on_observe`, or accept
that the latest version of a chain may not always rank first until
the next Rebalance.

### Q1. How is `idx_priority` created and updated through a memory's lifecycle?

**Created** during ingest in `src/mnemoss/encoder/event_encoder.py:45-46`. The
initial value is `sigmoid(η_0)` ≈ 0.731 (with default `η_0 = 1.0`), computed
by `initial_idx_priority()` at `src/mnemoss/formula/idx_priority.py:65-73`.
The schema column has a fallback default of 0.5
(`src/mnemoss/store/schema.py:50`) but the observe-path always overrides it.

**Read by the recall path** depends on which mode is active:

- **Tier-cascade-pure-cosine** (default, `use_tier_cascade_recall=True`):
  `idx_priority` is **not** read at recall. Tier membership (computed
  from idx_priority by Rebalance) drives cascade scan order; ranking
  within a tier is pure cosine.
- **Fast-index** (`use_fast_index_recall=True`): the cached value is
  combined with cosine — `score = sem_w·cos + pri_w·idx_priority`.
- **Legacy ACT-R** (both flags False): live-recomputed at recall via
  `compute_idx_priority(B_i, ...)` and used to gate matching weights.

Other readers (all modes): export filter
(`src/mnemoss/export/markdown.py:59`), store queries
(`src/mnemoss/store/_memory_ops.py:280-281`).

**Updated** only by the **Rebalance** dream phase
(`src/mnemoss/index/rebalance.py:103-160`) — now a two-stage rank-
and-bucket pass:

1. Recompute each memory's value from `B_i + α·salience + β·emotional + γ·pinned`
   (using `d_storage` for aggressive decay).
2. Sort all memories by `idx_priority` desc; pinned go to HOT first;
   fill HOT/WARM/COLD top-down by capacity caps from
   `TierCapacityParams`. Whatever doesn't fit goes to DEEP.

Triggered by nightly dream, manual `mem.rebalance()`, or
`dream(trigger="nightly")`. **Not updated on every recall** — it's
a cached snapshot until the next rebalance. Reminiscence
(DEEP→WARM on hit) is the only mid-recall write, and it bumps the
single hit memory to a soft-WARM-cap state without re-bucketing
the whole index.

**One exception**: when recall hits a DEEP memory, that single memory gets
bumped to WARM in-place (`src/mnemoss/recall/engine.py:291-298`) — the
"reminiscence" path — without waiting for rebalance.

### Q2. How is the HOT / WARM / COLD / DEEP memory amount managed?

**Capacity-bounded, ranked at Rebalance.** Defaults
(`src/mnemoss/core/config.py:TierCapacityParams`):

- HOT — top 200 by `idx_priority` (fixed cap; cognitively grounded
  in working-memory size)
- WARM — next 2,000 (easily-accessible long-term analogue)
- COLD — next 20,000 (recallable-with-effort analogue)
- DEEP — everything else (dormant long-term)

At Rebalance (`src/mnemoss/index/rebalance.py:_bucket_by_capacity`),
memories are sorted by `idx_priority` descending. Pinned memories take
the top of HOT regardless of rank (pin = "force into working memory"
with capacity displacement). The remaining seats fill top-down across
HOT, WARM, COLD; the residual goes to DEEP.

**Why capacity, not threshold.** The earlier
`idx_priority_to_tier()` function (still in code at
`formula/idx_priority.py:46-62`, used for *initial* tier of a fresh
observe — see `event_encoder.py:61, 112`) maps a single value to a
tier by fixed thresholds. This rule degenerates under any aged
corpus: `B_i = ln(Σ (t-t_k)^-d)` collapses for memories older than ~1
hour with default parameters; >99% of memories fall to
`idx_priority < 0.1` and the cascade has nothing to short-circuit at.
Capacity-based ranking is structurally bounded — HOT, WARM, COLD
stay constant-size regardless of formula tuning or corpus age.

**Recall** still cascades HOT → WARM → COLD (and DEEP if asked) but
the new default uses pure cosine within each tier, no per-candidate
activation math.

**Cascade early-stop is effectively disabled by default**
(`cascade_min_cosine = 0.99`). Real-world cosines rarely reach 0.99,
so the cascade exhausts every populated tier on each query. Empirical
finding from the rebalance-lift bench at N=20K MiniLM:

| `cascade_min_cosine` | recall@10 | p50 |
|---:|---:|---:|
| 0.5 (short-circuit at HOT) | 0.3737 | 16 ms |
| **0.99** (no short-circuit, shipped) | **0.4205** | 26 ms |
| raw_stack baseline | 0.4205 | 54 ms |

Short-circuit at 0.5 caused a 4.7pp recall regression vs raw_stack
because realistic Rebalance can't reliably put every gold answer in
HOT. Disabling short-circuit recovers full recall while keeping the
2× latency win — per-tier ANN with `tier_filter` is still cheaper
than a flat scan, even when scanning every populated tier.

**This default may be revisited.** When the Rebalance signal
improves (selective reconsolidation API, query-aware classification,
or higher-quality embedders), HOT becomes a high-precision pre-filter
and short-circuit at e.g. 0.6-0.8 could buy back latency without a
recall cost. The tier-oracle bench is the canonical measurement
gate: if its gap between realistic Rebalance and the oracle ceiling
(0.7122 on MiniLM, 0.7683 on Nomic) closes meaningfully, lower this
default. Until then, no early-stop.

**Migration runs only at Rebalance.** No live migration during
observe or recall, except the DEEP → WARM reminiscence bump on
recall hit. Soft cap: reminiscence may temporarily push WARM above
`warm_cap` until the next Rebalance enforces the limits.

### Q3. Does `export_markdown` scope by `agent_id` or by workspace?

**Both.** It honors whichever scope the caller binds.

The public API (`src/mnemoss/client.py:468-488`) accepts an `agent_id`
keyword argument. `AgentHandle.export_markdown()`
(`src/mnemoss/client.py:764-765`) automatically forwards the bound
`agent_id` from `mem.for_agent(id)`.

The store filter (`src/mnemoss/store/_memory_ops.py:270-294`) implements
the standard scope rule:

- `agent_id=None` → `WHERE agent_id IS NULL` (workspace-ambient memories only)
- `agent_id="A"` → `WHERE (agent_id = 'A' OR agent_id IS NULL)` (A's private + ambient)

This matches the recall scope rule from §5.3 ("workspace = gateway,
`agent_id` is a private filter"). Pinned IDs are scoped consistently
(`src/mnemoss/store/_graph_ops.py:166-178`).

### Q4. How is dreaming triggered?

The **5 triggers** are defined as a string enum in
`src/mnemoss/dream/types.py:12-24` (`IDLE`, `SESSION_END`, `SURPRISE`,
`COGNITIVE_LOAD`, `NIGHTLY`) and each has a phase-list mapping in
`src/mnemoss/dream/runner.py:50-82`. The light triggers (`idle`,
`session_end`) run only the encode-side phases; `surprise` /
`cognitive_load` skip replay/cluster and go straight to consolidate;
`nightly` runs everything plus rebalance and dispose.

**Auto-fired by the scheduler:**

- **NIGHTLY** — fires daily at `nightly_at` (default 03:00 UTC) — `src/mnemoss/scheduler/scheduler.py:118-121, 148-153`
- **IDLE** — fires when `now − last_observe > idle_after_seconds` (default 600s) AND a new observe has occurred since the last idle fire — `scheduler.py:123-126, 155-166`

**Caller-driven only — no auto-fire path in the codebase:**

- **SESSION_END** — caller must `await mem.dream(trigger="session_end")` after a session ends
- **SURPRISE** — no threshold or auto-trigger defined anywhere
- **COGNITIVE_LOAD** — no counter or auto-trigger defined anywhere

This is by design — see the explicit comment in
`src/mnemoss/scheduler/__init__.py:10-11`: *"The three remaining triggers
(session_end, surprise, cognitive_load) stay caller-driven because they
reflect semantic [signals only the caller knows]."*

**The scheduler is opt-in.** It is NOT always running. Users explicitly
call `await scheduler.start()` after constructing `Mnemoss(...)`
(`src/mnemoss/scheduler/scheduler.py:72-85`). It can also be disabled
entirely with `SchedulerConfig(enabled=False)`.

### Gaps flagged by the audit

These are doc/code disagreements found during the verification, kept here
so a future reader can see the work is not done:

1. **`SURPRISE` and `COGNITIVE_LOAD` have no auto-fire signal.** They
   exist as enum values and phase mappings, but no code path watches for
   "high surprise activation" or "high cognitive load" and fires the
   dream. §6.3 lists them as triggers alongside the working two — that
   reads as if all five fire automatically. Either the scheduler grows
   detection logic, or the doc is reworded to say "caller-driven".
2. ~~**No tier capacity caps.**~~ **Resolved (April 2026).**
   `TierCapacityParams` was added; Rebalance now buckets memories by
   capacity rank instead of by threshold. See the architecture-update
   block above.
3. ~~**Reconsolidation discipline.**~~ **Partially shipped (April 2026).**
   `FormulaParams.reconsolidate_min_cosine = 0.7` (default) gates the
   reconsolidation bump on cosine similarity. Empirical sweep on the
   rebalance-lift bench (N=20K LoCoMo, MiniLM):

   | Threshold | Test recall@10 | p50 |
   |---:|---:|---:|
   | ungated (-1.0) | 0.3737 | 16 ms |
   | 0.5 | 0.3737 | 16 ms |
   | **0.7** (shipped) | **0.3882** | 23 ms |
   | 0.8 | 0.3975 | 23 ms |

   The gate provides a small but real lift (+1.45pp at 0.7, +2.38pp
   at 0.8). It does not close most of the 33.85pp gap to the oracle
   ceiling (0.7122) — that gap is structural, not noise-driven, and
   needs a stronger signal than cosine for which retrieved memories
   actually answered the query. A selective-reconsolidation API
   (`mem.reinforce([m1, m3])` after the agent acted) remains on the
   roadmap.
4. **Fresh-ingest supersession does not recover from Rebalance.**
   Under tier-cascade-pure-cosine, the per-candidate `B_i` recency
   tiebreak that delivered fresh-ingest supersession is gone from
   the read path. Empirically (`bench_multi_step` with explicit
   `mem.rebalance()` between observe and score), Rebalance does
   not bring it back: tier differentiation is too coarse and
   within-tier ranking is pure cosine. The fix isn't tier-side —
   it's the orthogonal mechanisms. `supersede_on_observe` filters
   stale facts at SQL level for within-session contradictions;
   disposal removes them in bulk during Dream. For workloads
   where the chain-version supersession win matters more than the
   warm-cache speed, opt back into the legacy ACT-R recall path
   via `use_tier_cascade_recall=False`.

---

*End of MNEMOSS_PROJECT_KNOWLEDGE.md. Keep this updated as the project evolves.*