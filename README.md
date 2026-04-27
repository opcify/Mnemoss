# Mnemoss

> ACT-R based memory system for AI agents.  
> Recall what matters, when it matters.

**Mnemoss** (pronounced _"nee-moss"_, the M is silent — from Greek 
_Mnemosyne_, goddess of memory) is a memory system for AI agents, 
designed from the first principles of human cognition.

Unlike typical "agent memory" systems that are vector databases with 
extra steps, Mnemoss implements the ACT-R cognitive architecture: it 
ranks memories by activation, considering recency, frequency, context, 
and relevance — not just semantic similarity.

## Status

**Alpha (0.1.0)** — MVP is complete. The core library, REST server, 
Python/TypeScript SDKs, MCP wrapper, scheduler, observability, and 
Hermes + OpenClaw adapters all ship in this repo. PyPI publish is 
pending. API may still change.

## Why Mnemoss

Most agent memory systems today treat memory as "store and retrieve by 
similarity." But human memory doesn't work that way:

- Recent events recall by their literal details; old events recall by 
  their gist.
- Frequently-used memories stay sharp; unused ones fade but aren't 
  deleted — the right cue can bring them back.
- Important information gets consolidated during idle moments, not just 
  at night.
- Abstractions (facts, patterns) emerge from repeated experience, not 
  from upfront categorization.

Mnemoss implements all of this as a single mathematical system, built on 
a half-century of cognitive science research.

## Core Ideas

- **One formula drives everything** — retrieval ranking, index migration, 
  disposal, reminiscence all emerge from the ACT-R activation equation.
- **One table holds all memory types** — episodes, facts, entities, 
  patterns all live in a unified Memory table.
- **Storage is cheap, retrieval is expensive** — Mnemoss optimizes the 
  latter through multi-tier indexing with latency gradient.
- **Dreaming is opportunistic** — consolidation happens during idle 
  moments, session endings, surprises, and cognitive-load spikes — not 
  just on a nightly cron.
- **Framework-agnostic** — integrate with OpenClaw, Hermes, Claude Code, 
  Cursor, or your own agent.

## Quick Start

```python
import asyncio
from mnemoss import Mnemoss

async def main():
    mem = Mnemoss(workspace="my_agent")
    
    await mem.observe(
        role="user", 
        content="Meeting with Alice tomorrow at 4:20 PM"
    )
    
    results = await mem.recall("What time is the meeting?", k=3)
    for r in results:
        print(r.memory.content)

asyncio.run(main())
```

## Integration Patterns

Mnemoss is a library, not a middleware — it returns ranked memories and 
lets your agent framework decide how to assemble the prompt. Three 
patterns cover most real-world agents.

### Pattern 1 — Static system-prompt injection

Load a filtered `memory.md` view into the system prompt once per 
session. This is the OpenClaw-compatible pattern.

```python
md = await mem.export_markdown(agent_id="alice", min_idx_priority=0.5)
# md is deterministic Markdown: Facts → Entities → Patterns → Episodes,
# pinned items marked 📌. Paste into your system prompt.
```

Use for: standing profile-style memory that should always be "in mind" — 
preferences, long-term facts, project context. Deterministic, no LLM.

### Pattern 2 — Per-turn retrieval

Call `recall()` each turn, conditioned on the user's message. The full 
ACT-R formula picks what's relevant _now_ — base-level activation 
(recency + frequency), spreading from working memory, dynamic literal-vs-
semantic matching, and noise.

```python
results = await mem.recall(user_message, k=5, agent_id="alice")
context = "\n".join(r.memory.content for r in results)
# Prepend `context` to the current turn's prompt.
```

For queries that are explicitly about something older or rarer, pass 
`include_deep=True` to force the cascade through the DEEP tier. 
Mnemoss already auto-scans DEEP when the query contains temporal cues 
like "long ago" or "last year", but any caller can opt in 
unconditionally.

**Auto-expansion on follow-up.** When a recall hits the same topic as 
a previous one — detected by result overlap or query-embedding cosine, 
with no time limit — Mnemoss automatically widens the result set by 
spreading activation through the relation graph from the previous 
hits. Expanded memories are marked `r.source == "expanded"` (direct 
hits are `"direct"`) so your agent can tell recollection from 
association. 

Hop count escalates as the user keeps probing the topic in rapid 
succession (1 hop → 2 → 3, capped by `expand_hops_max`); after a 
gap longer than `streak_reset_seconds` (default 10 min) the streak 
resets to 1 hop — a user returning to a thread much later still gets 
expansion, just shallower to start. Pass `auto_expand=False` to 
disable this.

Use for: episodic recall, specific facts relevant only to the current 
question. This is the primary path Mnemoss was designed around.

### Pattern 3 — Hybrid (recommended for production agents)

Combine both: a standing memory block in the system prompt plus per-turn 
retrieval of query-specific hits.

```
┌── system prompt ──────────────────────────────┐
│  <static instructions>                        │
│                                               │
│  ## Your standing memory (pinned + hot)       │
│  <mem.export_markdown(min_idx_priority=0.7)>  │
└───────────────────────────────────────────────┘
┌── each user turn ─────────────────────────────┐
│  ## Relevant memories                         │
│  <mem.recall(user_msg, k=5)>                  │
│                                               │
│  User: <actual message>                       │
└───────────────────────────────────────────────┘
```

After the LLM responds, record the turn so future recalls see it:

```python
await mem.observe(role="user", content=user_message)
await mem.observe(role="assistant", content=llm_response)
```

On idle or session end, trigger consolidation so dreaming can extract 
facts, build relations, and rebalance tiers:

```python
await mem.dream(trigger="session_end")
```

### How the patterns compare

| | Pattern 1 (export) | Pattern 2 (recall) | Pattern 3 (hybrid) |
|---|---|---|---|
| Load frequency | Once per session | Every turn | Both |
| Selection | idx_priority + pinned + salience | Full ACT-R activation | Each at its layer |
| Query-aware | ❌ | ✓ | ✓ |
| Typical cost | 5–15ms | 5–20ms | ~30ms/turn |
| Best for | Standing profile | Query-specific recall | Production agents |

The HOT/WARM/COLD/DEEP index tiers keep Pattern 1's output bounded even 
after months of use, and the cascade keeps Pattern 2 fast as the store 
grows — neither pattern needs you to manage size manually.

### Integration surfaces

The three patterns work identically over four access modes, so the host 
framework can pick whichever fits its runtime:

- **Python library** — `pip install mnemoss`, embed directly.
- **REST server** — `mnemoss-server`, any language can call `/recall` 
  and `/export`.
- **Python / TypeScript SDK** — typed clients over the REST API.
- **MCP server** — `mnemoss-mcp`, exposes every operation as an MCP 
  tool so Claude, Cursor, etc. can use Mnemoss as a plug-in.

### Framework adapters

Pre-built glue for specific agent frameworks, so you skip the Pattern 3 
wiring:

- **[`mnemoss-hermes`](./adapters/hermes-agent/)** — Python memory 
  provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent). 
  Subclasses Hermes's `MemoryProvider`, handles per-turn auto-injection 
  via `prefetch`, exposes `mnemoss_recall`/`mnemoss_expand`/`mnemoss_pin` 
  tools to the hosted model, and dreams at session end.
- **[`mnemoss-openclaw`](./adapters/openclaw/)** — TypeScript plugin for 
  [OpenClaw](https://github.com/openclaw/openclaw) gateways. Implements 
  OpenClaw's `MemorySearchManager` contract and registers as a unified 
  memory capability via `api.registerMemoryCapability`. Delegates to a 
  shared `mnemoss-server` via the `@mnemoss/sdk` TS client.
- **[`mnemoss-claude-cowork`](./adapters/claude-cowork/)** — Plugin for 
  [Claude Cowork](https://claude.com/blog/cowork-plugins) and 
  [Claude Code](https://code.claude.com). Bundles the existing 
  `mnemoss-mcp` MCP server (13 tools: observe, recall, expand, pin, 
  dream, status, etc.) plus four skills (`/mnemoss:recall`, 
  `/mnemoss:observe`, `/mnemoss:status`, and the auto-invoked 
  `mnemoss:memory-aware`). Same plugin format works in both Cowork and 
  Code.

## Design Documents

- [`MNEMOSS_PROJECT_KNOWLEDGE.md`](./MNEMOSS_PROJECT_KNOWLEDGE.md) — 
  Complete project knowledge base
- [`MNEMOSS_FORMULA_AND_ARCHITECTURE.md`](./MNEMOSS_FORMULA_AND_ARCHITECTURE.md) — 
  The mathematical formula and architecture

## Cognitive Science Roots

Mnemoss builds directly on:

- **ACT-R activation equation** (Anderson & Schooler, 1991)
- **Episodic/Semantic continuum** (Tulving; Moscovitch)
- **Dual-process theory** (Yonelinas, 2002)
- **Event segmentation theory** (Zacks, 2007)
- **Awake replay and opportunistic consolidation** (Foster & Wilson, 2006)

See [`MNEMOSS_PROJECT_KNOWLEDGE.md`](./MNEMOSS_PROJECT_KNOWLEDGE.md) 
Section 2 for the full list.

## License

MIT

## Provenance

Built by **Guangyang Qi** ([@opcify](https://github.com/opcify)) as part of 
the Opcify project. Open-sourced for the broader agent ecosystem.