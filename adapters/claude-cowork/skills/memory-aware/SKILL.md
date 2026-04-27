---
description: Use Mnemoss as the persistent memory system for this conversation. Auto-invoke when the user asks Claude to remember, recall, forget, or check memory; or when the conversation references prior context that may live in Mnemoss.
---

# Mnemoss-aware memory

You have access to a Mnemoss memory system through the `mcp__mnemoss__*` MCP tools.
Mnemoss is an ACT-R-based memory backend: every recall is ranked by an
activation formula (recency + frequency + spreading + match) — the same
cognitive model human memory uses.

## When to use the tools

- The user asks you to remember a fact about them or their work →
  call `mcp__mnemoss__observe` with `role="user"` (or `"assistant"` for things
  you're committing to memory yourself).
- The user asks "what do you know about X?" / "did we discuss Y?" /
  "remind me of Z" → call `mcp__mnemoss__recall` with the query, then summarize
  the top hits in your reply.
- A surfaced memory looks promising but you want related context →
  call `mcp__mnemoss__expand` with that memory's id; it follows the relation
  graph one or more hops.
- The user says a memory is important and shouldn't be forgotten →
  call `mcp__mnemoss__pin` so it stays in the HOT tier and is exempted from
  disposal.
- The user wants to see what Mnemoss has on the workspace → call
  `mcp__mnemoss__status` and summarize `memory_count`, `tier_counts`, and
  `last_dream_at`.

## Agent scoping

Every observe and recall is implicitly scoped to the agent_id you set
in the `MNEMOSS_AGENT_ID` env var (defaults to none = workspace-wide).
The recall rule is `agent_id = X OR agent_id IS NULL`, so per-agent
private memories live alongside workspace-shared ambient ones.

## What not to do

- Don't call `mcp__mnemoss__dream`, `mcp__mnemoss__rebalance`, or `mcp__mnemoss__dispose`
  unprompted — those are operator tools that mutate the workspace; only
  run them when the user explicitly asks you to consolidate / clean up.
- Don't paraphrase what you recalled; cite the memory id and quote the
  content so the user can verify.
- Don't dump the whole `mcp__mnemoss__status` JSON; surface only the fields
  the user asked for.
