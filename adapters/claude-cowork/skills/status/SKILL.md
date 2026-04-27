---
description: Show the active Mnemoss workspace status — memory count, tier distribution, last dream, last observe.
disable-model-invocation: true
---

# /mnemoss:status

Call the `mcp__mnemoss__status` MCP tool and present a tight summary:

- **Workspace**: id + schema_version + embedder
- **Memories**: total `memory_count`, then `tier_counts` as a one-liner
  (`hot: N · warm: N · cold: N · deep: N`)
- **Activity**: `last_observe_at`, `last_dream_at`, `last_dream_trigger`
- **Cost** (if `llm_cost` is in the response): today / month / total
- **Recent dreams** (if `dreams.recent` is non-empty): each as
  `<timestamp>: <trigger> <duration_ms>ms <degraded?>`

Don't dump the full JSON — pick only the fields above. If the workspace is
empty (`memory_count == 0`), say so explicitly.
