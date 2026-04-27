---
description: Run an ACT-R-ranked recall against the Mnemoss workspace and surface the top matches.
disable-model-invocation: true
---

# /mnemoss:recall

Use the `mcp__mnemoss__recall` MCP tool to search the active workspace for memories
matching this query: **$ARGUMENTS**

Steps:
1. Call `mcp__mnemoss__recall` with `query="$ARGUMENTS"`, `k=5`.
2. For each hit, show: `id`, `score` (4 decimal places), and the first 120
   characters of `content`.
3. If `source` is `expanded` for any hit, mark it with `(via relation graph)`.
4. If there are zero hits, say so plainly — don't fabricate.

Don't summarize across hits; preserve each one verbatim. The user is the
judge of relevance.
