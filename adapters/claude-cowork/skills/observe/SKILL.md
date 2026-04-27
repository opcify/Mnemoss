---
description: Append a memory to the Mnemoss workspace via the observe MCP tool.
disable-model-invocation: true
---

# /mnemoss:observe

Append a new memory to the active Mnemoss workspace.

Content to observe: **$ARGUMENTS**

Call `mcp__mnemoss__observe` with:
- `role="user"`
- `content="$ARGUMENTS"`
- `metadata={}` (or augment if the user supplied structured fields)

After the call, report back the `memory_id` Mnemoss assigned. That id is
stable — the user can pin it later by asking Claude to call the
`mcp__mnemoss__pin` tool with that id.
