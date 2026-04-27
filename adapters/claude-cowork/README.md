# mnemoss — Claude Cowork plugin

[Mnemoss](https://github.com/opcify/mnemoss) as a [Claude Cowork](https://claude.com/blog/cowork-plugins) /
[Claude Code](https://code.claude.com) plugin. Same plugin format works for both;
Anthropic ships the spec across them.

Wraps the bundled `mnemoss-mcp` MCP server so any Cowork or Claude Code agent
can recall, observe, and consolidate memories backed by Mnemoss's ACT-R
activation engine. Adds three slash commands and one model-invoked skill on
top of the raw MCP tools.

## What you get

| Component | Type | Purpose |
|---|---|---|
| `mnemoss` MCP server | bundled via `.mcp.json` | 13 tools: `observe`, `recall`, `expand`, `pin`, `explain_recall`, `dream`, `rebalance`, `dispose`, `tombstones`, `tier_counts`, `export_markdown`, `flush_session`, `status`. |
| `/mnemoss:recall <query>` | slash command | Quick ACT-R-ranked lookup; surfaces the top 5 hits verbatim. |
| `/mnemoss:observe <content>` | slash command | Append a memory; returns the `memory_id` for pinning later. |
| `/mnemoss:status` | slash command | Workspace status summary (memory count, tier distribution, last dream). |
| `mnemoss-aware` skill | model-invoked | Tells Claude when to reach for the memory tools — auto-fires on "remember X" / "what did we say about Y" / "forget Z". |

## Install

For the full step-by-step (prereqs, embedded vs remote, multi-workspace
patterns, troubleshooting), see [INSTALL.md](./INSTALL.md).

Quickstart:

```bash
# 1. Install the Mnemoss server-side dependency
pip install "mnemoss[mcp]"

# 2. Add this plugin from the local repo (no marketplace required for dev)
claude --plugin-dir /path/to/mnemoss/adapters/claude-cowork
```

That's it. Inside Claude, run `/help` and you'll see four `/mnemoss:*`
entries plus the `mnemoss-aware` skill in `/agents`.

## Default workspace and storage

The bundled `.mcp.json` sets `MNEMOSS_WORKSPACE=claude-cowork`. With no other
env vars, the MCP server runs in **embedded mode** — Mnemoss opens a SQLite
workspace at `~/.mnemoss/claude-cowork/` and serves all recall in-process.

To use a different workspace, edit `.mcp.json`. The other env vars
(`MNEMOSS_API_URL`, `MNEMOSS_API_KEY`, `MNEMOSS_AGENT_ID`,
`MNEMOSS_STORAGE_ROOT`) inherit from your shell — set them before launching
Claude and the MCP wrapper picks them up via `MCPConfig.from_env()`.

To point at a shared `mnemoss-server` instead of embedded mode (multi-instance
/ multi-host deployments), export `MNEMOSS_API_URL` and `MNEMOSS_API_KEY`
before launching Claude.

## How recall works

Mnemoss's recall is **not** keyword search. It's the ACT-R activation
formula:

```
A_i = B_i + Σ W_j·S_ji + MP·[w_F·s̃_F + w_S·s̃_S] + ε
       ↑      ↑              ↑                    ↑
    base-   spreading       match (semantic       noise
    level   activation      + literal)            (small)
```

Recency, frequency, related context, and semantic match all fold into a
single score per candidate memory. Top-K wins. Tier migration (HOT → WARM →
COLD → DEEP) and disposal are also formula-driven; no LLM ever decides what
to keep or forget. See `MNEMOSS_FORMULA_AND_ARCHITECTURE.md` upstream for the
math.

## License

MIT
