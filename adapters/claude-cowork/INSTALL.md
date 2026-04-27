# Install Mnemoss as a Claude Cowork / Claude Code plugin

Step-by-step guide to add Mnemoss memory to Claude (Cowork or Code). Same
plugin format works for both — Anthropic ships the spec across Cowork and
anything built on the Claude Agent SDK.

The instructions below were verified against:

- Claude Code `2.1.119`
- `mnemoss` `0.0.2` (`[mcp]` extra)
- `mcp` (Anthropic SDK) `1.27.0`
- macOS, Python 3.12

## What you'll end up with

```
~/.claude/                     (or your equivalent plugin root)
└── plugins/
    └── mnemoss/               ← this plugin
        ├── .claude-plugin/
        │   └── plugin.json
        ├── .mcp.json          ← spawns mnemoss-mcp as a stdio MCP server
        ├── skills/
        │   ├── memory-aware/  ← model-invoked: tells Claude to use the tools
        │   ├── recall/        ← /mnemoss:recall <query>
        │   ├── observe/       ← /mnemoss:observe <content>
        │   └── status/        ← /mnemoss:status
        └── README.md
```

When loaded, Claude gets:

- 13 MCP tools under `mcp__mnemoss__*` (observe, recall, expand, pin, dream,
  status, etc.)
- 4 namespaced skills (`/mnemoss:recall`, `/mnemoss:observe`, `/mnemoss:status`,
  plus the auto-invoked `mnemoss:memory-aware`)

## Prerequisites

- Claude Code ≥ 2.1 (`claude --version`) or Claude Cowork with plugins enabled
- Python 3.10+ on the same machine that runs Claude
- ~600 MB disk for the multilingual MiniLM embedder (downloaded once on first
  recall, cached after that)

## Step 1 — Install the Mnemoss MCP backend on your PATH

The plugin's `.mcp.json` invokes `mnemoss-mcp` by name. That binary has to be
discoverable by the Claude process — meaning **on the user-shell PATH that
Claude inherits, not just inside a venv that you `source`'d for one terminal.**

The simplest path: install Mnemoss into a Python that's already on your global
PATH (system Python, pyenv default, Homebrew Python, whatever):

```bash
pip install "mnemoss[mcp]"
which mnemoss-mcp     # → must print a path
```

If `which mnemoss-mcp` returns nothing, the install went into a Python that
isn't first on your PATH. Either:

- Activate that venv before launching Claude (e.g.
  `source ~/.venvs/mnemoss/bin/activate && claude`), or
- Install into a Python whose `bin/` is already on PATH, or
- Edit `.mcp.json` and replace `"command": "mnemoss-mcp"` with the absolute
  path (e.g. `/Users/you/.venvs/mnemoss/bin/mnemoss-mcp`).

Confirm the install works:

```bash
python3 -c "from mnemoss.mcp.cli import main; print('mnemoss-mcp OK')"
```

## Step 2 — Drop the plugin into Claude

### Quick / dev: load directly with `--plugin-dir`

For local iteration, no marketplace required:

```bash
git clone https://github.com/opcify/mnemoss.git
cd mnemoss
claude --plugin-dir adapters/claude-cowork
```

Inside Claude, run `/help` — you should see four `/mnemoss:*` entries plus
the `mnemoss-aware` skill in `/agents`.

To edit the plugin and reload without restarting:

```text
/reload-plugins
```

### Permanent / team: install via marketplace

Once you've vetted the plugin, distribute it through Claude's marketplace
mechanism. The Anthropic-recommended pattern is a git-backed marketplace
repo containing the plugin. Reference docs:
https://code.claude.com/docs/en/plugin-marketplaces

For Cowork specifically, organizations build private marketplaces that
distribute plugins to all team members. See:
https://support.claude.com/en/articles/13837433-manage-claude-cowork-plugins-for-your-organization

## Step 3 — Configure (optional)

The plugin works out of the box with sensible defaults. Override via env vars
before launching Claude:

| Var | Default | What it controls | How to override |
|---|---|---|---|
| `MNEMOSS_WORKSPACE` | `claude-cowork` | Workspace id. One workspace per logical "memory bucket". | Hardcoded by `.mcp.json` — edit that file to change. |
| `MNEMOSS_AGENT_ID` | unset | If set, every observe/recall scopes to this agent. Recall returns this agent's memories plus workspace-ambient ones. | Set in your shell before launching Claude. |
| `MNEMOSS_API_URL` | unset | If set, MCP wrapper runs in **remote mode** against a `mnemoss-server` instead of opening an embedded SQLite workspace. | Set in your shell. |
| `MNEMOSS_API_KEY` | unset | Bearer token for remote-mode auth. | Set in your shell. |
| `MNEMOSS_STORAGE_ROOT` | `~/.mnemoss` | Where embedded-mode SQLite files live. | Set in your shell. |

`MNEMOSS_WORKSPACE` is hardcoded in the plugin's `.mcp.json` rather than read from
parent shell env, because Claude Code's MCP loader uses the `env` block as an
override (parent env loses to the block for the keys it sets). Edit the file if
you want a different workspace, or wire your own env-var substitution at the
loader level.

Embedded mode is the zero-config default. Use remote mode when multiple
Claude instances (e.g. several teammates' Cowork sessions) should share
memory through a single Mnemoss server — see the openclaw and hermes-agent
INSTALL guides for the server-side recipe.

## Step 4 — Verify

### Plugin validates

```bash
claude plugin validate "$(pwd)/adapters/claude-cowork"
# → ✔ Validation passed
```

### Skills appear in Claude

Inside a session loaded with `--plugin-dir`:

```text
/help
```

Look for:
- `mnemoss:recall` (your turn)
- `mnemoss:observe` (your turn)
- `mnemoss:status` (your turn)
- The `mnemoss:memory-aware` skill listed under `/agents`

### MCP tools appear in Claude

Ask Claude (in print mode for a quick check):

```bash
PATH="/path/to/python/bin:$PATH" claude --plugin-dir adapters/claude-cowork \
  -p "List MCP tools whose names contain 'mnemoss'. Names only, no tool calls."
```

Expected — 13 tools:

```
mcp__mnemoss__dispose
mcp__mnemoss__dream
mcp__mnemoss__expand
mcp__mnemoss__explain_recall
mcp__mnemoss__export_markdown
mcp__mnemoss__flush_session
mcp__mnemoss__observe
mcp__mnemoss__pin
mcp__mnemoss__rebalance
mcp__mnemoss__recall
mcp__mnemoss__status
mcp__mnemoss__tier_counts
mcp__mnemoss__tombstones
```

If you only see `(none)`, the `mnemoss-mcp` binary isn't on the PATH the
Claude process inherits. Re-do step 1 or use an absolute path in `.mcp.json`.

### End-to-end round-trip (manual, in an interactive session)

```bash
claude --plugin-dir adapters/claude-cowork
```

Then:

```text
/mnemoss:status
```

First time on a fresh workspace, expect an empty memory_count. Then:

```text
/mnemoss:observe Hello world from the Mnemoss plugin
```

Approve the `mcp__mnemoss__observe` call when prompted. You'll get back a
`memory_id` ULID. Then:

```text
/mnemoss:recall hello world
```

Approve the recall call. Expect the memory you just observed back as the
top hit.

Workspace files materialize at `~/.mnemoss/claude-cowork/`:

```bash
ls ~/.mnemoss/claude-cowork/
# memory.sqlite, raw_log.sqlite, .mnemoss.lock
```

## Troubleshooting

**`/help` doesn't show any `/mnemoss:*` entries.** Either `--plugin-dir`
isn't pointing at the right path, or the plugin manifest didn't validate.
Run `claude plugin validate /absolute/path/to/adapters/claude-cowork`. The
path must point at the *plugin root* (the dir containing `.claude-plugin/`),
not at `.claude-plugin/` itself.

**`mcp__mnemoss__*` tools missing.** `mnemoss-mcp` isn't on Claude's PATH.
See step 1.

**MCP server starts but `tools/call` fails with a workspace error.**
`MNEMOSS_STORAGE_ROOT` isn't writable, or a different Mnemoss process is
already holding the workspace lock (`.mnemoss.lock`). Embedded mode is
single-writer per workspace by design — switch to remote mode if you need
multiple concurrent writers.

**`Loading weights:` lag on first observe/recall.** That's the multilingual
MiniLM embedder downloading + warming. ~30s on first call, ~0ms after.

**Skills work but the `mnemoss-aware` skill never auto-fires.** The skill's
description has to match the user's intent for Claude to pick it up. Try
saying the magic words: "remember this", "what do we know about X",
"forget Y". If it still doesn't fire, sharpen the description in
`skills/memory-aware/SKILL.md` (it's user-editable — Claude reads frontmatter
descriptions when deciding which skills to invoke).

**Stale memories after upgrading Mnemoss.** Mnemoss has a migration framework
(`src/mnemoss/store/migrations.py`) that auto-applies older→newer schema
chains on workspace open. Run the upgrade once with `mnemoss-inspect <ws>`
before pointing Claude at the workspace, just to surface any migration
errors out-of-band.

## Uninstall

If you used `--plugin-dir` (no marketplace install): just stop passing the
flag. Nothing to remove from Claude's settings.

If you marketplace-installed: `claude plugin uninstall mnemoss`.

```bash
# Optionally wipe the workspace (irreversible)
rm -rf ~/.mnemoss/claude-cowork
pip uninstall mnemoss
```
