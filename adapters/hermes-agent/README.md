# mnemoss-hermes

[Mnemoss](https://github.com/opcify/mnemoss) as a memory provider for 
[Hermes Agent](https://github.com/NousResearch/hermes-agent).

Drops into Hermes's memory plugin slot alongside the built-in 
MEMORY.md/USER.md store. Mnemoss handles per-turn recall (ACT-R ranked), 
tool-driven on-demand lookups, and opportunistic consolidation via 
Dreaming. The built-in memory stays active; Mnemoss adds the 
cross-session ACT-R layer.

## What you get

- **Auto-injection** — before each turn, Mnemoss recalls the top-k most 
  activated memories for the user's message and prepends them to the 
  prompt.
- **Tool-driven recall** — three tools (`mnemoss_recall`, 
  `mnemoss_expand`, `mnemoss_pin`) let the model do ad-hoc lookups, 
  follow relation graphs, and pin important facts.
- **Session consolidation** — at session end, a `session_end` dream 
  cycle extracts facts, updates the relation graph, and rebalances 
  the activation tiers.
- **Pre-compression rescue** — before Hermes compresses old context, 
  Mnemoss contributes its standing-memory block so important facts 
  survive the compression pass.
- **Builtin-write mirroring** — when Hermes writes to MEMORY.md/USER.md, 
  Mnemoss observes the same content, keeping the two stores in sync.

## Install

Two forms; pick one. **Embedded** runs Mnemoss in-process — 
low-latency, database lives under `$HERMES_HOME/mnemoss/`. **Remote** 
points at a shared `mnemoss-server`, which is what you want when 
multiple Hermes instances should share memory.

```bash
# Embedded (default) — Mnemoss runs in-process
pip install mnemoss-hermes

# Remote — talk to a shared mnemoss-server over HTTP
pip install "mnemoss-hermes[remote]"
```

Then drop the plugin into Hermes's plugin tree. The simplest route is 
to symlink it into `plugins/memory/mnemoss/`:

```bash
ln -s $(python -c "import mnemoss_hermes, pathlib; print(pathlib.Path(mnemoss_hermes.__file__).parent)") \
      ~/path/to/hermes-agent/plugins/memory/mnemoss
```

Alternatively, copy the package into your Hermes fork and commit.

## Configure

Mnemoss reads config in this precedence (highest first):

1. `$HERMES_HOME/mnemoss.json` — user-editable JSON.
2. Environment variables: `MNEMOSS_BASE_URL`, `MNEMOSS_API_KEY`, 
   `MNEMOSS_WORKSPACE`.
3. Defaults: embedded mode, workspace = Hermes agent identity (e.g. 
   `"coder"`, falling back to `"hermes"`).

### Embedded example

`$HERMES_HOME/mnemoss.json`:

```json
{
  "workspace": "my-hermes-agent"
}
```

That's it. Mnemoss opens a SQLite database at 
`$HERMES_HOME/mnemoss/workspaces/my-hermes-agent/` and runs everything 
in-process.

### Remote example

`$HERMES_HOME/mnemoss.json`:

```json
{
  "baseUrl": "https://memory.internal.example.com",
  "workspace": "shared-agents"
}
```

Plus `MNEMOSS_API_KEY` in your env (or `.env`). Mnemoss routes all 
observe/recall/dream calls through the REST server. Multiple Hermes 
instances pointing at the same base URL share memory.

### Activate the provider

In Hermes's config, set the active memory provider to `mnemoss`:

```json5
{
  memory: {
    provider: "mnemoss",
  },
}
```

Or use `hermes memory setup` — the CLI wizard lists Mnemoss once the 
plugin is discovered and will walk you through the schema.

## How it integrates with Hermes

| Hermes lifecycle hook | Mnemoss side effect |
|---|---|
| `initialize` | Resolve config, build embedded `Mnemoss` or remote `WorkspaceHandle`. |
| `system_prompt_block` | Static block explaining the three tools. |
| `queue_prefetch(query)` | Run `mem.recall(query, k=5)` and cache for next turn. |
| `prefetch(query)` | Return the cached block; first turn runs recall synchronously. |
| `sync_turn(user, assistant)` | `mem.observe(role="user")` then `mem.observe(role="assistant")`. |
| `on_session_end` | `mem.dream(trigger="session_end")` — P1–P5 consolidation. |
| `on_pre_compress` | `mem.export_markdown(min_idx_priority=0.7)` — hand HOT block to compressor. |
| `on_memory_write(add, user/memory, content)` | Mirror the write as a Mnemoss observation. |
| `shutdown` | Close backend + SDK client; flush any pending work. |

## Tools exposed to the model

- **`mnemoss_recall(query, k?, include_deep?)`** — ACT-R ranked search.
- **`mnemoss_expand(memory_id, hops?, k?)`** — follow the relation 
  graph from one memory (associations).
- **`mnemoss_pin(memory_id)`** — lock a memory to HOT, exempt it from 
  disposal.

The tools are only exposed when the plugin is active (not under cron 
or flush contexts).

## Troubleshooting

**Plugin not appearing in `hermes memory setup`?** Check that the 
symlink/copy landed in `plugins/memory/mnemoss/` AND that the 
directory contains `plugin.yaml`. Hermes discovers plugins by walking 
that tree.

**Memory is empty on the first turn?** Mnemoss's first-turn prefetch 
runs synchronously — if it's timing out, the server (remote mode) is 
probably unreachable. Check `MNEMOSS_BASE_URL` and your API key.

**Schema mismatch error?** Your Mnemoss database was created by a 
different version of Mnemoss. Run `rm -rf $HERMES_HOME/mnemoss` to 
start fresh (Stage-1 policy: no migrations), or point at a fresh 
workspace via `MNEMOSS_WORKSPACE`.

## Standalone tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use a mock Mnemoss backend and a stubbed `MemoryProvider` base 
class (since hermes-agent isn't a pip dep), so they run cleanly 
outside a Hermes install.

## License

MIT
