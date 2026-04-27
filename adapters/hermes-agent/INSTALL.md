# Install Mnemoss as your default Hermes-Agent memory provider

Step-by-step guide for a Hermes-Agent user who wants to replace the bundled
`MEMORY.md`/`USER.md` store with Mnemoss's ACT-R memory — either embedded
in-process or against a shared `mnemoss-server`.

This is the install side of [README.md](./README.md), which covers *how* the
provider works. Read README first if you want the architecture; this doc walks
you through getting it running.

The instructions below were verified against:

- `mnemoss-hermes` `0.1.0`
- `mnemoss` `0.0.2` (embedded + remote modes)
- Python 3.12 (3.10+ should work)

If you'd rather run the embedded path, skip the server section and jump to
"[Embedded mode](#embedded-mode)" below.

## What you'll end up with

```
Hermes-Agent (your fork or upstream)
├── plugins/
│   └── memory/
│       └── mnemoss/        ← the Mnemoss provider plugin
│           ├── plugin.yaml
│           └── src/mnemoss_hermes/...
└── ~/.hermes/
    ├── config.json5        ← memory.provider = "mnemoss"
    └── mnemoss.json        ← optional: workspace, baseUrl, api_key
```

Two modes, pick one per Hermes install:

| Mode | When to use | Storage |
|---|---|---|
| **Embedded** (default) | Single-user / single-Hermes-instance. Lowest latency. | `~/.hermes/mnemoss/workspaces/<id>/{memory,raw_log}.sqlite` |
| **Remote** | Multiple Hermes instances should share memory. | A running `mnemoss-server` you point them all at. |

## Prerequisites

- Hermes-Agent checked out and runnable on your machine
  (https://github.com/NousResearch/hermes-agent — clone or fork it; not yet on
  PyPI as a pip-installable package).
- Python 3.10+.
- For remote mode: a running `mnemoss-server` (covered below).
- ~600 MB disk on first embedded launch — Mnemoss downloads the multilingual
  MiniLM embedder model (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dims).
  Cached after that.

## Step 1 — Install the plugin package

In whatever Python environment Hermes-Agent runs in:

```bash
# Embedded — Mnemoss runs in-process, all you need is the core package
pip install mnemoss-hermes

# Remote — adds the [sdk] extra so the provider can talk to mnemoss-server
pip install "mnemoss-hermes[remote]"
```

For local development against this monorepo:

```bash
pip install -e "/path/to/mnemoss/adapters/hermes-agent[remote]"
```

Confirm the import works:

```bash
python3 -c 'from mnemoss_hermes import MnemossMemoryProvider; print(MnemossMemoryProvider().is_available())'
# → True
```

If `is_available()` returns False, `import mnemoss` failed inside the provider
— the core `mnemoss` package isn't on the same Python path. Re-check the
install (`pip show mnemoss`).

## Step 2 — Drop the plugin into Hermes's plugin tree

Hermes-Agent discovers memory plugins by walking `plugins/memory/<name>/` and
reading each `plugin.yaml`. Symlink (or copy) the installed package into that
tree:

```bash
PLUGIN_SRC="$(python3 -c 'import mnemoss_hermes, pathlib; print(pathlib.Path(mnemoss_hermes.__file__).parent.parent.parent)')"
ln -s "$PLUGIN_SRC" /path/to/hermes-agent/plugins/memory/mnemoss
```

`PLUGIN_SRC` is the package root (where `plugin.yaml` lives). For an editable
install of this monorepo it's the `adapters/hermes-agent/` directory.

If you'd rather copy than symlink (e.g. you commit it into your Hermes fork):

```bash
cp -r "$PLUGIN_SRC" /path/to/hermes-agent/plugins/memory/mnemoss
```

Confirm Hermes sees it:

```bash
ls /path/to/hermes-agent/plugins/memory/mnemoss/plugin.yaml
# → ...plugin.yaml exists
```

## Step 3 — Configure

Hermes reads provider-specific config from `$HERMES_HOME/mnemoss.json` (default
`~/.hermes/mnemoss.json`). Create one to match the mode you picked.

### Embedded mode

`~/.hermes/mnemoss.json`:

```json
{
  "workspace": "my-hermes-agent"
}
```

That's it. `workspace` defaults to your Hermes agent identity (e.g. `coder`)
falling back to `"hermes"` — set it explicitly only if you want to override.

The SQLite files live under `~/.hermes/mnemoss/workspaces/<workspace>/`. Back
that path up if you want to preserve your agent's memory across reinstalls.

### Remote mode

#### 3a. Run a Mnemoss server

```bash
python3 -m venv /tmp/mnemoss-venv
source /tmp/mnemoss-venv/bin/activate
pip install "mnemoss[server]"
mkdir -p /var/lib/mnemoss

export MNEMOSS_HOST=127.0.0.1
export MNEMOSS_PORT=8765
export MNEMOSS_API_KEY="$(openssl rand -hex 24)"
export MNEMOSS_STORAGE_ROOT=/var/lib/mnemoss
mnemoss-server
# → INFO: Uvicorn running on http://127.0.0.1:8765 (Press CTRL+C to quit)
```

Save the `MNEMOSS_API_KEY` you generated for the next step.

For multi-Hermes deployments where Hermes runs on a different machine, bind
the server to the address that Hermes can reach (it must NOT be exposed to
hostile networks unless `MNEMOSS_API_KEY` is set).

#### 3b. Point the provider at the server

`~/.hermes/mnemoss.json`:

```json
{
  "baseUrl": "http://127.0.0.1:8765",
  "api_key": "<the MNEMOSS_API_KEY you just generated>",
  "workspace": "shared-agents"
}
```

Or via env (lower precedence than the JSON file but works for one-off testing):

```bash
export MNEMOSS_BASE_URL=http://127.0.0.1:8765
export MNEMOSS_API_KEY=...
export MNEMOSS_WORKSPACE=shared-agents
```

Multiple Hermes instances pointing at the same `baseUrl` + `workspace` share
memory. Distinct workspaces give isolation. Per-agent isolation within a
workspace is automatic via `agent_id` (Mnemoss's `agent_id = X OR agent_id IS
NULL` rule).

### Activate the provider in Hermes config

Edit `~/.hermes/config.json5` (or wherever your Hermes config lives) and set
the active memory provider:

```json5
{
  memory: {
    provider: "mnemoss",
  },
}
```

Or use the CLI wizard which lists discovered plugins:

```bash
hermes memory setup
# → choose "mnemoss" from the list, fill in any prompts
```

## Step 4 — Verify

### From the Hermes process

Start Hermes once and open a conversation. Watch for two things:

- The system prompt block that Mnemoss injects: `# Mnemoss Memory\nActive. ...
  When you need to dig deeper, use the mnemoss_recall tool ...`. If you see
  it, `initialize()` succeeded.
- Three new tools available to the model: `mnemoss_recall`, `mnemoss_expand`,
  `mnemoss_pin`.

If you don't see them, run `hermes memory status` (if your Hermes version has
it) or check `hermes` logs for "Mnemoss init failed" — that's the provider
catching an exception during startup so the agent doesn't crash.

### Drive the provider directly (the most decisive check)

Independent of Hermes. Demonstrates the wire path is correct:

```bash
cat > /tmp/probe.py <<'EOF'
import json, time
from mnemoss_hermes import MnemossMemoryProvider

p = MnemossMemoryProvider(config={
    "baseUrl": "http://127.0.0.1:8765",   # remove for embedded
    "api_key": "<key>",                    # remove for embedded
    "workspace": "verify",
})
p.initialize(session_id="s1", user_id="u1", agent_identity="probe", agent_context="primary")
print("backend:", type(p._backend).__name__)
print("tools:", [t["name"] for t in p.get_tool_schemas()])

p.sync_turn("What is Mnemoss?", "ACT-R memory for AI agents.")
p.sync_turn("How does it rank?", "Activation: base-level + spreading + match.")
time.sleep(0.2)

out = json.loads(p.handle_tool_call("mnemoss_recall", {"query":"how does mnemoss rank","k":3}))
print("hits:", len(out["results"]))
for r in out["results"]:
    print(f"  {r['id']}  score={r['score']:.4f}")

p.shutdown()
EOF
python3 /tmp/probe.py
```

**Embedded expected output:**
```
backend: Mnemoss
tools: ['mnemoss_recall', 'mnemoss_expand', 'mnemoss_pin']
hits: 2
  01K...  score=0.7021
  01K...  score=0.4723
```

**Remote expected output:** same as above but `backend: WorkspaceHandle`, and
`mnemoss-server` logs three POST /observe calls (sync_turn × 2 = 4 observes,
the recall search) plus a POST /recall.

If you see this, the plugin is wired end-to-end.

### Confirm the workspace exists (embedded only)

```bash
ls ~/.hermes/mnemoss/workspaces/<workspace>/
# Expect: memory.sqlite, raw_log.sqlite, .mnemoss.lock
```

After a session ends and `on_session_end` runs the consolidation dream:

```bash
ls ~/.hermes/mnemoss/workspaces/<workspace>/dreams/
# Dream artifacts (clusters, summaries, relations) land here.
```

## How Hermes lifecycle hooks map to Mnemoss

For reference; you don't have to do anything for these to fire:

| Hermes hook | What Mnemoss does |
|---|---|
| `initialize` | Build embedded `Mnemoss` or remote `WorkspaceHandle`. |
| `system_prompt_block` | Static block explaining the three tools. |
| `queue_prefetch(query)` | `mem.recall(query, k=5)` cached for next turn. |
| `prefetch(query)` | Return cached block; first turn runs synchronously. |
| `sync_turn(user, assistant)` | Two `mem.observe` calls. |
| `on_session_end` | `mem.dream(trigger="session_end")` — full P1–P5 consolidation. |
| `on_pre_compress` | `mem.export_markdown(min_idx_priority=0.7)` returns the HOT block. |
| `on_memory_write(action, target, content)` | Mirror the Hermes builtin write as an observation. |
| `shutdown` | Close backend + SDK client. |

## Troubleshooting

**`is_available()` returns False.** `import mnemoss` is failing inside the
provider. The package might be installed in a different Python env than the
one Hermes uses — check `pip show mnemoss` against `which python3` from inside
Hermes's venv.

**`Plugin not appearing in 'hermes memory setup'`.** The symlink/copy at
`plugins/memory/mnemoss/` doesn't contain `plugin.yaml`, or Hermes doesn't see
that path. Verify with `ls plugins/memory/mnemoss/plugin.yaml`. Hermes
discovers by walking the directory tree — broken symlinks count as missing.

**Empty memory on first turn.** First-turn prefetch runs synchronously. If
that times out (remote mode), the server is unreachable. Check
`MNEMOSS_BASE_URL` and the API key.

**Schema-mismatch error on startup.** Your workspace was created by a
different Mnemoss version. Either upgrade Mnemoss (the migration framework
auto-applies older→newer), or point at a fresh workspace via
`MNEMOSS_WORKSPACE=<new-id>`. The migration chain is in
`src/mnemoss/store/migrations.py`.

**`Mnemoss init failed` in Hermes logs.** The provider is designed never to
hard-fail Hermes startup; it logs the exception and goes inactive. Check
the log for the underlying error — usually a missing dep
(`pip install "mnemoss-hermes[remote]"` for SDK mode), wrong baseUrl, or a
file-system permission issue under `$HERMES_HOME`.

**Embedded mode: workspace dir doesn't appear.** Mnemoss creates
`~/.hermes/mnemoss/workspaces/<id>/` lazily on the first observe. If you
called `initialize()` but no `sync_turn` ever fired, the dir won't exist yet.

**Multiple Hermes processes against the same embedded workspace.** Don't.
Embedded mode uses an `fcntl` advisory lock (`.mnemoss.lock`) — a second
process raises `WorkspaceLockError`. If you want shared memory across Hermes
instances, run `mnemoss-server` and point them all at it (remote mode).

## Uninstall

```bash
# Remove the symlink/copy from Hermes
rm /path/to/hermes-agent/plugins/memory/mnemoss

# Remove the config
rm ~/.hermes/mnemoss.json

# Reset memory.provider back to the builtin in Hermes config

# (Optional) wipe Mnemoss data
rm -rf ~/.hermes/mnemoss   # embedded
# Remote: leave the server alone, or delete the workspace via the CLI:
# mnemoss-inspect <workspace>  # check what's there first
```

```bash
pip uninstall mnemoss-hermes
```
