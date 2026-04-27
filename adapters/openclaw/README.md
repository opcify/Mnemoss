# @mnemoss/openclaw-plugin

[Mnemoss](https://github.com/opcify/mnemoss) as an 
[OpenClaw](https://github.com/openclaw/openclaw) memory plugin.

Wires Mnemoss's ACT-R recall engine into OpenClaw via the 
`registerMemoryCapability` path. Replaces OpenClaw's bundled SQLite-based 
memory engine with Mnemoss's activation-ranked recall, relation-graph 
expansion, and opportunistic consolidation.

## How it works

OpenClaw's memory host defines a `MemorySearchManager` interface — 
`search`, `readFile`, `status`, `probeEmbeddingAvailability`, 
`probeVectorAvailability`. The plugin implements that interface by 
delegating every call to a shared `mnemoss-server` via the 
[`@mnemoss/sdk`](../../sdks/typescript) TypeScript client:

```
OpenClaw request handler
       │
       ▼
  memory.search(query)
       │
       ▼
  MnemossSearchManager.search(query)   ← this plugin
       │
       ▼
  @mnemoss/sdk WorkspaceHandle.recall()
       │
       ▼  HTTP
  mnemoss-server /workspaces/{id}/recall
       │
       ▼
  ACT-R formula + cascade + expansion
```

All OpenClaw's built-in memory features stay available alongside — this 
plugin replaces the memory *engine*, not the plugin host that sits above 
it.

## Install

For an end-to-end walkthrough — Docker container, Mnemoss server, plugin
config, verification — see [INSTALL.md](./INSTALL.md).

Quickstart for a published plugin:

```bash
openclaw plugins install @mnemoss/openclaw-plugin
openclaw config set plugins.slots.memory mnemoss
```

For local development against a checkout of this monorepo:

```bash
cd adapters/openclaw
npm install
npm run build
openclaw plugins install --link $(pwd)
```

## Configure

The plugin needs a running `mnemoss-server`. Any combination of config 
file and environment variables works; explicit config wins:

### `openclaw.config.json5` (plugin config)

```json5
{
  plugins: {
    entries: {
      "mnemoss": {
        config: {
          baseUrl: "https://memory.internal.example.com",
          apiKey: "",                // optional
          workspace: "shared-agents", // default: gateway id
          timeoutMs: 30000,
        },
      },
    },
  },
  memory: {
    backend: "builtin",   // label — MemorySearchManager slot is shared
  },
}
```

### Environment variables

| Variable | Purpose |
|---|---|
| `MNEMOSS_BASE_URL` | URL of the Mnemoss REST server. Required. |
| `MNEMOSS_API_KEY` | Bearer token (if the server is protected). |
| `MNEMOSS_WORKSPACE` | Default workspace id. |

Precedence: plugin config > env vars > fallback to OpenClaw's agent id 
as the workspace.

## Running a Mnemoss server

```bash
pip install "mnemoss[server,observability]"
export MNEMOSS_API_KEY=...
mnemoss-server --host 0.0.0.0 --port 8000
```

Multiple OpenClaw gateways can share the same server — memories end up 
scoped per-workspace and per-agent, so gateways can either share a 
workspace (shared memory) or use distinct ones (isolated).

## Agent scoping

Every `search` call from OpenClaw includes an `agentId`. The plugin 
forwards this as Mnemoss's `agent_id` — the recall returns memories 
owned by that agent plus workspace-ambient ones (matching Mnemoss's 
standard `agent_id = X OR agent_id IS NULL` rule). OpenClaw multi-agent 
gateways "just work"; agent isolation is enforced by Mnemoss.

## What the status row shows

OpenClaw's `memory status` command reports:

```
Backend: builtin
Provider: mnemoss
Sources: memory
Notes: Remote mnemoss-server; files/chunks counts not exposed over REST.
```

"builtin" is the capability slot — Mnemoss takes over the role OpenClaw's 
bundled engine normally fills. File/chunk counts aren't reported because 
Mnemoss's model isn't file-oriented; the `memory_count` and `tier_counts` 
fields from the Mnemoss status endpoint give a more meaningful view via 
`mnemoss-server /workspaces/{id}/status`.

## Read contract

Mnemoss doesn't have a file-line model — memories are self-contained. 
`readFile(relPath, from?, lines?)` works only for `mnemoss://{id}` URIs 
the plugin has seen in a recent `search` response (cached in an LRU of 
the last 256 hits). `from`/`lines` slicing treats the memory content as 
a single document; out-of-cache URIs return empty text rather than 
erroring, matching OpenClaw's tolerance for missing references.

## Tests

```bash
npm test
```

Tests mock the server's `fetch` response at the HTTP boundary — the 
real SDK code runs. No running Mnemoss needed.

```bash
npm run typecheck
```

Typechecks against the SDK's own `dist/` output.

## License

MIT
