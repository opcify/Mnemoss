# Install Mnemoss as your default OpenClaw memory backend

Step-by-step guide for an OpenClaw 2026.4.14+ user who wants to replace the
bundled SQLite-backed memory engine with Mnemoss's ACT-R recall.

This is the install side of [README.md](./README.md), which covers
*how* the adapter works. Read README first if you want the architecture; this
doc just walks you through getting it running.

The instructions below were verified against:

- OpenClaw `2026.4.14` (Docker image `qiguangyang/openclaw:latest`)
- `@mnemoss/openclaw-plugin` `0.1.0`
- `@mnemoss/sdk` `0.1.0`
- `mnemoss[server]` `0.0.2` (PyPI track), Python 3.12

## What you'll end up with

```
┌─────────────────────────┐         ┌──────────────────────────┐
│ OpenClaw gateway        │         │ mnemoss-server (FastAPI) │
│ (Docker container)      │  HTTP   │                          │
│   plugins.slots.memory  │ ──────▶ │ /workspaces/{id}/recall  │
│         = mnemoss       │         │ /workspaces/{id}/observe │
│                         │         │ /workspaces/{id}/status  │
└─────────────────────────┘         └──────────────────────────┘
            │                                    │
            │ memory_search tool calls           │ ACT-R formula
            │ pass through MnemossSearchManager  │ ANN top-K + cascade
            │                                    │ HNSW + WAL SQLite
            ▼                                    ▼
   OpenClaw agent uses                  ~/.mnemoss/{workspace}/
   Mnemoss as the only                  ├── memory.sqlite
   memory backend                       └── raw_log.sqlite
```

OpenClaw's `MemorySearchManager` slot is occupied by `MnemossSearchManager`. Every
`memory_search` tool call from any agent on this gateway is forwarded over HTTP to
the Mnemoss REST server. Agent isolation is enforced by Mnemoss's per-row
`agent_id` (the recall is `agent_id = X OR agent_id IS NULL`, the standard
"private + ambient" rule).

## Prerequisites

- Docker (with `host.docker.internal` resolving — Docker Desktop on macOS/Windows or
  Linux with the `--add-host=host.docker.internal:host-gateway` flag).
- Python 3.10+ (only required if you run `mnemoss-server` on the host; not
  needed if you run it in a container).
- OpenClaw 2026.4.14 or newer.
- Disk: ~600 MB the first time `mnemoss-server` boots, because it downloads the
  default multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`,
  384 dims, 50+ languages).

## Step 1 — Run a Mnemoss server

There are two paths. Pick one. Path A is the simplest and the only one verified
end-to-end below; path B is sketched if you'd rather have everything in Docker.

### Path A — host process (verified)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "mnemoss[server]"

mkdir -p /var/lib/mnemoss        # or wherever you want the SQLite files
export MNEMOSS_HOST=127.0.0.1
export MNEMOSS_PORT=8765
export MNEMOSS_API_KEY="$(openssl rand -hex 24)"
export MNEMOSS_STORAGE_ROOT=/var/lib/mnemoss
mnemoss-server
# → INFO: Uvicorn running on http://127.0.0.1:8765
```

Save the `MNEMOSS_API_KEY` you generated — you'll paste it into the OpenClaw
config in step 3.

Sanity-check that the OpenClaw container will be able to reach it:

```bash
docker run --rm curlimages/curl \
  -sf -m 5 -w "HTTP %{http_code}\n" \
  http://host.docker.internal:8765/health
# → {"ok":true}HTTP 200
```

If that fails, your Docker doesn't route `host.docker.internal` to the host
loopback. On Linux Docker, you can opt in by adding
`--add-host=host.docker.internal:host-gateway` to every `docker run`.

### Path B — both in Docker

There's no official Mnemoss image yet. Roll your own:

```dockerfile
# Dockerfile.mnemoss
FROM python:3.12-slim
RUN pip install --no-cache-dir "mnemoss[server]"
ENV MNEMOSS_HOST=0.0.0.0
ENV MNEMOSS_PORT=8765
ENV MNEMOSS_STORAGE_ROOT=/data
EXPOSE 8765
VOLUME ["/data"]
CMD ["mnemoss-server"]
```

```bash
docker build -t mnemoss-server -f Dockerfile.mnemoss .
docker network create openclaw-net 2>/dev/null || true
docker run -d --name mnemoss-server \
  --network openclaw-net \
  -e MNEMOSS_API_KEY="$(openssl rand -hex 24)" \
  -v mnemoss-data:/data \
  mnemoss-server
```

When you launch OpenClaw on the same `openclaw-net` network, your `baseUrl`
becomes `http://mnemoss-server:8765` (container DNS) instead of
`http://host.docker.internal:8765`.

## Step 2 — Install the plugin into OpenClaw

The plugin is published on npm as `@mnemoss/openclaw-plugin`, but if you want
to track a specific commit (or run from a local checkout while the package is
pre-1.0), use the `--link` mode below. Both options end up in the same place.

### Option 2a — npm (when published)

```bash
docker exec <your-openclaw-container> openclaw plugins install @mnemoss/openclaw-plugin
```

### Option 2b — local checkout (verified)

The plugin's `package.json` declares `@mnemoss/sdk` as a `file:` ref to the
sibling SDK package. That works fine on the dev machine, but not after the
folder is mounted into a Docker container — the relative path no longer
resolves. Build a self-contained bundle first:

```bash
# In the mnemoss repo
cd sdks/typescript && npm install && npm run build && cd -
cd adapters/openclaw && npm install && npm run build && cd -

# Vendor the SDK under the plugin's node_modules
mkdir -p /tmp/mnemoss-plugin/node_modules/@mnemoss/sdk
cp -r adapters/openclaw/dist /tmp/mnemoss-plugin/
cp adapters/openclaw/openclaw.plugin.json /tmp/mnemoss-plugin/
cp adapters/openclaw/README.md /tmp/mnemoss-plugin/
node -e '
  const fs = require("fs");
  const p = JSON.parse(fs.readFileSync("adapters/openclaw/package.json", "utf8"));
  delete p.dependencies["@mnemoss/sdk"];
  delete p.devDependencies; delete p.scripts;
  p.bundledDependencies = ["@mnemoss/sdk"];
  fs.writeFileSync("/tmp/mnemoss-plugin/package.json", JSON.stringify(p, null, 2));
'
cp -r sdks/typescript/dist /tmp/mnemoss-plugin/node_modules/@mnemoss/sdk/
cp sdks/typescript/package.json /tmp/mnemoss-plugin/node_modules/@mnemoss/sdk/

# Mount it into the container (do this when starting your OpenClaw container)
docker run -d \
  --name openclaw \
  -v /tmp/mnemoss-plugin:/opt/mnemoss-plugin:ro \
  -v <your-openclaw-data-dir>:/home/node/.openclaw \
  -p <host-port>:18789 \
  qiguangyang/openclaw:latest \
  docker-entrypoint.sh node openclaw.mjs gateway --allow-unconfigured \
  --port 18790 --bind loopback
```

The two extra gateway flags are required because the image runs a `socat` port
forwarder on `18789` and the gateway itself binds to `18790`. Existing
production containers already configure this in `openclaw.json`; if you're
launching a fresh one, pass it on the command line.

## Step 3 — Configure the plugin

`openclaw plugins install` validates plugin config against the plugin's schema,
which requires a `baseUrl`. Seed the config block before installing.

Inside the container:

```bash
# Pre-seed plugin config (so install validation passes)
node -e '
  const fs = require("fs");
  const path = "/home/node/.openclaw/openclaw.json";
  const c = JSON.parse(fs.readFileSync(path, "utf8"));
  c.plugins = c.plugins || {};
  c.plugins.entries = c.plugins.entries || {};
  c.plugins.entries.mnemoss = {
    config: {
      baseUrl: "http://host.docker.internal:8765",
      apiKey: process.env.MNEMOSS_API_KEY || "",
      workspace: "openclaw",
      timeoutMs: 30000
    }
  };
  fs.writeFileSync(path, JSON.stringify(c, null, 2));
'

# Now install the plugin (npm or local --link)
openclaw plugins install --link /opt/mnemoss-plugin

# Switch the memory slot to mnemoss
openclaw config set plugins.slots.memory mnemoss
```

Or do all three with `openclaw config set --batch-file` if you prefer JSON:

```json
[
  { "path": "plugins.slots.memory", "value": "mnemoss" },
  { "path": "plugins.entries.mnemoss.config.baseUrl",
    "value": "http://host.docker.internal:8765" },
  { "path": "plugins.entries.mnemoss.config.apiKey",
    "value": "<your-mnemoss-api-key>" },
  { "path": "plugins.entries.mnemoss.config.workspace",
    "value": "openclaw" },
  { "path": "plugins.entries.mnemoss.config.timeoutMs",
    "value": 30000 }
]
```

```bash
openclaw config set --batch-file /tmp/mnemoss-config.json
```

Restart the gateway so the slot change takes effect:

```bash
docker restart <your-openclaw-container>
```

### Configuration keys

| Key | Required | Notes |
|---|---|---|
| `plugins.slots.memory` | yes | Set to `"mnemoss"`. Replaces the bundled `memory-core` slot owner. |
| `plugins.entries.mnemoss.config.baseUrl` | yes | URL of the Mnemoss REST server. |
| `plugins.entries.mnemoss.config.apiKey` | when server has auth | Bearer token. Match what `mnemoss-server` reads from `MNEMOSS_API_KEY`. |
| `plugins.entries.mnemoss.config.workspace` | no | Mnemoss workspace id. Defaults to the OpenClaw gateway id at runtime. |
| `plugins.entries.mnemoss.config.timeoutMs` | no | Per-request timeout. Default: `30000`. |

The plugin also reads three env vars as a fallback if config keys are unset:
`MNEMOSS_BASE_URL`, `MNEMOSS_API_KEY`, `MNEMOSS_WORKSPACE`. Plugin config wins.

## Step 4 — Verify

### Plugin loaded and selected

```bash
docker exec <your-openclaw-container> openclaw plugins doctor
# Expected: no warnings about mnemoss
docker exec <your-openclaw-container> openclaw plugins inspect mnemoss
# Expected:
#   Status: loaded
#   Source: /opt/mnemoss-plugin/dist/index.js   (or the npm path)
#   No diagnostics
```

If you see `only memory plugins can register a memory capability`, your plugin
manifest is missing `"kind": "memory"`. Fix the manifest and re-link.

If you see `plugin disabled (memory slot set to "memory-core")`, you forgot to
set `plugins.slots.memory: mnemoss` (or didn't restart the gateway after).

### Recall round-trip

`openclaw memory status` is a CLI command provided by the bundled `memory-core`
plugin; once you switch the slot to `mnemoss`, that CLI command goes away. Verify
recall by driving the plugin runtime directly:

```bash
docker exec <your-openclaw-container> sh -c 'cd /opt/mnemoss-plugin && cat > /tmp/probe.mjs <<EOF
import { createMnemossRuntime } from "/opt/mnemoss-plugin/dist/index.js";
const runtime = createMnemossRuntime({});
const { manager } = await runtime.getMemorySearchManager({
  cfg: {}, agentId: "probe", purpose: "default",
});
console.log("status:", JSON.stringify(manager.status()));
console.log("vector:", await manager.probeVectorAvailability());
console.log("hits:", (await manager.search("hello", { maxResults: 3 })).length);
EOF
node /tmp/probe.mjs'
```

Expected output:

```
status: {"backend":"builtin","provider":"mnemoss","sources":["memory"],...}
vector: true
hits: 0   (or N, depending on how many memories your workspace has)
```

`backend: builtin` is correct: that's OpenClaw's slot label for "in-process
memory engine", not a reference to the bundled SQLite pipeline.

### From outside the container

```bash
curl -sf -H "Authorization: Bearer <MNEMOSS_API_KEY>" \
  http://localhost:8765/workspaces/openclaw/status
# Confirms the workspace exists, embedder is bound, memory_count visible.
```

## Step 5 — Use it

Once the slot is `mnemoss`, every OpenClaw `memory_search` tool call from any
agent on this gateway routes through the adapter. Agent A's memories aren't
visible to agent B unless you observed them as workspace-ambient (`agent_id:
null`). Multi-agent gateways "just work" — Mnemoss enforces the
`agent_id = X OR agent_id IS NULL` rule.

If you want to seed the workspace before hooking up real agents, observe
directly via the Mnemoss REST API:

```bash
curl -sf -X POST -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  -d '{"role":"user","content":"Hello world","agent_id":"my-agent","metadata":{}}' \
  http://localhost:8765/workspaces/openclaw/observe
```

## Troubleshooting

**`memory slot set to "memory-core"` warning.** You set the plugin config but
forgot `plugins.slots.memory: mnemoss`. Set it; restart the gateway.

**`only memory plugins can register a memory capability`.** Your plugin
manifest (`openclaw.plugin.json`) is missing `"kind": "memory"` at the top
level. Add it and re-install. (Fixed in `@mnemoss/openclaw-plugin >= 0.1.1`.)

**`plugin not found: mnemoss (stale config entry ignored)`.** You added the
config block but didn't actually install the plugin yet. Run `plugins install
--link /opt/mnemoss-plugin` (or the npm command).

**`HTTP 422 Unprocessable Entity`** from a search. Either you're calling
`manager.search()` with the wrong arg shape (it's `search(query: string, opts)`,
not `search({ query, ... })`), or the `mnemoss-server` schema changed under the
adapter version you're running — check the adapter's compat range against the
server version in `mnemoss-server /openapi.json`.

**`HTTP 401`.** The plugin sent a bearer token the server doesn't accept. Check
that `plugins.entries.mnemoss.config.apiKey` matches what the server was
launched with (`MNEMOSS_API_KEY`).

**Container can't reach `host.docker.internal`.** On Linux, add
`--add-host=host.docker.internal:host-gateway` to your `docker run`. Or put
both containers on the same `docker network` and use container DNS instead.

**Embedding model download takes forever the first time.** That's the 384-dim
multilingual MiniLM (~470 MB). It happens once per `MNEMOSS_STORAGE_ROOT`. Mount
that as a volume so subsequent restarts are instant.

**The schema-mismatch error on `mnemoss-server` startup.** You re-launched the
server with a different embedder than the one the workspace was created with.
Mnemoss pins the embedder dim at workspace-create time. Either restore the
original embedder, or create a fresh workspace with a different id.

## Uninstall

```bash
docker exec <your-openclaw-container> openclaw plugins uninstall mnemoss
docker exec <your-openclaw-container> openclaw config unset plugins.slots.memory
docker exec <your-openclaw-container> openclaw config unset plugins.entries.mnemoss
docker restart <your-openclaw-container>
```

The Mnemoss server keeps the SQLite workspaces under `MNEMOSS_STORAGE_ROOT`;
delete that directory if you want to wipe memory state too.
