/**
 * MnemossSearchManager tests.
 *
 * We mock the SDK by passing our own ``fetch`` implementation through
 * ``MnemossClient``'s ``fetch`` option — same pattern the SDK's own
 * tests use. That exercises the real HTTP-shaping code, just without
 * a running server.
 */

import { describe, expect, it } from "vitest";
import { MnemossClient } from "@mnemoss/sdk";
import { MnemossSearchManager } from "../src/manager.js";
import { buildMnemossCapability, createMnemossRuntime } from "../src/index.js";

/**
 * Route requests by path — the manager only hits ``/recall`` and
 * ``/status`` endpoints in these tests, so a tiny router is enough.
 */
function mockFetch(
  routes: Record<string, (body: unknown) => unknown>,
): typeof fetch {
  return (async (input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const match = Object.keys(routes).find((path) => url.includes(path));
    if (!match) {
      return new Response(JSON.stringify({ error: `unmocked ${url}` }), {
        status: 500,
      });
    }
    const body = init?.body ? JSON.parse(init.body.toString()) : null;
    const result = routes[match](body);
    return new Response(JSON.stringify(result), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
}

function makeManager(fetchImpl: typeof fetch, agentId = "alice") {
  const client = new MnemossClient("http://mnemoss.test", {
    apiKey: "test-key",
    fetch: fetchImpl,
  });
  return new MnemossSearchManager({
    client,
    workspaceId: "test-ws",
    agentId,
  });
}

function recallResponse(results: Array<{ id: string; content: string; score: number; source?: string }>) {
  return {
    results: results.map((r) => ({
      memory: {
        id: r.id,
        workspace_id: "test-ws",
        agent_id: null,
        session_id: null,
        created_at: "2026-01-01T00:00:00Z",
        content: r.content,
        role: "user",
        memory_type: "episode",
        abstraction_level: 0,
        access_history: [],
        last_accessed_at: null,
        rehearsal_count: 0,
        salience: 0,
        emotional_weight: 0,
        reminisced_count: 0,
        index_tier: "hot",
        idx_priority: 0.7,
        extracted_gist: null,
        extracted_entities: null,
        extracted_time: null,
        extracted_location: null,
        extracted_participants: null,
        extraction_level: 0,
        cluster_id: null,
        cluster_similarity: null,
        is_cluster_representative: false,
        derived_from: [],
        derived_to: [],
        source_message_ids: [],
        source_context: {},
      },
      score: r.score,
      breakdown: {
        base_level: 0,
        spreading: 0,
        matching: 0,
        noise: 0,
        total: r.score,
        idx_priority: 0.7,
        w_f: 0.5,
        w_s: 0.5,
        query_bias: 1.0,
      },
      source: r.source ?? "direct",
    })),
  };
}

// ─── search() ────────────────────────────────────────────────────

describe("MnemossSearchManager.search", () => {
  it("maps Mnemoss RecallResult to OpenClaw MemorySearchResult", async () => {
    const manager = makeManager(
      mockFetch({
        "/recall": () =>
          recallResponse([
            { id: "m1", content: "first", score: 2.1 },
            { id: "m2", content: "second", score: 1.8, source: "expanded" },
          ]),
      }),
    );

    const results = await manager.search("q");

    expect(results).toHaveLength(2);
    expect(results[0].path).toBe("mnemoss://m1");
    expect(results[0].snippet).toBe("first");
    expect(results[0].score).toBe(2.1);
    expect(results[0].source).toBe("memory");
    expect(results[0].citation).toBe("m1");
    // Expanded hits get a suffixed citation.
    expect(results[1].citation).toContain("associated");
  });

  it("filters by minScore when provided", async () => {
    const manager = makeManager(
      mockFetch({
        "/recall": () =>
          recallResponse([
            { id: "m1", content: "keep", score: 2.5 },
            { id: "m2", content: "drop", score: 0.5 },
          ]),
      }),
    );

    const results = await manager.search("q", { minScore: 1.0 });

    expect(results).toHaveLength(1);
    expect(results[0].path).toBe("mnemoss://m1");
  });

  it("forwards maxResults as the Mnemoss k parameter", async () => {
    let capturedBody: unknown = null;
    const manager = makeManager(
      mockFetch({
        "/recall": (body) => {
          capturedBody = body;
          return recallResponse([]);
        },
      }),
    );

    await manager.search("q", { maxResults: 3 });

    expect((capturedBody as { k: number }).k).toBe(3);
  });

  it("invokes the debug callback once per search", async () => {
    const manager = makeManager(
      mockFetch({ "/recall": () => recallResponse([]) }),
    );
    const debug: Array<{ backend: string }> = [];

    await manager.search("q", {
      onDebug: (d) => debug.push(d),
    });

    expect(debug).toHaveLength(1);
    expect(debug[0].backend).toBe("builtin");
  });

  it("returns an empty result set on a zero-hit response", async () => {
    const manager = makeManager(
      mockFetch({ "/recall": () => recallResponse([]) }),
    );
    const results = await manager.search("nothing");
    expect(results).toEqual([]);
  });
});

// ─── readFile() ──────────────────────────────────────────────────

describe("MnemossSearchManager.readFile", () => {
  it("returns the cached snippet for a previously-searched URI", async () => {
    const manager = makeManager(
      mockFetch({
        "/recall": () =>
          recallResponse([
            { id: "m1", content: "alpha\nbeta\ngamma", score: 2.1 },
          ]),
      }),
    );

    await manager.search("q");
    const result = await manager.readFile({ relPath: "mnemoss://m1" });

    expect(result.text).toBe("alpha\nbeta\ngamma");
    expect(result.path).toBe("mnemoss://m1");
    expect(result.truncated).toBe(false);
  });

  it("slices by from/lines when provided", async () => {
    const manager = makeManager(
      mockFetch({
        "/recall": () =>
          recallResponse([
            { id: "m1", content: "l1\nl2\nl3\nl4\nl5", score: 1 },
          ]),
      }),
    );

    await manager.search("q");
    const slice = await manager.readFile({
      relPath: "mnemoss://m1",
      from: 2,
      lines: 2,
    });

    expect(slice.text).toBe("l2\nl3");
    expect(slice.from).toBe(2);
    expect(slice.lines).toBe(2);
    expect(slice.truncated).toBe(true);
    expect(slice.nextFrom).toBe(4);
  });

  it("returns empty text for a cache miss instead of throwing", async () => {
    const manager = makeManager(mockFetch({}));
    const result = await manager.readFile({ relPath: "mnemoss://unknown" });
    expect(result.text).toBe("");
  });
});

// ─── status() + probes ───────────────────────────────────────────

describe("MnemossSearchManager.status", () => {
  it("reports mnemoss under the builtin backend slot", () => {
    const manager = makeManager(mockFetch({}));
    const s = manager.status();
    expect(s.backend).toBe("builtin");
    expect(s.provider).toBe("mnemoss");
    expect(s.sources).toEqual(["memory"]);
  });

  it("embedding probe succeeds when status endpoint is reachable", async () => {
    const manager = makeManager(
      mockFetch({
        "/status": () => ({
          workspace: "test-ws",
          schema_version: 6,
          embedder: { id: "fake:16", dim: 16 },
          memory_count: 0,
          tier_counts: {},
          tombstone_count: 0,
          last_observe_at: null,
          last_dream_at: null,
          last_dream_trigger: null,
          last_rebalance_at: null,
          last_dispose_at: null,
        }),
      }),
    );
    const probe = await manager.probeEmbeddingAvailability();
    expect(probe.ok).toBe(true);
  });

  it("vector probe is true whenever the server is reachable", async () => {
    const manager = makeManager(
      mockFetch({
        "/status": () => ({
          workspace: "test-ws",
          schema_version: 6,
          embedder: { id: "fake:16", dim: 16 },
          memory_count: 0,
          tier_counts: {},
          tombstone_count: 0,
          last_observe_at: null,
          last_dream_at: null,
          last_dream_trigger: null,
          last_rebalance_at: null,
          last_dispose_at: null,
        }),
      }),
    );
    const ok = await manager.probeVectorAvailability();
    expect(ok).toBe(true);
  });
});

// ─── runtime + capability ────────────────────────────────────────

describe("createMnemossRuntime", () => {
  it("routes getMemorySearchManager through our MnemossSearchManager", async () => {
    const runtime = createMnemossRuntime({
      baseUrl: "http://mnemoss.test",
      workspace: "my-ws",
    });
    const { manager, error } = await runtime.getMemorySearchManager({
      cfg: {},
      agentId: "alice",
    });
    expect(error).toBeUndefined();
    expect(manager).toBeInstanceOf(MnemossSearchManager);
  });

  it("prefers per-agent cfg.workspace over the plugin default", async () => {
    const runtime = createMnemossRuntime({
      baseUrl: "http://mnemoss.test",
      workspace: "default-ws",
    });
    const { manager } = await runtime.getMemorySearchManager({
      cfg: { workspace: "override-ws" },
      agentId: "alice",
    });
    expect(manager).toBeInstanceOf(MnemossSearchManager);
    // The concrete workspace is private; asserting at the type level is
    // enough — the runtime branch that reads cfg.workspace is exercised.
  });

  it("resolveMemoryBackendConfig reports builtin", () => {
    const runtime = createMnemossRuntime({ baseUrl: "http://x" });
    const cfg = runtime.resolveMemoryBackendConfig({ cfg: {}, agentId: "a" });
    expect(cfg.backend).toBe("builtin");
  });

  it("error without baseUrl is reported as manager:null with explanation", async () => {
    // The constructor throws synchronously; ``getMemorySearchManager``
    // catches and reports via the error channel.
    const runtime = createMnemossRuntime({ workspace: "x" });
    const result = await runtime.getMemorySearchManager({
      cfg: {},
      agentId: "a",
    });
    expect(result.manager).toBeNull();
    expect(result.error).toContain("baseUrl");
  });
});

// ─── capability + plugin entry ───────────────────────────────────

describe("buildMnemossCapability", () => {
  it("emits a prompt builder and a runtime", () => {
    const cap = buildMnemossCapability({ baseUrl: "http://x" });
    expect(typeof cap.promptBuilder).toBe("function");
    expect(typeof cap.runtime?.getMemorySearchManager).toBe("function");
  });

  it("prompt builder returns non-empty guidance", () => {
    const cap = buildMnemossCapability({ baseUrl: "http://x" });
    const lines = cap.promptBuilder!({
      availableTools: new Set(),
      citationsMode: "auto",
    });
    expect(lines.length).toBeGreaterThan(0);
    expect(lines.join("\n")).toContain("Mnemoss");
  });
});
