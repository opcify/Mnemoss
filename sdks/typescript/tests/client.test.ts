/**
 * Unit tests for the TypeScript SDK.
 *
 * We inject a mock ``fetch`` so requests never leave the process. Every
 * test asserts either (a) that the request was shaped correctly, or
 * (b) that the response was parsed into camelCase types with dates as
 * ``Date`` objects and enums as strings.
 */

import { describe, it, expect, vi } from "vitest";

import { MnemossClient, MnemossHTTPError } from "../src/client.js";

// ─── helpers ─────────────────────────────────────────────────────

function memoryDTO(overrides: Record<string, unknown> = {}) {
  return {
    id: "mem_1",
    workspace_id: "ws",
    agent_id: null,
    session_id: "default",
    created_at: "2026-04-21T12:00:00+00:00",
    content: "hello",
    role: "user",
    memory_type: "episode",
    abstraction_level: 0.0,
    access_history: [],
    last_accessed_at: null,
    rehearsal_count: 0,
    salience: 0.0,
    emotional_weight: 0.0,
    reminisced_count: 0,
    index_tier: "hot",
    idx_priority: 0.5,
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
    ...overrides,
  };
}

function breakdownDTO() {
  return {
    base_level: 1.0,
    spreading: 0.0,
    matching: 2.0,
    noise: 0.0,
    total: 3.0,
    idx_priority: 0.73,
    w_f: 0.5,
    w_s: 0.5,
    query_bias: 1.0,
  };
}

function jsonResponse(body: unknown, init: ResponseInit = { status: 200 }): Response {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
  });
}

function mockFetch(
  handler: (url: string, init?: RequestInit) => Response,
): typeof fetch {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    return handler(url, init);
  }) as unknown as typeof fetch;
}

// ─── tests ───────────────────────────────────────────────────────

describe("MnemossClient", () => {
  it("sends observe to the correct URL with correct body", async () => {
    const seen: { url?: string; body?: any } = {};
    const fetchImpl = mockFetch((url, init) => {
      seen.url = url;
      seen.body = init?.body ? JSON.parse(init.body as string) : undefined;
      return jsonResponse({ memory_id: "mem_1" });
    });

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    const mid = await client.workspace("ws").observe("user", "hi");

    expect(mid).toBe("mem_1");
    expect(seen.url).toBe("http://test/workspaces/ws/observe");
    expect(seen.body).toMatchObject({ role: "user", content: "hi" });
  });

  it("includes agent_id query param when provided", async () => {
    let capturedUrl: string | undefined;
    const fetchImpl = mockFetch((url) => {
      capturedUrl = url;
      return jsonResponse({ memory_id: "m" });
    });

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    await client.workspace("ws").observe("user", "hi", { agentId: "alice" });

    expect(capturedUrl).toContain("agent_id=alice");
  });

  it("omits agent_id param when undefined", async () => {
    let capturedUrl: string | undefined;
    const fetchImpl = mockFetch((url) => {
      capturedUrl = url;
      return jsonResponse({ memory_id: "m" });
    });

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    await client.workspace("ws").observe("user", "hi");

    expect(capturedUrl).not.toContain("agent_id");
  });

  it("sends Authorization header when apiKey is set", async () => {
    let seenHeaders: HeadersInit | undefined;
    const fetchImpl = mockFetch((_url, init) => {
      seenHeaders = init?.headers;
      return jsonResponse({ ok: true });
    });

    const client = new MnemossClient("http://test", {
      apiKey: "s3cret",
      fetch: fetchImpl,
    });
    await client.health();

    const headers = new Headers(seenHeaders);
    expect(headers.get("Authorization")).toBe("Bearer s3cret");
  });

  it("parses recall response into camelCase RecallResult with Date objects", async () => {
    const fetchImpl = mockFetch(() =>
      jsonResponse({
        results: [
          {
            memory: memoryDTO({ content: "alpha" }),
            score: 4.2,
            breakdown: breakdownDTO(),
          },
        ],
      }),
    );

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    const results = await client.workspace("ws").recall("alpha", { k: 1 });

    expect(results).toHaveLength(1);
    const r = results[0]!;
    expect(r.memory.content).toBe("alpha");
    expect(r.memory.memoryType).toBe("episode");
    expect(r.memory.indexTier).toBe("hot");
    expect(r.memory.createdAt).toBeInstanceOf(Date);
    expect(r.memory.createdAt.toISOString()).toBe("2026-04-21T12:00:00.000Z");
    expect(r.breakdown.total).toBe(3.0);
    expect(r.breakdown.idxPriority).toBe(0.73);
  });

  it("agent handle binds agentId on every call", async () => {
    const urls: string[] = [];
    const fetchImpl = mockFetch((url) => {
      urls.push(url);
      if (url.includes("/observe")) return jsonResponse({ memory_id: "m" });
      if (url.includes("/recall")) return jsonResponse({ results: [] });
      if (url.includes("/pin")) return jsonResponse({ ok: true });
      return jsonResponse({ ok: true });
    });

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    const alice = client.workspace("ws").forAgent("alice");
    await alice.observe("user", "x");
    await alice.recall("y", { k: 3 });
    await alice.pin("mem_1");

    for (const u of urls) {
      expect(u).toContain("agent_id=alice");
    }
  });

  it("parses explain response into ActivationBreakdown", async () => {
    const fetchImpl = mockFetch(() =>
      jsonResponse({ breakdown: breakdownDTO() }),
    );

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    const b = await client.workspace("ws").explainRecall("q", "mem_1");

    expect(b.total).toBe(3.0);
    expect(b.idxPriority).toBe(0.73);
    expect(b.wF).toBe(0.5);
  });

  it("parses dream report phases and diary path", async () => {
    const fetchImpl = mockFetch(() =>
      jsonResponse({
        trigger: "idle",
        started_at: "2026-04-21T12:00:00+00:00",
        finished_at: "2026-04-21T12:00:05+00:00",
        duration_seconds: 5.0,
        agent_id: null,
        outcomes: [
          { phase: "replay", status: "ok", details: { selected: 3 } },
        ],
        diary_path: "/tmp/diary.md",
      }),
    );

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    const report = await client.workspace("ws").dream({ trigger: "idle" });

    expect(report.trigger).toBe("idle");
    expect(report.outcomes).toHaveLength(1);
    expect(report.outcomes[0]!.phase).toBe("replay");
    expect(report.outcomes[0]!.details["selected"]).toBe(3);
    expect(report.diaryPath).toBe("/tmp/diary.md");
    expect(report.startedAt).toBeInstanceOf(Date);
  });

  it("parses tombstones into camelCase objects", async () => {
    const fetchImpl = mockFetch(() =>
      jsonResponse({
        tombstones: [
          {
            original_id: "mem_gone",
            workspace_id: "ws",
            agent_id: null,
            dropped_at: "2026-04-21T12:00:00+00:00",
            reason: "redundant",
            gist_snapshot: "was here",
            b_at_drop: -2.5,
            source_message_ids: ["msg_1"],
          },
        ],
      }),
    );

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    const tombs = await client.workspace("ws").tombstones();

    expect(tombs).toHaveLength(1);
    expect(tombs[0]!.reason).toBe("redundant");
    expect(tombs[0]!.sourceMessageIds).toEqual(["msg_1"]);
    expect(tombs[0]!.droppedAt).toBeInstanceOf(Date);
  });

  it("parses rebalance stats with IndexTier keys preserved", async () => {
    const fetchImpl = mockFetch(() =>
      jsonResponse({
        scanned: 10,
        migrated: 3,
        tier_before: { hot: 10, warm: 0, cold: 0, deep: 0 },
        tier_after: { hot: 7, warm: 2, cold: 1, deep: 0 },
      }),
    );

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    const stats = await client.workspace("ws").rebalance();

    expect(stats.scanned).toBe(10);
    expect(stats.tierAfter["warm"]).toBe(2);
  });

  it("parses dispose stats with camelCase fields", async () => {
    const fetchImpl = mockFetch(() =>
      jsonResponse({
        scanned: 5,
        disposed: 1,
        activation_dead: 1,
        redundant: 0,
        protected: 4,
        disposed_ids: ["mem_x"],
      }),
    );

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    const stats = await client.workspace("ws").dispose();

    expect(stats.disposed).toBe(1);
    expect(stats.activationDead).toBe(1);
    expect(stats.disposedIds).toEqual(["mem_x"]);
  });

  it("returns markdown string from export", async () => {
    const fetchImpl = mockFetch(() =>
      jsonResponse({ markdown: "## Facts\n- x" }),
    );

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    const md = await client.workspace("ws").exportMarkdown();

    expect(md).toBe("## Facts\n- x");
  });

  it("returns flushed count", async () => {
    const fetchImpl = mockFetch(() => jsonResponse({ flushed: 2 }));

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    const n = await client.workspace("ws").flushSession();

    expect(n).toBe(2);
  });

  it("throws MnemossHTTPError with status and body on non-2xx", async () => {
    const fetchImpl = mockFetch(() =>
      jsonResponse({ detail: "bad key" }, { status: 401 }),
    );

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    await expect(
      client.workspace("ws").observe("user", "hi"),
    ).rejects.toBeInstanceOf(MnemossHTTPError);
  });

  it("trims trailing slashes on the base URL", async () => {
    let capturedUrl: string | undefined;
    const fetchImpl = mockFetch((url) => {
      capturedUrl = url;
      return jsonResponse({ ok: true });
    });

    const client = new MnemossClient("http://test/", { fetch: fetchImpl });
    await client.health();

    expect(capturedUrl).toBe("http://test/health");
  });

  it("URL-encodes the workspace id", async () => {
    let capturedUrl: string | undefined;
    const fetchImpl = mockFetch((url) => {
      capturedUrl = url;
      return jsonResponse({ memory_id: "m" });
    });

    const client = new MnemossClient("http://test", { fetch: fetchImpl });
    await client.workspace("with space").observe("user", "hi");

    expect(capturedUrl).toContain("/workspaces/with%20space/observe");
  });
});
