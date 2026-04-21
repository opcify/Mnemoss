/**
 * Mnemoss implementation of OpenClaw's MemorySearchManager.
 *
 * Maps the two search contracts onto each other:
 *
 *   Mnemoss recall()  ─┐                 ┌─▶ OpenClaw MemorySearchResult
 *                      │ RecallResult    │   (path, score, snippet, ...)
 *   Mnemoss.sdk client ┴─ adapt(result) ─┘
 *
 * OpenClaw's ``search(query, opts)`` is stateless and read-only. It
 * doesn't distinguish "direct" vs "expanded" hits the way Mnemoss does
 * internally — we preserve that distinction via the ``citation`` field
 * of the result, which OpenClaw surfaces in its UI when present. Result
 * ``path`` uses the ``mnemoss://{id}`` scheme so nothing downstream
 * tries to read these as filesystem paths.
 *
 * ``readFile`` returns the raw memory content (as text), mapping
 * OpenClaw's line-range contract onto Mnemoss's content-oriented model
 * by returning the entire memory body; ``from``/``lines`` are honored
 * when callers slice explicitly. Markdown-style deep-links are out of
 * scope — they don't apply to Mnemoss memories.
 *
 * ``status`` reports ``backend: "builtin"`` because Mnemoss is a
 * ground-up memory backend, not a QMD frontend. ``provider: "mnemoss"``
 * distinguishes it from the bundled SQLite engine.
 */

import type { MnemossClient, WorkspaceHandle } from "@mnemoss/sdk";
import type {
  MemoryEmbeddingProbeResult,
  MemoryProviderStatus,
  MemoryReadResult,
  MemorySearchManager,
  MemorySearchResult,
} from "./openclaw-types.js";

const MNEMOSS_URI_PREFIX = "mnemoss://";

export interface MnemossManagerOptions {
  client: MnemossClient;
  workspaceId: string;
  /**
   * Per-agent scoping. When non-empty, recall runs with
   * ``agent_id = agentId``; Mnemoss returns the agent's own memories
   * plus workspace-ambient ones (``agent_id IS NULL``). Empty string
   * means ambient-only recall for this manager instance.
   */
  agentId: string;
}

export class MnemossSearchManager implements MemorySearchManager {
  private readonly client: MnemossClient;
  private readonly workspace: WorkspaceHandle;
  private readonly agentId: string;
  /**
   * Snippet cache keyed by the ``mnemoss://`` URI returned from
   * ``search``. ``readFile`` reads from this cache rather than fetching
   * by id, because Mnemoss doesn't expose a single-memory endpoint. The
   * cache is bounded to the last 256 search results — anything older
   * rolls off and readFile returns a "not found" result.
   */
  private readonly snippetCache: Map<string, string> = new Map();
  private static readonly SNIPPET_CACHE_MAX = 256;

  constructor(opts: MnemossManagerOptions) {
    this.client = opts.client;
    this.agentId = opts.agentId;
    const ws = this.client.workspace(opts.workspaceId);
    this.workspace = ws;
  }

  async search(
    query: string,
    opts: {
      maxResults?: number;
      minScore?: number;
      sessionKey?: string;
      onDebug?: (debug: { backend: "builtin" | "qmd"; effectiveMode?: string }) => void;
    } = {},
  ): Promise<MemorySearchResult[]> {
    const k = opts.maxResults ?? 10;
    const results = await this.workspace.recall(query, {
      k,
      agentId: this.agentId || undefined,
    });

    opts.onDebug?.({ backend: "builtin", effectiveMode: "activation" });

    const minScore = opts.minScore;
    const adapted: MemorySearchResult[] = [];
    for (const r of results) {
      if (minScore !== undefined && r.score < minScore) continue;
      const uri = `${MNEMOSS_URI_PREFIX}${r.memory.id}`;
      this.rememberSnippet(uri, r.memory.content);
      adapted.push({
        path: uri,
        startLine: 0,
        endLine: 0,
        score: r.score,
        snippet: r.memory.content,
        source: "memory",
        citation: r.source === "expanded" ? `${r.memory.id} (associated)` : r.memory.id,
      });
    }
    return adapted;
  }

  async readFile(params: {
    relPath: string;
    from?: number;
    lines?: number;
  }): Promise<MemoryReadResult> {
    const { relPath, from, lines } = params;
    const text = this.snippetCache.get(relPath);

    if (text === undefined) {
      // Not in cache — the caller is asking for a memory we haven't
      // surfaced via search. Without a server-side get-by-id endpoint we
      // can't fulfill that here. Return a benign empty result so OpenClaw
      // doesn't hard-fail.
      return {
        text: "",
        path: relPath,
        truncated: false,
        from: from ?? 0,
        lines: 0,
      };
    }

    if (from === undefined && lines === undefined) {
      return {
        text,
        path: relPath,
        truncated: false,
        from: 0,
        lines: text.split("\n").length,
      };
    }

    // Line-range slicing. Memories are conceptually one "document" —
    // ``from``/``lines`` slice into the content as if it were a file.
    const allLines = text.split("\n");
    const start = Math.max(0, (from ?? 1) - 1);
    const count = lines ?? Math.max(0, allLines.length - start);
    const slice = allLines.slice(start, start + count);
    const truncated = start + count < allLines.length;
    const nextFrom = truncated ? start + count + 1 : undefined;

    return {
      text: slice.join("\n"),
      path: relPath,
      truncated,
      from: start + 1,
      lines: slice.length,
      ...(nextFrom !== undefined ? { nextFrom } : {}),
    };
  }

  status(): MemoryProviderStatus {
    return {
      backend: "builtin",
      provider: "mnemoss",
      workspaceDir: undefined,
      dbPath: undefined,
      sources: ["memory"],
      custom: {
        engine: "mnemoss",
        activation: "act-r",
        note: "Remote mnemoss-server; files/chunks counts not exposed over REST.",
      },
    };
  }

  async probeEmbeddingAvailability(): Promise<MemoryEmbeddingProbeResult> {
    // Mnemoss always has an embedder bound to its workspace schema —
    // the server would refuse to open the workspace otherwise. We treat
    // reachability of the server as the availability signal.
    try {
      await this.workspace.status();
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  async probeVectorAvailability(): Promise<boolean> {
    // Same rationale: vectors are always on in Mnemoss. If the server's
    // up, vector search works.
    try {
      await this.workspace.status();
      return true;
    } catch {
      return false;
    }
  }

  async close(): Promise<void> {
    // The SDK client is owned by whoever constructed this manager; we
    // don't close it here. OpenClaw will call ``close`` per-manager
    // when deactivating, and ``closeAllMemorySearchManagers`` (defined
    // at the runtime level) handles the shared client.
  }

  // ─── internal ──────────────────────────────────────────────────

  private rememberSnippet(uri: string, content: string): void {
    if (this.snippetCache.size >= MnemossSearchManager.SNIPPET_CACHE_MAX) {
      // Naive FIFO eviction; good enough for a ~256-entry LRU surrogate.
      const firstKey = this.snippetCache.keys().next().value;
      if (firstKey !== undefined) this.snippetCache.delete(firstKey);
    }
    this.snippetCache.set(uri, content);
  }
}
