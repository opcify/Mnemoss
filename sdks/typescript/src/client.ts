/**
 * MnemossClient — HTTP client for the Mnemoss REST API.
 *
 * Mirrors the Python SDK method-for-method so framework plugins can
 * target either runtime uniformly. Uses built-in ``fetch`` (Node 18+,
 * all modern browsers); no runtime dependencies.
 */

import {
  parseBreakdown,
  parseDisposalStats,
  parseDreamReport,
  parseRebalanceStats,
  parseRecallResult,
  parseTombstone,
  parseWorkspaceStatus,
} from "./parse.js";
import type {
  ActivationBreakdown,
  ClientOptions,
  DisposalStats,
  DreamOptions,
  DreamReport,
  ExportOptions,
  FlushOptions,
  IndexTier,
  ObserveOptions,
  RebalanceStats,
  RecallOptions,
  RecallResult,
  Tombstone,
  TombstonesOptions,
  WorkspaceStatus,
} from "./types.js";

// ─── error type ──────────────────────────────────────────────────

export class MnemossHTTPError extends Error {
  constructor(
    public readonly status: number,
    public readonly statusText: string,
    public readonly body: unknown,
  ) {
    super(`HTTP ${status} ${statusText}`);
    this.name = "MnemossHTTPError";
  }
}

// ─── client ──────────────────────────────────────────────────────

export class MnemossClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(baseUrl: string, options: ClientOptions = {}) {
    // Trim trailing slash so ``/workspaces/...`` concatenation always
    // produces ``http://host/workspaces/...`` not ``http://host//...``.
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = { Accept: "application/json" };
    if (options.apiKey) {
      this.headers["Authorization"] = `Bearer ${options.apiKey}`;
    }
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.fetchImpl = options.fetch ?? fetch;
  }

  workspace(workspaceId: string): WorkspaceHandle {
    return new WorkspaceHandle(this, workspaceId);
  }

  async health(): Promise<boolean> {
    const body = await this.get<{ ok: boolean }>("/health");
    return !!body.ok;
  }

  // ─── internal HTTP ──────────────────────────────────────────────

  async post<T>(
    path: string,
    body?: Record<string, unknown>,
    queryParams?: Record<string, string | number | undefined>,
  ): Promise<T> {
    return this.request<T>("POST", path, body, queryParams);
  }

  async get<T>(
    path: string,
    queryParams?: Record<string, string | number | undefined>,
  ): Promise<T> {
    return this.request<T>("GET", path, undefined, queryParams);
  }

  private async request<T>(
    method: "GET" | "POST",
    path: string,
    body?: Record<string, unknown>,
    queryParams?: Record<string, string | number | undefined>,
  ): Promise<T> {
    const url = this.buildUrl(path, queryParams);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const headers: Record<string, string> = { ...this.headers };
      if (body !== undefined) headers["Content-Type"] = "application/json";
      const resp = await this.fetchImpl(url, {
        method,
        headers,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
      if (!resp.ok) {
        let errorBody: unknown = null;
        try {
          errorBody = await resp.json();
        } catch {
          errorBody = await resp.text().catch(() => null);
        }
        throw new MnemossHTTPError(resp.status, resp.statusText, errorBody);
      }
      if (resp.status === 204) return undefined as T;
      return (await resp.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  private buildUrl(
    path: string,
    queryParams?: Record<string, string | number | undefined>,
  ): string {
    const qs = buildQuery(queryParams);
    return `${this.baseUrl}${path}${qs}`;
  }
}

function buildQuery(
  params?: Record<string, string | number | undefined>,
): string {
  if (!params) return "";
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue;
    sp.append(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

// ─── workspace handle ───────────────────────────────────────────

export class WorkspaceHandle {
  constructor(
    private readonly client: MnemossClient,
    public readonly workspaceId: string,
  ) {}

  forAgent(agentId: string): AgentHandle {
    return new AgentHandle(this, agentId);
  }

  // ─── core ─────────────────────────────────────────────────────

  async observe(
    role: string,
    content: string,
    options: ObserveOptions = {},
  ): Promise<string | null> {
    const body: Record<string, unknown> = {
      role,
      content,
      session_id: options.sessionId ?? null,
      turn_id: options.turnId ?? null,
      parent_id: options.parentId ?? null,
      metadata: options.metadata ?? null,
    };
    const resp = await this.client.post<{ memory_id: string | null }>(
      this.path("observe"),
      body,
      { agent_id: options.agentId },
    );
    return resp.memory_id;
  }

  async recall(
    query: string,
    options: RecallOptions = {},
  ): Promise<RecallResult[]> {
    const body = {
      query,
      k: options.k ?? 5,
      include_deep: options.includeDeep ?? false,
    };
    const resp = await this.client.post<{ results: any[] }>(
      this.path("recall"),
      body,
      { agent_id: options.agentId },
    );
    return resp.results.map(parseRecallResult);
  }

  async pin(memoryId: string, options: { agentId?: string } = {}): Promise<void> {
    await this.client.post<{ ok: boolean }>(
      this.path("pin"),
      { memory_id: memoryId },
      { agent_id: options.agentId },
    );
  }

  async explainRecall(
    query: string,
    memoryId: string,
    options: { agentId?: string } = {},
  ): Promise<ActivationBreakdown> {
    const resp = await this.client.post<{ breakdown: any }>(
      this.path("explain"),
      { query, memory_id: memoryId },
      { agent_id: options.agentId },
    );
    return parseBreakdown(resp.breakdown);
  }

  // ─── dream / housekeeping ─────────────────────────────────────

  async dream(options: DreamOptions = {}): Promise<DreamReport> {
    const resp = await this.client.post<any>(
      this.path("dream"),
      { trigger: options.trigger ?? "session_end" },
      { agent_id: options.agentId },
    );
    return parseDreamReport(resp);
  }

  async rebalance(): Promise<RebalanceStats> {
    const resp = await this.client.post<any>(this.path("rebalance"));
    return parseRebalanceStats(resp);
  }

  async dispose(): Promise<DisposalStats> {
    const resp = await this.client.post<any>(this.path("dispose"));
    return parseDisposalStats(resp);
  }

  async tombstones(options: TombstonesOptions = {}): Promise<Tombstone[]> {
    const resp = await this.client.get<{ tombstones: any[] }>(
      this.path("tombstones"),
      { agent_id: options.agentId, limit: options.limit ?? 100 },
    );
    return resp.tombstones.map(parseTombstone);
  }

  async tierCounts(): Promise<Record<IndexTier, number>> {
    const resp = await this.client.get<{ tiers: Record<IndexTier, number> }>(
      this.path("tiers"),
    );
    return resp.tiers;
  }

  async exportMarkdown(options: ExportOptions = {}): Promise<string> {
    const resp = await this.client.post<{ markdown: string }>(
      this.path("export"),
      { min_idx_priority: options.minIdxPriority ?? 0.5 },
      { agent_id: options.agentId },
    );
    return resp.markdown;
  }

  async flushSession(options: FlushOptions = {}): Promise<number> {
    const resp = await this.client.post<{ flushed: number }>(
      this.path("flush"),
      { session_id: options.sessionId ?? null },
      { agent_id: options.agentId },
    );
    return resp.flushed;
  }

  async status(): Promise<WorkspaceStatus> {
    const resp = await this.client.get<any>(this.path("status"));
    return parseWorkspaceStatus(resp);
  }

  private path(suffix: string): string {
    return `/workspaces/${encodeURIComponent(this.workspaceId)}/${suffix}`;
  }
}

// ─── agent handle ───────────────────────────────────────────────

export class AgentHandle {
  constructor(
    private readonly workspace: WorkspaceHandle,
    public readonly agentId: string,
  ) {}

  async observe(
    role: string,
    content: string,
    options: Omit<ObserveOptions, "agentId"> = {},
  ): Promise<string | null> {
    return this.workspace.observe(role, content, {
      ...options,
      agentId: this.agentId,
    });
  }

  async recall(
    query: string,
    options: Omit<RecallOptions, "agentId"> = {},
  ): Promise<RecallResult[]> {
    return this.workspace.recall(query, { ...options, agentId: this.agentId });
  }

  async pin(memoryId: string): Promise<void> {
    await this.workspace.pin(memoryId, { agentId: this.agentId });
  }

  async explainRecall(
    query: string,
    memoryId: string,
  ): Promise<ActivationBreakdown> {
    return this.workspace.explainRecall(query, memoryId, {
      agentId: this.agentId,
    });
  }

  async exportMarkdown(
    options: Omit<ExportOptions, "agentId"> = {},
  ): Promise<string> {
    return this.workspace.exportMarkdown({
      ...options,
      agentId: this.agentId,
    });
  }
}
