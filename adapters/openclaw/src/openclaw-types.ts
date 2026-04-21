/**
 * Structural-typing shims for the OpenClaw plugin host.
 *
 * OpenClaw's ``@openclaw/plugin-sdk`` and ``@openclaw/memory-host-sdk``
 * packages are marked ``private`` inside the openclaw monorepo, so we
 * can't npm-install them from this adapter during development. The
 * shapes the plugin actually uses are small and stable; we redeclare
 * them here so the package builds and typechecks standalone.
 *
 * When the plugin is loaded inside a running openclaw install, these
 * types agree structurally with the real ones — duck-typing at the
 * TypeScript level. Any drift in openclaw's interfaces will show up
 * as test failures at load time, which is the right place to catch it.
 *
 * Mirror source: ``openclaw/src/memory-host-sdk/host/types.ts`` and
 * ``openclaw/src/plugins/memory-state.ts`` at the commit this plugin
 * was written against.
 */

export type MemorySource = "memory" | "sessions";

export interface MemorySearchResult {
  path: string;
  startLine: number;
  endLine: number;
  score: number;
  snippet: string;
  source: MemorySource;
  citation?: string;
}

export interface MemoryReadResult {
  text: string;
  path: string;
  truncated?: boolean;
  from?: number;
  lines?: number;
  nextFrom?: number;
}

export interface MemoryEmbeddingProbeResult {
  ok: boolean;
  error?: string;
}

export type MemoryBackend = "builtin" | "qmd";

export interface MemoryProviderStatus {
  backend: MemoryBackend;
  provider: string;
  model?: string;
  files?: number;
  chunks?: number;
  workspaceDir?: string;
  dbPath?: string;
  sources?: MemorySource[];
  custom?: Record<string, unknown>;
}

export interface MemorySearchManager {
  search(
    query: string,
    opts?: {
      maxResults?: number;
      minScore?: number;
      sessionKey?: string;
      qmdSearchModeOverride?: "query" | "search" | "vsearch";
      onDebug?: (debug: { backend: MemoryBackend; effectiveMode?: string }) => void;
    },
  ): Promise<MemorySearchResult[]>;
  readFile(params: {
    relPath: string;
    from?: number;
    lines?: number;
  }): Promise<MemoryReadResult>;
  status(): MemoryProviderStatus;
  sync?(params?: {
    reason?: string;
    force?: boolean;
  }): Promise<void>;
  probeEmbeddingAvailability(): Promise<MemoryEmbeddingProbeResult>;
  probeVectorAvailability(): Promise<boolean>;
  close?(): Promise<void>;
}

export interface MemoryPluginRuntime {
  getMemorySearchManager(params: {
    cfg: unknown;
    agentId: string;
    purpose?: "default" | "status";
  }): Promise<{ manager: MemorySearchManager | null; error?: string }>;
  resolveMemoryBackendConfig(params: {
    cfg: unknown;
    agentId: string;
  }): { backend: MemoryBackend };
  closeAllMemorySearchManagers?(): Promise<void>;
}

export interface MemoryPluginCapability {
  promptBuilder?: (params: {
    availableTools: Set<string>;
    citationsMode?: "auto" | "on" | "off";
  }) => string[];
  runtime?: MemoryPluginRuntime;
}

/**
 * Slimmed OpenClaw plugin API surface we touch.
 *
 * Real type: ``OpenClawPluginApi`` in ``src/plugins/types.ts``. We only
 * declare the single registration method this plugin uses; the rest of
 * the API exists at runtime but isn't our concern here.
 */
export interface OpenClawPluginApi {
  registerMemoryCapability(pluginId: string, capability: MemoryPluginCapability): void;
}

export interface OpenClawPluginEntry {
  id: string;
  name: string;
  description: string;
  register: (api: OpenClawPluginApi) => void;
}
