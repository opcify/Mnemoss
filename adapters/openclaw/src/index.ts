/**
 * OpenClaw memory plugin for Mnemoss.
 *
 * Exports the plugin's default entry (``registerMemoryCapability``) and
 * a standalone ``createMnemossRuntime`` factory for tests and
 * non-OpenClaw embeddings (someone reusing this glue in a different
 * host).
 *
 * The plugin registers itself under the capability id ``"mnemoss"``.
 * Only one memory capability can be active per OpenClaw install; set
 * ``memory.backend: "builtin"`` and install this plugin to route recall
 * through Mnemoss. (The "builtin" backend label is a historical name
 * for the non-QMD path — any in-process capability, including ours,
 * registers under that slot.)
 */

import { MnemossClient } from "@mnemoss/sdk";

import { type MnemossPluginConfig, resolveConfig } from "./config.js";
import { MnemossSearchManager } from "./manager.js";
import type {
  MemoryPluginCapability,
  MemoryPluginRuntime,
  OpenClawPluginApi,
  OpenClawPluginEntry,
} from "./openclaw-types.js";

export { MnemossSearchManager } from "./manager.js";
export type { MnemossPluginConfig } from "./config.js";
export type { MnemossManagerOptions } from "./manager.js";
export * from "./openclaw-types.js";

/**
 * Build the ``MemoryPluginRuntime`` that OpenClaw expects.
 *
 * Exposed separately from the default plugin entry so callers in tests
 * (and alternative plugin hosts that share OpenClaw's memory surface)
 * can construct the runtime without going through the full plugin
 * registration path.
 */
export function createMnemossRuntime(
  explicit?: Partial<MnemossPluginConfig>,
): MemoryPluginRuntime {
  // Lazy client construction: we only build the SDK client when
  // ``getMemorySearchManager`` actually fires, so a disabled plugin
  // doesn't ping the server on boot.
  let client: MnemossClient | null = null;
  let resolvedConfig: MnemossPluginConfig | null = null;

  function getClient(): MnemossClient {
    if (resolvedConfig === null) {
      resolvedConfig = resolveConfig(explicit);
    }
    if (client === null) {
      client = new MnemossClient(resolvedConfig.baseUrl, {
        apiKey: resolvedConfig.apiKey,
        timeoutMs: resolvedConfig.timeoutMs,
      });
    }
    return client;
  }

  function getWorkspaceId(params: { cfg: unknown; agentId: string }): string {
    // Precedence: per-agent config's workspace > plugin-wide default >
    // agentId fallback (so each OpenClaw agent at minimum gets its own
    // workspace if nothing else is set).
    const cfgWorkspace = readString(params.cfg, "workspace");
    if (cfgWorkspace) return cfgWorkspace;
    if (resolvedConfig === null) resolvedConfig = resolveConfig(explicit);
    return resolvedConfig.workspace ?? params.agentId ?? "openclaw";
  }

  return {
    async getMemorySearchManager({ cfg, agentId, purpose }) {
      void purpose; // reserved: "default" vs "status" could toggle cheap mode
      try {
        const workspaceId = getWorkspaceId({ cfg, agentId });
        const manager = new MnemossSearchManager({
          client: getClient(),
          workspaceId,
          agentId,
        });
        return { manager };
      } catch (err) {
        return {
          manager: null,
          error: err instanceof Error ? err.message : String(err),
        };
      }
    },
    resolveMemoryBackendConfig(_params) {
      return { backend: "builtin" };
    },
    async closeAllMemorySearchManagers() {
      // The SDK client has no explicit close today; null it out so a
      // re-activation rebuilds it (picking up any config changes). If
      // ``@mnemoss/sdk`` grows an ``aclose`` method, call it here.
      client = null;
    },
  };
}

export function buildMnemossCapability(
  explicit?: Partial<MnemossPluginConfig>,
): MemoryPluginCapability {
  return {
    promptBuilder: ({ availableTools }) => {
      void availableTools;
      return [
        "# Mnemoss memory",
        "Recall is ranked by ACT-R activation: recency + frequency +",
        "spreading context + semantic/literal match. Use the surfaced",
        "snippets — the host will re-recall on the next turn as the",
        "conversation progresses.",
      ];
    },
    runtime: createMnemossRuntime(explicit),
  };
}

/**
 * Default OpenClaw plugin entry.
 *
 * Mirrors what ``openclaw/plugin-sdk/plugin-entry``'s
 * ``definePluginEntry`` produces — we hand-assemble the shape so the
 * adapter doesn't carry a runtime dep on the private
 * ``@openclaw/plugin-sdk`` package. OpenClaw's plugin loader looks for
 * ``id``, ``name``, ``description``, and ``register(api)``; that's
 * everything it needs.
 */
const pluginEntry: OpenClawPluginEntry = {
  id: "mnemoss",
  name: "Mnemoss",
  description:
    "ACT-R memory system for OpenClaw — recall ranked by activation, with spreading and dreaming.",
  register(api: OpenClawPluginApi) {
    api.registerMemoryCapability("mnemoss", buildMnemossCapability());
  },
};

export default pluginEntry;

// ─── helpers ───────────────────────────────────────────────────

function readString(source: unknown, key: string): string | undefined {
  if (source === null || typeof source !== "object") return undefined;
  const v = (source as Record<string, unknown>)[key];
  return typeof v === "string" && v.length > 0 ? v : undefined;
}
