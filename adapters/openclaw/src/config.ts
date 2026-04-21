/**
 * Resolves plugin configuration.
 *
 * Precedence (highest first):
 * 1. Explicit options passed to ``createMnemossManager(...)``.
 * 2. ``MNEMOSS_BASE_URL``, ``MNEMOSS_API_KEY``, ``MNEMOSS_WORKSPACE``
 *    environment variables.
 *
 * OpenClaw also supplies its own per-agent config via the capability
 * registration path (``cfg`` and ``agentId`` land in
 * ``getMemorySearchManager``), but we don't assume a particular schema
 * — whatever the plugin host passes is checked for ``mnemoss`` config
 * at the capability level.
 */

export interface MnemossPluginConfig {
  /** URL of a running ``mnemoss-server``. Required. */
  baseUrl: string;
  /** Bearer token. Optional if the server is unauthenticated. */
  apiKey?: string;
  /**
   * Workspace id. Defaults to the OpenClaw gateway id when the plugin
   * is registered; per-agent scoping uses OpenClaw's own ``agentId``
   * which becomes Mnemoss's ``agent_id`` on every call.
   */
  workspace?: string;
  /** Per-request timeout in milliseconds. Default 30000. */
  timeoutMs?: number;
}

export function resolveConfig(
  explicit?: Partial<MnemossPluginConfig>,
): MnemossPluginConfig {
  const env = globalThis.process?.env ?? {};

  const baseUrl = explicit?.baseUrl ?? env.MNEMOSS_BASE_URL ?? "";
  if (!baseUrl) {
    throw new Error(
      "Mnemoss plugin requires baseUrl (set via plugin config or MNEMOSS_BASE_URL).",
    );
  }

  return {
    baseUrl,
    apiKey: explicit?.apiKey ?? env.MNEMOSS_API_KEY,
    workspace: explicit?.workspace ?? env.MNEMOSS_WORKSPACE,
    timeoutMs: explicit?.timeoutMs ?? 30000,
  };
}
