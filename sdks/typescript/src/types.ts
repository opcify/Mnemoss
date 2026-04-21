/**
 * Type mirrors for the Mnemoss REST API.
 *
 * Fields are ``camelCase`` on the TypeScript side even though the
 * wire format is ``snake_case`` — the client converts both directions.
 */

export type MemoryType = "episode" | "fact" | "entity" | "pattern";

export type IndexTier = "hot" | "warm" | "cold" | "deep";

export type TriggerType =
  | "idle"
  | "session_end"
  | "surprise"
  | "cognitive_load"
  | "nightly";

export type PhaseName =
  | "replay"
  | "cluster"
  | "extract"
  | "refine"
  | "relations"
  | "generalize"
  | "rebalance"
  | "dispose";

export interface Memory {
  id: string;
  workspaceId: string;
  agentId: string | null;
  sessionId: string | null;
  createdAt: Date;
  content: string;
  role: string | null;
  memoryType: MemoryType;
  abstractionLevel: number;
  accessHistory: Date[];
  lastAccessedAt: Date | null;
  rehearsalCount: number;
  salience: number;
  emotionalWeight: number;
  reminiscedCount: number;
  indexTier: IndexTier;
  idxPriority: number;
  extractedGist: string | null;
  extractedEntities: string[] | null;
  extractedTime: Date | null;
  extractedLocation: string | null;
  extractedParticipants: string[] | null;
  extractionLevel: number;
  clusterId: string | null;
  clusterSimilarity: number | null;
  isClusterRepresentative: boolean;
  derivedFrom: string[];
  derivedTo: string[];
  sourceMessageIds: string[];
  sourceContext: Record<string, unknown>;
}

export interface ActivationBreakdown {
  baseLevel: number;
  spreading: number;
  matching: number;
  noise: number;
  total: number;
  idxPriority: number;
  wF: number;
  wS: number;
  queryBias: number;
}

export interface RecallResult {
  memory: Memory;
  score: number;
  breakdown: ActivationBreakdown;
  source: "direct" | "expanded";
}

export interface Tombstone {
  originalId: string;
  workspaceId: string;
  agentId: string | null;
  droppedAt: Date;
  reason: string;
  gistSnapshot: string;
  bAtDrop: number;
  sourceMessageIds: string[];
}

export interface PhaseOutcome {
  phase: PhaseName;
  status: string;
  details: Record<string, unknown>;
}

export interface DreamReport {
  trigger: TriggerType;
  startedAt: Date;
  finishedAt: Date;
  durationSeconds: number;
  agentId: string | null;
  outcomes: PhaseOutcome[];
  diaryPath: string | null;
}

export interface RebalanceStats {
  scanned: number;
  migrated: number;
  tierBefore: Record<IndexTier, number>;
  tierAfter: Record<IndexTier, number>;
}

export interface DisposalStats {
  scanned: number;
  disposed: number;
  activationDead: number;
  redundant: number;
  protected: number;
  disposedIds: string[];
}

export interface EmbedderInfo {
  id: string;
  dim: number;
}

export interface WorkspaceStatus {
  workspace: string;
  schemaVersion: number;
  embedder: EmbedderInfo;
  memoryCount: number;
  tierCounts: Record<string, number>;
  tombstoneCount: number;
  lastObserveAt: Date | null;
  lastDreamAt: Date | null;
  lastDreamTrigger: string | null;
  lastRebalanceAt: Date | null;
  lastDisposeAt: Date | null;
}

// ─── request options ─────────────────────────────────────────────

export interface ObserveOptions {
  agentId?: string;
  sessionId?: string;
  turnId?: string;
  parentId?: string;
  metadata?: Record<string, unknown>;
}

export interface RecallOptions {
  k?: number;
  agentId?: string;
  includeDeep?: boolean;
  autoExpand?: boolean;
}

export interface ExpandOptions {
  agentId?: string;
  query?: string;
  hops?: number;
  k?: number;
}

export interface DreamOptions {
  trigger?: TriggerType;
  agentId?: string;
}

export interface TombstonesOptions {
  agentId?: string;
  limit?: number;
}

export interface ExportOptions {
  agentId?: string;
  minIdxPriority?: number;
}

export interface FlushOptions {
  agentId?: string;
  sessionId?: string;
}

export interface ClientOptions {
  /** Bearer token; omit for local-dev servers with no auth. */
  apiKey?: string;
  /** Default 30 seconds. */
  timeoutMs?: number;
  /** Fetch implementation override (tests). */
  fetch?: typeof fetch;
}
