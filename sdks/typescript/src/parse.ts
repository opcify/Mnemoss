/**
 * Wire-format (snake_case) → library-type (camelCase) converters.
 *
 * The server emits Pydantic models with snake_case field names; the SDK
 * normalizes to camelCase so callers use idiomatic TypeScript.
 */

import type {
  ActivationBreakdown,
  DisposalStats,
  DreamReport,
  IndexTier,
  Memory,
  MemoryType,
  PhaseName,
  PhaseOutcome,
  RebalanceStats,
  RecallResult,
  Tombstone,
  TriggerType,
  WorkspaceStatus,
} from "./types.js";

type Raw = Record<string, any>;

const parseDate = (v: string): Date => new Date(v);
const parseDateOpt = (v: string | null | undefined): Date | null =>
  v ? new Date(v) : null;

export function parseMemory(dto: Raw): Memory {
  return {
    id: dto.id,
    workspaceId: dto.workspace_id,
    agentId: dto.agent_id ?? null,
    sessionId: dto.session_id ?? null,
    createdAt: parseDate(dto.created_at),
    content: dto.content,
    role: dto.role ?? null,
    memoryType: dto.memory_type as MemoryType,
    abstractionLevel: dto.abstraction_level,
    accessHistory: (dto.access_history ?? []).map(parseDate),
    lastAccessedAt: parseDateOpt(dto.last_accessed_at),
    rehearsalCount: dto.rehearsal_count ?? 0,
    salience: dto.salience ?? 0,
    emotionalWeight: dto.emotional_weight ?? 0,
    reminiscedCount: dto.reminisced_count ?? 0,
    indexTier: (dto.index_tier ?? "hot") as IndexTier,
    idxPriority: dto.idx_priority ?? 0.5,
    extractedGist: dto.extracted_gist ?? null,
    extractedEntities: dto.extracted_entities ?? null,
    extractedTime: parseDateOpt(dto.extracted_time),
    extractedLocation: dto.extracted_location ?? null,
    extractedParticipants: dto.extracted_participants ?? null,
    extractionLevel: dto.extraction_level ?? 0,
    clusterId: dto.cluster_id ?? null,
    clusterSimilarity: dto.cluster_similarity ?? null,
    isClusterRepresentative: dto.is_cluster_representative ?? false,
    derivedFrom: [...(dto.derived_from ?? [])],
    derivedTo: [...(dto.derived_to ?? [])],
    sourceMessageIds: [...(dto.source_message_ids ?? [])],
    sourceContext: { ...(dto.source_context ?? {}) },
  };
}

export function parseBreakdown(dto: Raw): ActivationBreakdown {
  return {
    baseLevel: dto.base_level,
    spreading: dto.spreading,
    matching: dto.matching,
    noise: dto.noise,
    total: dto.total,
    idxPriority: dto.idx_priority,
    wF: dto.w_f,
    wS: dto.w_s,
    queryBias: dto.query_bias,
  };
}

export function parseRecallResult(dto: Raw): RecallResult {
  return {
    memory: parseMemory(dto.memory),
    score: dto.score,
    breakdown: parseBreakdown(dto.breakdown),
    source: (dto.source ?? "direct") as "direct" | "expanded",
  };
}

export function parseTombstone(dto: Raw): Tombstone {
  return {
    originalId: dto.original_id,
    workspaceId: dto.workspace_id,
    agentId: dto.agent_id ?? null,
    droppedAt: parseDate(dto.dropped_at),
    reason: dto.reason,
    gistSnapshot: dto.gist_snapshot,
    bAtDrop: dto.b_at_drop,
    sourceMessageIds: [...(dto.source_message_ids ?? [])],
  };
}

export function parsePhaseOutcome(dto: Raw): PhaseOutcome {
  return {
    phase: dto.phase as PhaseName,
    status: dto.status,
    details: { ...(dto.details ?? {}) },
  };
}

export function parseDreamReport(dto: Raw): DreamReport {
  return {
    trigger: dto.trigger as TriggerType,
    startedAt: parseDate(dto.started_at),
    finishedAt: parseDate(dto.finished_at),
    durationSeconds: dto.duration_seconds,
    agentId: dto.agent_id ?? null,
    outcomes: (dto.outcomes ?? []).map(parsePhaseOutcome),
    diaryPath: dto.diary_path ?? null,
  };
}

export function parseRebalanceStats(dto: Raw): RebalanceStats {
  return {
    scanned: dto.scanned,
    migrated: dto.migrated,
    tierBefore: dto.tier_before as Record<IndexTier, number>,
    tierAfter: dto.tier_after as Record<IndexTier, number>,
  };
}

export function parseDisposalStats(dto: Raw): DisposalStats {
  return {
    scanned: dto.scanned,
    disposed: dto.disposed,
    activationDead: dto.activation_dead,
    redundant: dto.redundant,
    protected: dto.protected,
    disposedIds: [...(dto.disposed_ids ?? [])],
  };
}

export function parseWorkspaceStatus(dto: Raw): WorkspaceStatus {
  return {
    workspace: dto.workspace,
    schemaVersion: dto.schema_version,
    embedder: { id: dto.embedder.id, dim: dto.embedder.dim },
    memoryCount: dto.memory_count,
    tierCounts: dto.tier_counts,
    tombstoneCount: dto.tombstone_count,
    lastObserveAt: parseDateOpt(dto.last_observe_at),
    lastDreamAt: parseDateOpt(dto.last_dream_at),
    lastDreamTrigger: dto.last_dream_trigger ?? null,
    lastRebalanceAt: parseDateOpt(dto.last_rebalance_at),
    lastDisposeAt: parseDateOpt(dto.last_dispose_at),
  };
}
