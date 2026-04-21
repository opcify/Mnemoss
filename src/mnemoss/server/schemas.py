"""Pydantic request/response DTOs for the HTTP surface.

Every response is a plain JSON-serializable Pydantic model. ``Memory``
rows drop ``content_embedding`` (raw ndarray, not useful on the wire);
every other field survives. ``ActivationBreakdown`` serializes as a
sub-model so ``explain_recall`` remains informative over HTTP.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from mnemoss.core.types import Memory, Tombstone
from mnemoss.dream.dispose import DisposalStats
from mnemoss.dream.types import DreamReport, PhaseOutcome
from mnemoss.formula.activation import ActivationBreakdown
from mnemoss.index import RebalanceStats
from mnemoss.recall import RecallResult

# ─── memory ──────────────────────────────────────────────────────


class MemoryDTO(BaseModel):
    id: str
    workspace_id: str
    agent_id: str | None
    session_id: str | None
    created_at: datetime
    content: str
    role: str | None
    memory_type: str
    abstraction_level: float
    access_history: list[datetime] = Field(default_factory=list)
    last_accessed_at: datetime | None = None
    rehearsal_count: int = 0
    salience: float = 0.0
    emotional_weight: float = 0.0
    reminisced_count: int = 0
    index_tier: str
    idx_priority: float
    extracted_gist: str | None = None
    extracted_entities: list[str] | None = None
    extracted_time: datetime | None = None
    extracted_location: str | None = None
    extracted_participants: list[str] | None = None
    extraction_level: int = 0
    cluster_id: str | None = None
    cluster_similarity: float | None = None
    is_cluster_representative: bool = False
    derived_from: list[str] = Field(default_factory=list)
    derived_to: list[str] = Field(default_factory=list)
    source_message_ids: list[str] = Field(default_factory=list)
    source_context: dict[str, Any] = Field(default_factory=dict)


def memory_to_dto(memory: Memory) -> MemoryDTO:
    return MemoryDTO(
        id=memory.id,
        workspace_id=memory.workspace_id,
        agent_id=memory.agent_id,
        session_id=memory.session_id,
        created_at=memory.created_at,
        content=memory.content,
        role=memory.role,
        memory_type=memory.memory_type.value,
        abstraction_level=memory.abstraction_level,
        access_history=list(memory.access_history),
        last_accessed_at=memory.last_accessed_at,
        rehearsal_count=memory.rehearsal_count,
        salience=memory.salience,
        emotional_weight=memory.emotional_weight,
        reminisced_count=memory.reminisced_count,
        index_tier=memory.index_tier.value,
        idx_priority=memory.idx_priority,
        extracted_gist=memory.extracted_gist,
        extracted_entities=memory.extracted_entities,
        extracted_time=memory.extracted_time,
        extracted_location=memory.extracted_location,
        extracted_participants=memory.extracted_participants,
        extraction_level=memory.extraction_level,
        cluster_id=memory.cluster_id,
        cluster_similarity=memory.cluster_similarity,
        is_cluster_representative=memory.is_cluster_representative,
        derived_from=list(memory.derived_from),
        derived_to=list(memory.derived_to),
        source_message_ids=list(memory.source_message_ids),
        source_context=dict(memory.source_context),
    )


# ─── observe ─────────────────────────────────────────────────────


class ObserveRequest(BaseModel):
    role: str
    content: str
    session_id: str | None = None
    turn_id: str | None = None
    parent_id: str | None = None
    metadata: dict[str, Any] | None = None


class ObserveResponse(BaseModel):
    # ``None`` when the role was filtered out by ``encoded_roles``.
    memory_id: str | None


# ─── recall ──────────────────────────────────────────────────────


class RecallRequest(BaseModel):
    query: str
    k: int = 5
    include_deep: bool = False
    auto_expand: bool = True


class ActivationBreakdownDTO(BaseModel):
    base_level: float
    spreading: float
    matching: float
    noise: float
    total: float
    idx_priority: float
    w_f: float
    w_s: float
    query_bias: float


def breakdown_to_dto(b: ActivationBreakdown) -> ActivationBreakdownDTO:
    return ActivationBreakdownDTO(
        base_level=b.base_level,
        spreading=b.spreading,
        matching=b.matching,
        noise=b.noise,
        total=b.total,
        idx_priority=b.idx_priority,
        w_f=b.w_f,
        w_s=b.w_s,
        query_bias=b.query_bias,
    )


class RecallResultDTO(BaseModel):
    memory: MemoryDTO
    score: float
    breakdown: ActivationBreakdownDTO
    source: str = "direct"


def recall_result_to_dto(r: RecallResult) -> RecallResultDTO:
    return RecallResultDTO(
        memory=memory_to_dto(r.memory),
        score=r.score,
        breakdown=breakdown_to_dto(r.breakdown),
        source=r.source,
    )


class RecallResponse(BaseModel):
    results: list[RecallResultDTO]


class ExplainRequest(BaseModel):
    query: str
    memory_id: str


class ExplainResponse(BaseModel):
    breakdown: ActivationBreakdownDTO


class ExpandRequest(BaseModel):
    memory_id: str
    query: str | None = None
    hops: int = 1
    k: int = 5


# ─── pin ─────────────────────────────────────────────────────────


class PinRequest(BaseModel):
    memory_id: str


class OkResponse(BaseModel):
    ok: bool = True


# ─── tombstone ───────────────────────────────────────────────────


class TombstoneDTO(BaseModel):
    original_id: str
    workspace_id: str
    agent_id: str | None
    dropped_at: datetime
    reason: str
    gist_snapshot: str
    b_at_drop: float
    source_message_ids: list[str] = Field(default_factory=list)


def tombstone_to_dto(t: Tombstone) -> TombstoneDTO:
    return TombstoneDTO(
        original_id=t.original_id,
        workspace_id=t.workspace_id,
        agent_id=t.agent_id,
        dropped_at=t.dropped_at,
        reason=t.reason,
        gist_snapshot=t.gist_snapshot,
        b_at_drop=t.b_at_drop,
        source_message_ids=list(t.source_message_ids),
    )


class TombstonesResponse(BaseModel):
    tombstones: list[TombstoneDTO]


# ─── dream ───────────────────────────────────────────────────────


class DreamRequest(BaseModel):
    trigger: str = "session_end"


class PhaseOutcomeDTO(BaseModel):
    phase: str
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


def phase_outcome_to_dto(o: PhaseOutcome) -> PhaseOutcomeDTO:
    # ``details`` may carry non-JSON objects (e.g. Memory instances from
    # P1 Replay). Convert the well-known keys into plain primitives so
    # the response serializes cleanly regardless of phase.
    safe_details: dict[str, Any] = {}
    for key, value in o.details.items():
        safe_details[key] = _json_safe(value)
    return PhaseOutcomeDTO(
        phase=o.phase.value,
        status=o.status,
        details=safe_details,
    )


def _json_safe(value: Any) -> Any:
    from mnemoss.core.types import Memory as _Memory

    if isinstance(value, _Memory):
        return memory_to_dto(value).model_dump(mode="json")
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class DreamResponse(BaseModel):
    trigger: str
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    agent_id: str | None
    outcomes: list[PhaseOutcomeDTO]
    diary_path: str | None


def dream_report_to_dto(r: DreamReport) -> DreamResponse:
    return DreamResponse(
        trigger=r.trigger.value,
        started_at=r.started_at,
        finished_at=r.finished_at,
        duration_seconds=r.duration_seconds(),
        agent_id=r.agent_id,
        outcomes=[phase_outcome_to_dto(o) for o in r.outcomes],
        diary_path=str(r.diary_path) if r.diary_path is not None else None,
    )


# ─── dispose / rebalance / export / flush ────────────────────────


class DisposeResponse(BaseModel):
    scanned: int
    disposed: int
    activation_dead: int
    redundant: int
    protected: int
    disposed_ids: list[str] = Field(default_factory=list)


def disposal_stats_to_dto(s: DisposalStats) -> DisposeResponse:
    return DisposeResponse(
        scanned=s.scanned,
        disposed=s.disposed,
        activation_dead=s.activation_dead,
        redundant=s.redundant,
        protected=s.protected,
        disposed_ids=list(s.disposed_ids),
    )


class RebalanceResponse(BaseModel):
    scanned: int
    migrated: int
    tier_before: dict[str, int]
    tier_after: dict[str, int]


def rebalance_stats_to_dto(s: RebalanceStats) -> RebalanceResponse:
    return RebalanceResponse(
        scanned=s.scanned,
        migrated=s.migrated,
        tier_before={k.value: v for k, v in s.tier_before.items()},
        tier_after={k.value: v for k, v in s.tier_after.items()},
    )


class TierCountsResponse(BaseModel):
    tiers: dict[str, int]


class ExportMarkdownRequest(BaseModel):
    min_idx_priority: float = 0.5


class ExportMarkdownResponse(BaseModel):
    markdown: str


class FlushSessionRequest(BaseModel):
    session_id: str | None = None


class FlushSessionResponse(BaseModel):
    flushed: int


# ─── status ──────────────────────────────────────────────────────


class EmbedderInfo(BaseModel):
    id: str
    dim: int


class StatusResponse(BaseModel):
    workspace: str
    schema_version: int
    embedder: EmbedderInfo
    memory_count: int
    tier_counts: dict[str, int]
    tombstone_count: int
    last_observe_at: datetime | None = None
    last_dream_at: datetime | None = None
    last_dream_trigger: str | None = None
    last_rebalance_at: datetime | None = None
    last_dispose_at: datetime | None = None
