"""DTO → library-type converters.

Every SDK call returns library types (``Memory``, ``Tombstone``,
``RecallResult``, …) so framework plugins work with the same objects
they'd see from the in-process ``Mnemoss`` client. The server does not
ship raw embeddings, so ``Memory.content_embedding`` is always ``None``
on the wire.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from dateutil import parser as dateutil_parser

from mnemoss.core.types import IndexTier, Memory, MemoryType, Tombstone
from mnemoss.dream.dispose import DisposalStats
from mnemoss.dream.types import DreamReport, PhaseName, PhaseOutcome, TriggerType
from mnemoss.formula.activation import ActivationBreakdown
from mnemoss.index import RebalanceStats
from mnemoss.recall import RecallResult


def parse_dt(value: str) -> datetime:
    # dateutil accepts every form FastAPI emits (``+00:00`` and ``Z``)
    # plus historical variants. Safer than ``fromisoformat`` on 3.10.
    return dateutil_parser.isoparse(value)


def parse_dt_opt(value: str | None) -> datetime | None:
    return parse_dt(value) if value is not None else None


def parse_memory(dto: dict[str, Any]) -> Memory:
    return Memory(
        id=dto["id"],
        workspace_id=dto["workspace_id"],
        agent_id=dto.get("agent_id"),
        session_id=dto.get("session_id"),
        created_at=parse_dt(dto["created_at"]),
        content=dto["content"],
        content_embedding=None,
        role=dto.get("role"),
        memory_type=MemoryType(dto["memory_type"]),
        abstraction_level=dto["abstraction_level"],
        access_history=[parse_dt(t) for t in dto.get("access_history", [])],
        last_accessed_at=parse_dt_opt(dto.get("last_accessed_at")),
        rehearsal_count=dto.get("rehearsal_count", 0),
        salience=dto.get("salience", 0.0),
        emotional_weight=dto.get("emotional_weight", 0.0),
        reminisced_count=dto.get("reminisced_count", 0),
        index_tier=IndexTier(dto.get("index_tier", "hot")),
        idx_priority=dto.get("idx_priority", 0.5),
        extracted_gist=dto.get("extracted_gist"),
        extracted_entities=dto.get("extracted_entities"),
        extracted_time=parse_dt_opt(dto.get("extracted_time")),
        extracted_location=dto.get("extracted_location"),
        extracted_participants=dto.get("extracted_participants"),
        extraction_level=dto.get("extraction_level", 0),
        cluster_id=dto.get("cluster_id"),
        cluster_similarity=dto.get("cluster_similarity"),
        is_cluster_representative=dto.get("is_cluster_representative", False),
        derived_from=list(dto.get("derived_from", [])),
        derived_to=list(dto.get("derived_to", [])),
        source_message_ids=list(dto.get("source_message_ids", [])),
        source_context=dict(dto.get("source_context", {})),
    )


def parse_tombstone(dto: dict[str, Any]) -> Tombstone:
    return Tombstone(
        original_id=dto["original_id"],
        workspace_id=dto["workspace_id"],
        agent_id=dto.get("agent_id"),
        dropped_at=parse_dt(dto["dropped_at"]),
        reason=dto["reason"],
        gist_snapshot=dto["gist_snapshot"],
        b_at_drop=dto["b_at_drop"],
        source_message_ids=list(dto.get("source_message_ids", [])),
    )


def parse_breakdown(dto: dict[str, Any]) -> ActivationBreakdown:
    return ActivationBreakdown(
        base_level=dto["base_level"],
        spreading=dto["spreading"],
        matching=dto["matching"],
        noise=dto["noise"],
        total=dto["total"],
        idx_priority=dto["idx_priority"],
        w_f=dto["w_f"],
        w_s=dto["w_s"],
        query_bias=dto["query_bias"],
    )


def parse_recall_result(dto: dict[str, Any]) -> RecallResult:
    return RecallResult(
        memory=parse_memory(dto["memory"]),
        score=dto["score"],
        breakdown=parse_breakdown(dto["breakdown"]),
    )


def parse_phase_outcome(dto: dict[str, Any]) -> PhaseOutcome:
    return PhaseOutcome(
        phase=PhaseName(dto["phase"]),
        status=dto["status"],
        details=dict(dto.get("details", {})),
    )


def parse_dream_report(dto: dict[str, Any]) -> DreamReport:
    diary = dto.get("diary_path")
    return DreamReport(
        trigger=TriggerType(dto["trigger"]),
        started_at=parse_dt(dto["started_at"]),
        finished_at=parse_dt(dto["finished_at"]),
        agent_id=dto.get("agent_id"),
        outcomes=[parse_phase_outcome(o) for o in dto.get("outcomes", [])],
        diary_path=Path(diary) if diary is not None else None,
    )


def parse_rebalance_stats(dto: dict[str, Any]) -> RebalanceStats:
    return RebalanceStats(
        scanned=dto["scanned"],
        migrated=dto["migrated"],
        tier_before={IndexTier(k): v for k, v in dto["tier_before"].items()},
        tier_after={IndexTier(k): v for k, v in dto["tier_after"].items()},
    )


def parse_disposal_stats(dto: dict[str, Any]) -> DisposalStats:
    return DisposalStats(
        scanned=dto["scanned"],
        disposed=dto["disposed"],
        activation_dead=dto["activation_dead"],
        redundant=dto["redundant"],
        protected=dto["protected"],
        disposed_ids=list(dto.get("disposed_ids", [])),
    )
