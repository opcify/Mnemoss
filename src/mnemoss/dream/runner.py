"""Dream dispatcher: trigger → phase sequence → DreamReport.

Each trigger picks a subset of the available phases per §6.3. The
runner records a ``PhaseOutcome`` for every phase it attempts, so the
report always tells the caller what happened even when a phase was
skipped (e.g. no LLM configured, empty replay set).

Six phases: Replay → Cluster → Consolidate → Relations → Rebalance →
Dispose. Consolidate is a single LLM call per cluster that produces
the summary, per-member refinements, and any intra-cluster patterns —
it replaces the former three-phase Extract / Refine / Generalize trio.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from mnemoss.core.config import FormulaParams, TierCapacityParams
from mnemoss.core.types import Memory
from mnemoss.dream.cluster import ClusterAssignment, cluster_embeddings, group_by_cluster
from mnemoss.dream.consolidate import (
    ConsolidationResult,
    consolidate_cluster,
)
from mnemoss.dream.cost import CostLedger, CostLimits
from mnemoss.dream.dispose import dispose_pass
from mnemoss.dream.relations import (
    write_derived_from_edges,
    write_similar_to_edges,
)
from mnemoss.dream.replay import select_replay_candidates
from mnemoss.dream.types import (
    DreamReport,
    PhaseName,
    PhaseOutcome,
    TriggerType,
)
from mnemoss.encoder import Embedder
from mnemoss.index import rebalance as _rebalance
from mnemoss.llm.client import LLMClient
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


PHASES_BY_TRIGGER: dict[TriggerType, list[PhaseName]] = {
    # The former Extract / Refine / Generalize trio is now a single
    # Consolidate phase. Surprise and cognitive_load intentionally skip
    # REPLAY — they run on memories already surfaced by the host
    # framework (Stage 6+ will add an explicit `memories=` parameter).
    TriggerType.IDLE: [
        PhaseName.REPLAY,
        PhaseName.CLUSTER,
        PhaseName.CONSOLIDATE,
        PhaseName.RELATIONS,
    ],
    TriggerType.SESSION_END: [
        PhaseName.REPLAY,
        PhaseName.CLUSTER,
        PhaseName.CONSOLIDATE,
        PhaseName.RELATIONS,
    ],
    TriggerType.SURPRISE: [
        PhaseName.CONSOLIDATE,
        PhaseName.RELATIONS,
    ],
    TriggerType.COGNITIVE_LOAD: [
        PhaseName.CONSOLIDATE,
    ],
    TriggerType.NIGHTLY: [
        PhaseName.REPLAY,
        PhaseName.CLUSTER,
        PhaseName.CONSOLIDATE,
        PhaseName.RELATIONS,
        PhaseName.REBALANCE,
        PhaseName.DISPOSE,
    ],
}


@dataclass
class _DreamState:
    """In-memory state that flows between phases within one dream run."""

    replay_set: list[Memory] = field(default_factory=list)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    cluster_assignments: dict[str, ClusterAssignment] = field(default_factory=dict)
    # Every Memory emitted by Consolidate (summaries + patterns). P5
    # Relations writes derived_from edges from these back to their sources.
    consolidated: list[Memory] = field(default_factory=list)


class DreamRunner:
    """Runs one dream cycle for a trigger."""

    def __init__(
        self,
        store: SQLiteBackend,
        params: FormulaParams,
        *,
        tier_capacity: TierCapacityParams | None = None,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        replay_limit: int = 100,
        replay_min_base_level: float | None = None,
        cluster_min_size: int = 3,
        cost_limits: CostLimits | None = None,
        cost_ledger: CostLedger | None = None,
    ) -> None:
        self._store = store
        self._params = params
        self._tier_capacity = tier_capacity if tier_capacity is not None else TierCapacityParams()
        self._llm = llm
        self._embedder = embedder
        self._replay_limit = replay_limit
        self._replay_min_base_level = replay_min_base_level
        self._cluster_min_size = cluster_min_size
        # Cost governance on LLM calls in Consolidate. When both are
        # ``None`` the runner makes unlimited calls (historical
        # behaviour); when provided, the ledger enforces the limits
        # and persists counts across runs.
        self._cost_limits = cost_limits or CostLimits()
        self._cost_ledger = cost_ledger

    async def run(
        self,
        trigger: TriggerType,
        *,
        agent_id: str | None = None,
        now: datetime | None = None,
    ) -> DreamReport:
        t0 = now if now is not None else datetime.now(UTC)
        report = DreamReport(
            trigger=trigger,
            started_at=t0,
            finished_at=t0,
            agent_id=agent_id,
        )
        state = _DreamState()

        for phase in PHASES_BY_TRIGGER.get(trigger, []):
            outcome = await self._run_phase(phase, state, agent_id, t0)
            report.outcomes.append(outcome)

        report.finished_at = datetime.now(UTC)
        return report

    async def _run_phase(
        self,
        phase: PhaseName,
        state: _DreamState,
        agent_id: str | None,
        now: datetime,
    ) -> PhaseOutcome:
        try:
            if phase is PhaseName.REPLAY:
                return await self._phase_replay(state, agent_id, now)
            if phase is PhaseName.CLUSTER:
                return await self._phase_cluster(state)
            if phase is PhaseName.CONSOLIDATE:
                return await self._phase_consolidate(state, now)
            if phase is PhaseName.RELATIONS:
                return await self._phase_relations(state)
            if phase is PhaseName.REBALANCE:
                return await self._phase_rebalance(now)
            if phase is PhaseName.DISPOSE:
                return await self._phase_dispose(state, now)
            raise RuntimeError(f"unknown phase {phase}")
        except Exception as e:
            # Never let a phase crash the whole dream. Record the
            # failure and let downstream phases run on whatever partial
            # state exists — most phases already tolerate empty inputs
            # (see the ``state.replay_set`` / ``state.cluster_assignments``
            # guards below).
            return PhaseOutcome(
                phase=phase,
                status="error",
                error=f"{type(e).__name__}: {e}",
                details={"error_type": type(e).__name__},
            )

    # ─── P1 Replay ─────────────────────────────────────────────────

    async def _phase_replay(
        self,
        state: _DreamState,
        agent_id: str | None,
        now: datetime,
    ) -> PhaseOutcome:
        memories = await select_replay_candidates(
            self._store,
            agent_id,
            self._params,
            now=now,
            limit=self._replay_limit,
            min_base_level=self._replay_min_base_level,
        )
        state.replay_set = memories
        return PhaseOutcome(
            phase=PhaseName.REPLAY,
            status="ok",
            details={
                "selected": len(memories),
                "memories": memories,
                "memory_ids": [m.id for m in memories],
            },
        )

    # ─── P2 Cluster ────────────────────────────────────────────────

    async def _phase_cluster(self, state: _DreamState) -> PhaseOutcome:
        if not state.replay_set:
            # Empty replay set isn't an error — nightly runs on fresh
            # workspaces hit this legitimately. Skip with a reason so
            # the report makes it obvious.
            return PhaseOutcome(
                phase=PhaseName.CLUSTER,
                status="skipped",
                skip_reason="empty replay set",
                details={"clusters": 0},
            )

        ids = [m.id for m in state.replay_set]
        embeddings = await self._store.get_embeddings(ids)
        state.embeddings = embeddings

        assignments = cluster_embeddings(embeddings, min_cluster_size=self._cluster_min_size)
        state.cluster_assignments = assignments

        for mid, a in assignments.items():
            await self._store.update_cluster_assignment(
                mid, a.cluster_id, a.similarity, a.is_representative
            )

        clusters = group_by_cluster(assignments)
        noise = sum(1 for a in assignments.values() if a.cluster_id is None)
        return PhaseOutcome(
            phase=PhaseName.CLUSTER,
            status="ok",
            details={
                "clusters": len(clusters),
                "noise": noise,
                "total": len(assignments),
            },
        )

    # ─── P3 Consolidate (merged Extract + Refine + Generalize) ─────

    async def _phase_consolidate(self, state: _DreamState, now: datetime) -> PhaseOutcome:
        if self._llm is None:
            return PhaseOutcome(
                phase=PhaseName.CONSOLIDATE,
                status="skipped",
                skip_reason="no llm configured",
            )
        if self._embedder is None:
            return PhaseOutcome(
                phase=PhaseName.CONSOLIDATE,
                status="skipped",
                skip_reason="no embedder configured",
            )
        if not state.replay_set and not state.cluster_assignments:
            # Upstream REPLAY/CLUSTER failed or produced nothing — no
            # point asking the LLM about empty state.
            return PhaseOutcome(
                phase=PhaseName.CONSOLIDATE,
                status="skipped",
                skip_reason="no replay set or clusters from upstream",
            )

        # Without clustering (surprise/cognitive_load triggers), fall back
        # to treating the whole replay set as one synthetic cluster so
        # the LLM still gets one pass over what's active.
        clusters = self._clusters_for_consolidation(state)

        summaries: list[Memory] = []
        patterns: list[Memory] = []
        refined_ids: list[str] = []
        llm_failures = 0
        llm_calls_made = 0
        budget_skips: list[str] = []

        for cluster_members in clusters:
            if len(cluster_members) < 2:
                continue  # Singletons have nothing to consolidate.

            # Budget gate — check before the call so we don't pay for
            # clusters we won't persist.
            if self._cost_ledger is not None:
                reason = self._cost_ledger.check_budget(
                    self._cost_limits,
                    run_calls=llm_calls_made,
                    now=now,
                )
                if reason is not None:
                    budget_skips.append(reason)
                    break

            result = await consolidate_cluster(cluster_members, self._llm, self._params, now=now)
            llm_calls_made += 1
            if self._cost_ledger is not None:
                self._cost_ledger.record_call(now=now)

            if result.is_empty:
                llm_failures += 1
                continue

            # (A) Summary — write new memory + derived edges, embed it.
            if result.summary is not None:
                await self._persist_derived(result.summary)
                summaries.append(result.summary)

            # (B) Refinements — bump extraction_level=2 on members.
            for ref in result.refinements:
                member = cluster_members[ref.member_index]
                fields = ref.fields
                member.extracted_gist = fields.gist
                member.extracted_entities = fields.entities
                member.extracted_time = fields.time
                member.extracted_location = fields.location
                member.extracted_participants = fields.participants
                member.extraction_level = fields.level
                await self._store.update_extraction(
                    member.id,
                    gist=fields.gist,
                    entities=fields.entities,
                    time=fields.time,
                    location=fields.location,
                    participants=fields.participants,
                    level=fields.level,
                )
                refined_ids.append(member.id)

            # (C) Patterns — intra-cluster PATTERN memories.
            for pattern in result.patterns:
                await self._persist_derived(pattern)
                patterns.append(pattern)

        # Relations phase edges off summaries + patterns together.
        state.consolidated = summaries + patterns

        return PhaseOutcome(
            phase=PhaseName.CONSOLIDATE,
            status="ok",
            details={
                "clusters_processed": sum(1 for c in clusters if len(c) >= 2),
                "summaries": len(summaries),
                "patterns": len(patterns),
                "refined": len(refined_ids),
                "llm_failures": llm_failures,
                "llm_calls_made": llm_calls_made,
                "budget_skips": budget_skips,
                "summary_ids": [m.id for m in summaries],
                "pattern_ids": [m.id for m in patterns],
                "refined_ids": refined_ids,
                "cross_agent_promotions": sum(
                    1 for m in (summaries + patterns) if m.agent_id is None
                ),
            },
        )

    def _clusters_for_consolidation(self, state: _DreamState) -> list[list[Memory]]:
        """Resolve the flat clusters list the phase iterates over."""

        if state.cluster_assignments:
            groups = group_by_cluster(state.cluster_assignments)
            by_id = {m.id: m for m in state.replay_set}
            return [[by_id[mid] for mid in members if mid in by_id] for members in groups.values()]
        if state.replay_set:
            return [state.replay_set]
        return []

    async def _persist_derived(self, memory: Memory) -> None:
        """Embed + write a derived memory, link its derived_from edges."""

        assert self._embedder is not None  # guarded by phase entry
        embedding = await asyncio.to_thread(self._embedder.embed, [memory.content])
        await self._store.write_memory(memory, embedding[0])
        await self._store.link_derived(memory.derived_from, memory.id)

    # ─── P5 Relations ──────────────────────────────────────────────

    async def _phase_relations(self, state: _DreamState) -> PhaseOutcome:
        similar_edges = await write_similar_to_edges(self._store, state.cluster_assignments)
        derived_edges = await write_derived_from_edges(self._store, state.consolidated)
        return PhaseOutcome(
            phase=PhaseName.RELATIONS,
            status="ok",
            details={
                "similar_to_edges": similar_edges,
                "derived_from_edges": derived_edges,
                "total_edges": similar_edges + derived_edges,
            },
        )

    # ─── P7 Rebalance ──────────────────────────────────────────────

    async def _phase_rebalance(self, now: datetime) -> PhaseOutcome:
        stats = await _rebalance(
            self._store, self._params, self._tier_capacity, now=now
        )
        return PhaseOutcome(
            phase=PhaseName.REBALANCE,
            status="ok",
            details={
                "scanned": stats.scanned,
                "migrated": stats.migrated,
                "tier_after": {tier.value: count for tier, count in stats.tier_after.items()},
            },
        )

    # ─── P8 Dispose ────────────────────────────────────────────────

    async def _phase_dispose(self, state: _DreamState, now: datetime) -> PhaseOutcome:
        # Dispose scans the replay set when P1 ran, else the full table.
        candidates = state.replay_set if state.replay_set else None
        stats = await dispose_pass(self._store, self._params, now=now, candidates=candidates)
        return PhaseOutcome(
            phase=PhaseName.DISPOSE,
            status="ok",
            details={
                "scanned": stats.scanned,
                "disposed": stats.disposed,
                "activation_dead": stats.activation_dead,
                "redundant": stats.redundant,
                "protected": stats.protected,
                "disposed_ids": stats.disposed_ids,
            },
        )


# Public re-export so tests can import without reaching into consolidate.py
__all__ = [
    "DreamRunner",
    "PHASES_BY_TRIGGER",
    "ConsolidationResult",
]
