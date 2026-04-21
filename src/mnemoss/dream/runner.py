"""Dream dispatcher: trigger → phase sequence → DreamReport.

Each trigger picks a subset of the available phases per §6.3. The
runner records a ``PhaseOutcome`` for every phase it attempts, so the
report always tells the caller what happened even when a phase was
skipped (e.g. no LLM configured, empty replay set).

Stage 4 ships P1 Replay, P2 Cluster, P3 Extract, P5 Relations. P4
Refine is deferred (see CLAUDE.md D10). If no LLM is configured the
LLM-dependent phases record ``status="skipped"`` and the remaining
phases still run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory
from mnemoss.dream.cluster import ClusterAssignment, cluster_embeddings, group_by_cluster
from mnemoss.dream.extract import extract_from_cluster
from mnemoss.dream.refine import refine_memory_fields
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
from mnemoss.llm.client import LLMClient
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


PHASES_BY_TRIGGER: dict[TriggerType, list[PhaseName]] = {
    # Per §6.3. P6 Generalize / P7 Rebalance / P8 Dispose + the
    # surprise / cognitive_load / nightly triggers are Stage 5 additions.
    TriggerType.IDLE: [
        PhaseName.REPLAY,
        PhaseName.CLUSTER,
        PhaseName.EXTRACT,
        PhaseName.RELATIONS,
    ],
    TriggerType.SESSION_END: [
        PhaseName.REPLAY,
        PhaseName.CLUSTER,
        PhaseName.EXTRACT,
        PhaseName.REFINE,
        PhaseName.RELATIONS,
    ],
    TriggerType.TASK_COMPLETION: [
        PhaseName.REPLAY,
        PhaseName.EXTRACT,
        PhaseName.RELATIONS,
    ],
}


@dataclass
class _DreamState:
    """In-memory state that flows between phases within one dream run."""

    replay_set: list[Memory] = field(default_factory=list)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    cluster_assignments: dict[str, ClusterAssignment] = field(default_factory=dict)
    extracted: list[Memory] = field(default_factory=list)


class DreamRunner:
    """Runs one dream cycle for a trigger."""

    def __init__(
        self,
        store: SQLiteBackend,
        params: FormulaParams,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        replay_limit: int = 100,
        replay_min_base_level: float | None = None,
        cluster_min_size: int = 3,
        refine_batch_size: int = 25,
    ) -> None:
        self._store = store
        self._params = params
        self._llm = llm
        self._embedder = embedder
        self._replay_limit = replay_limit
        self._replay_min_base_level = replay_min_base_level
        self._cluster_min_size = cluster_min_size
        self._refine_batch_size = refine_batch_size

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
        if phase is PhaseName.REPLAY:
            return await self._phase_replay(state, agent_id, now)
        if phase is PhaseName.CLUSTER:
            return await self._phase_cluster(state)
        if phase is PhaseName.EXTRACT:
            return await self._phase_extract(state, now)
        if phase is PhaseName.REFINE:
            return await self._phase_refine(state)
        if phase is PhaseName.RELATIONS:
            return await self._phase_relations(state)
        raise RuntimeError(f"unknown phase {phase}")

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
            return PhaseOutcome(
                phase=PhaseName.CLUSTER,
                status="ok",
                details={"clusters": 0, "reason": "empty replay"},
            )

        ids = [m.id for m in state.replay_set]
        embeddings = await self._store.get_embeddings(ids)
        state.embeddings = embeddings

        assignments = cluster_embeddings(
            embeddings, min_cluster_size=self._cluster_min_size
        )
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

    # ─── P3 Extract ────────────────────────────────────────────────

    async def _phase_extract(
        self, state: _DreamState, now: datetime
    ) -> PhaseOutcome:
        if self._llm is None:
            return PhaseOutcome(
                phase=PhaseName.EXTRACT,
                status="skipped",
                details={"reason": "no llm configured"},
            )
        if self._embedder is None:
            return PhaseOutcome(
                phase=PhaseName.EXTRACT,
                status="skipped",
                details={"reason": "no embedder configured"},
            )

        # Without clustering (e.g. task_completion trigger), treat the full
        # replay set as one synthetic cluster so the LLM still gets to
        # summarize the recent window.
        if state.cluster_assignments:
            clusters_by_id = group_by_cluster(state.cluster_assignments)
            by_id = {m.id: m for m in state.replay_set}
            clusters = [
                [by_id[mid] for mid in members if mid in by_id]
                for members in clusters_by_id.values()
            ]
        elif state.replay_set:
            clusters = [state.replay_set]
        else:
            clusters = []

        extracted: list[Memory] = []
        llm_failures = 0
        for cluster_members in clusters:
            if len(cluster_members) < 2:
                continue  # Singleton clusters aren't worth extracting.
            new_mem = await extract_from_cluster(
                cluster_members, self._llm, self._params, now=now
            )
            if new_mem is None:
                llm_failures += 1
                continue

            embedding = await asyncio.to_thread(
                self._embedder.embed, [new_mem.content]
            )
            await self._store.write_memory(new_mem, embedding[0])
            await self._store.link_derived(new_mem.derived_from, new_mem.id)
            extracted.append(new_mem)

        state.extracted = extracted
        return PhaseOutcome(
            phase=PhaseName.EXTRACT,
            status="ok",
            details={
                "extracted": len(extracted),
                "ids": [m.id for m in extracted],
                "llm_failures": llm_failures,
                "cross_agent_promotions": sum(
                    1 for m in extracted if m.agent_id is None
                ),
            },
        )

    # ─── P4 Refine ─────────────────────────────────────────────────

    async def _phase_refine(self, state: _DreamState) -> PhaseOutcome:
        if self._llm is None:
            return PhaseOutcome(
                phase=PhaseName.REFINE,
                status="skipped",
                details={"reason": "no llm configured"},
            )

        # Replay set is already B_i-sorted desc, so taking the first N keeps
        # the highest-priority memories.
        candidates = [m for m in state.replay_set if m.extraction_level < 2]
        batch = candidates[: self._refine_batch_size]

        if not batch:
            return PhaseOutcome(
                phase=PhaseName.REFINE,
                status="ok",
                details={
                    "refined": 0,
                    "candidates": 0,
                    "reason": "no memories below level=2",
                },
            )

        refined = 0
        failures = 0
        for memory in batch:
            fields = await refine_memory_fields(memory, self._llm)
            if fields is None:
                failures += 1
                continue
            memory.extracted_gist = fields.gist
            memory.extracted_entities = fields.entities
            memory.extracted_time = fields.time
            memory.extracted_location = fields.location
            memory.extracted_participants = fields.participants
            memory.extraction_level = fields.level
            await self._store.update_extraction(
                memory.id,
                gist=fields.gist,
                entities=fields.entities,
                time=fields.time,
                location=fields.location,
                participants=fields.participants,
                level=fields.level,
            )
            refined += 1

        return PhaseOutcome(
            phase=PhaseName.REFINE,
            status="ok",
            details={
                "refined": refined,
                "failures": failures,
                "candidates": len(candidates),
                "batch_limit": self._refine_batch_size,
            },
        )

    # ─── P5 Relations ──────────────────────────────────────────────

    async def _phase_relations(self, state: _DreamState) -> PhaseOutcome:
        similar_edges = await write_similar_to_edges(
            self._store, state.cluster_assignments
        )
        derived_edges = await write_derived_from_edges(
            self._store, state.extracted
        )
        return PhaseOutcome(
            phase=PhaseName.RELATIONS,
            status="ok",
            details={
                "similar_to_edges": similar_edges,
                "derived_from_edges": derived_edges,
                "total_edges": similar_edges + derived_edges,
            },
        )
