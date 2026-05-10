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
    extract_atomic_facts_from_singleton,
)
from mnemoss.dream.cost import CostLedger, CostLimits
from mnemoss.dream.dispose import dispose_pass
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
    #
    # The former Relations phase was removed in 2026-04-27 after the
    # dreaming-validation study found it actively hurt multi-hop recall
    # by 17pp on the topology corpus (full clique similar_to edges
    # caused spreading activation to surface peripheral cluster
    # members). derived_from edges are still written inline by
    # Consolidate's _persist_derived. See docs/dreaming-decision.md.
    TriggerType.IDLE: [
        PhaseName.REPLAY,
        PhaseName.CLUSTER,
        PhaseName.CONSOLIDATE,
    ],
    TriggerType.SESSION_END: [
        PhaseName.REPLAY,
        PhaseName.CLUSTER,
        PhaseName.CONSOLIDATE,
    ],
    TriggerType.SURPRISE: [
        PhaseName.CONSOLIDATE,
    ],
    TriggerType.COGNITIVE_LOAD: [
        PhaseName.CONSOLIDATE,
    ],
    TriggerType.NIGHTLY: [
        PhaseName.REPLAY,
        PhaseName.CLUSTER,
        PhaseName.CONSOLIDATE,
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
        process_singletons: bool = False,
        singleton_salience_threshold: float = 0.5,
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
        # Opt-in: after the cluster loop, run a per-memory atomic-fact
        # extraction pass over HDBSCAN-noise singletons. Closes the gap
        # where a uniquely-named entity (LongMemEval-S 51a45a95 Target,
        # for instance) lives in a turn with no semantic neighbours and
        # therefore never reaches the cluster-level LLM. Disabled by
        # default — the per-memory pass adds N_singleton LLM calls per
        # dream run (typically O(N_replay)) and bench evidence is
        # young; gate via this flag until the prod budget story lands.
        self._process_singletons = process_singletons
        # Encoder-side salience cutoff for which singletons are
        # eligible. The pilot's blanket sweep (threshold=0) regressed
        # accuracy because filler turns flooded the recall pool with
        # noise; a fact-bearing threshold (≥0.5) keeps the upside.
        self._singleton_salience_threshold = singleton_salience_threshold

    async def run(
        self,
        trigger: TriggerType,
        *,
        agent_id: str | None = None,
        now: datetime | None = None,
        phases: set[str] | None = None,
    ) -> DreamReport:
        """Run one dream cycle.

        ``phases`` (optional) is an ablation mask — a set of phase
        ``value`` strings (e.g. ``{"replay", "cluster"}``). When provided,
        only listed phases run; excluded phases get a ``PhaseOutcome``
        with ``status="excluded_by_mask"``. Default ``None`` runs every
        phase the trigger normally runs.

        Dependency rules are intentionally **not** enforced by the mask
        — downstream phases that consume empty state record their
        existing skip reasons (``"empty replay set"``, etc.). The mask
        only filters the iteration loop. See design doc.
        """

        t0 = now if now is not None else datetime.now(UTC)
        report = DreamReport(
            trigger=trigger,
            started_at=t0,
            finished_at=t0,
            agent_id=agent_id,
        )
        state = _DreamState()

        for phase in PHASES_BY_TRIGGER.get(trigger, []):
            if phases is not None and phase.value not in phases:
                report.outcomes.append(
                    PhaseOutcome(
                        phase=phase,
                        status="excluded_by_mask",
                        skip_reason="excluded by phases mask",
                    )
                )
                continue
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
        atomic_facts: list[Memory] = []
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

            # (D) Atomic facts — concrete propositional FACT memories.
            # These are the LongMemEval-S architectural lever:
            # narrative summaries collapse multiple discrete claims
            # into one row; atomic facts surface each claim as its
            # own indexable Memory so recall can rank the specific
            # fact (e.g. "$400K mortgage at Wells Fargo") above the
            # raw turn that contained it.
            for fact in result.atomic_facts:
                await self._persist_derived(fact)
                atomic_facts.append(fact)

        # (E) Singleton sweep — opt-in. HDBSCAN-noise memories never
        # reach the cluster loop above; the LongMemEval-S 51a45a95
        # (Target store) miss is exactly this case. When
        # ``process_singletons=True``, run a tighter per-memory atomic-
        # fact extraction over every replay-set member that wasn't
        # placed in a cluster. Same budget gate as the cluster loop.
        singleton_atomic_facts: list[Memory] = []
        singletons_processed = 0
        singletons_filtered_low_salience = 0
        if self._process_singletons:
            singletons = self._singletons_from_state(state)
            # Salience filter: skip filler/agreement/small-talk
            # singletons that would cost an LLM call without surfacing
            # a fact. The encoder already scores salience ∈ [0, 1] from
            # 5 signals (proper nouns, numerics, length, etc.).
            eligible: list[Memory] = []
            for sm in singletons:
                if sm.salience >= self._singleton_salience_threshold:
                    eligible.append(sm)
                else:
                    singletons_filtered_low_salience += 1
            for sm in eligible:
                if self._cost_ledger is not None:
                    reason = self._cost_ledger.check_budget(
                        self._cost_limits,
                        run_calls=llm_calls_made,
                        now=now,
                    )
                    if reason is not None:
                        budget_skips.append(reason)
                        break
                facts = await extract_atomic_facts_from_singleton(
                    sm, self._llm, self._params, now=now
                )
                llm_calls_made += 1
                if self._cost_ledger is not None:
                    self._cost_ledger.record_call(now=now)
                singletons_processed += 1
                if not facts:
                    continue
                for fact in facts:
                    await self._persist_derived(fact)
                    singleton_atomic_facts.append(fact)

        # Cross-session edges fan out from every consolidated memory.
        state.consolidated = (
            summaries + patterns + atomic_facts + singleton_atomic_facts
        )
        atomic_facts_total = atomic_facts + singleton_atomic_facts

        return PhaseOutcome(
            phase=PhaseName.CONSOLIDATE,
            status="ok",
            details={
                "clusters_processed": sum(1 for c in clusters if len(c) >= 2),
                "summaries": len(summaries),
                "patterns": len(patterns),
                "atomic_facts": len(atomic_facts),
                "singleton_atomic_facts": len(singleton_atomic_facts),
                "singletons_processed": singletons_processed,
                "singletons_filtered_low_salience": singletons_filtered_low_salience,
                "refined": len(refined_ids),
                "llm_failures": llm_failures,
                "llm_calls_made": llm_calls_made,
                "budget_skips": budget_skips,
                "summary_ids": [m.id for m in summaries],
                "pattern_ids": [m.id for m in patterns],
                "atomic_fact_ids": [m.id for m in atomic_facts_total],
                "refined_ids": refined_ids,
                "cross_agent_promotions": sum(
                    1 for m in (summaries + patterns + atomic_facts_total) if m.agent_id is None
                ),
            },
        )

    def _singletons_from_state(self, state: _DreamState) -> list[Memory]:
        """Return replay-set memories that HDBSCAN labelled as noise.

        These are the ``cluster_assignments[mid].cluster_id is None``
        rows — semantic outliers that didn't end up in any cluster.
        Returns ``[]`` when the trigger skipped CLUSTER (e.g. surprise
        / cognitive_load), since "every memory is a singleton" isn't
        a meaningful sweep target.
        """

        if not state.cluster_assignments or not state.replay_set:
            return []
        unclustered = {
            mid
            for mid, a in state.cluster_assignments.items()
            if a.cluster_id is None
        }
        if not unclustered:
            return []
        return [m for m in state.replay_set if m.id in unclustered]

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
        """Embed + write a derived memory, link its derived_from edges.

        ``link_derived`` only updates the ``memory.derived_to`` JSON
        column. That's invisible to ``expand_via_relations``, which
        reads the ``relation`` table. Without the explicit edge
        writes below, summaries and patterns produced by Consolidate
        would be unreachable through spreading activation — and since
        clusters span sessions, those summary↔source edges are the
        only cross-session links Mnemoss creates. We write both
        directions so BFS works either way: from a recalled summary
        out to its sources, or from a recalled source up to its
        summary (and from there to sibling sources in other sessions).
        """

        assert self._embedder is not None  # guarded by phase entry
        embedding = await asyncio.to_thread(self._embedder.embed, [memory.content])
        await self._store.write_memory(memory, embedding[0])
        await self._store.link_derived(memory.derived_from, memory.id)
        for src_id in memory.derived_from:
            await self._store.write_relation(memory.id, src_id, "derived_from")
            await self._store.write_relation(src_id, memory.id, "derived_to")

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
