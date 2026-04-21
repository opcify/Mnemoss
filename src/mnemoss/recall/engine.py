"""Retrieval pipeline.

Stage 2 cascade: scan HOT first, fall through to WARM and COLD only when
no candidate clears the tier's confidence threshold. DEEP is excluded
from default recall and only included when the caller opts in via
``include_deep`` (or, in Checkpoint H, when the query contains a strong
temporal-distance cue).

Candidates are scored exactly once — when a tier adds new memories, only
the fresh ones go through the formula, so noise samples stay stable
across tiers and the ordering is deterministic within a single call.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import IndexTier, Memory
from mnemoss.encoder import Embedder
from mnemoss.encoder.extraction import extract_heuristic
from mnemoss.formula.activation import ActivationBreakdown, compute_activation
from mnemoss.formula.query_bias import has_deep_cue
from mnemoss.recall.expand import expand_from_seeds, hops_for_streak
from mnemoss.recall.history import PastQuery, RecallHistory, is_same_topic
from mnemoss.store.sqlite_backend import SQLiteBackend
from mnemoss.working import WorkingMemory

UTC = timezone.utc


@dataclass
class RecallResult:
    """One entry in the list returned to the caller of ``recall``.

    ``source`` distinguishes direct cascade hits from memories surfaced
    by the auto-expand path. Direct hits cleared the activation formula
    on the query itself; expanded hits were reached via the relation
    graph from the direct hits, so their ranking leans on spreading
    activation and their ``bm25_raw`` / ``cos_sim`` are zero.
    """

    memory: Memory
    score: float
    breakdown: ActivationBreakdown
    source: Literal["direct", "expanded"] = "direct"


@dataclass
class CascadeStats:
    """Per-call cascade telemetry. Useful in tests and for tuning."""

    tiers_scanned: list[IndexTier]
    stopped_at: IndexTier | None  # None means scan exhausted without early-stop
    candidates_scored: int


class RecallEngine:
    def __init__(
        self,
        store: SQLiteBackend,
        embedder: Embedder,
        working: WorkingMemory,
        params: FormulaParams,
        rng: random.Random | None = None,
        history: RecallHistory | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._working = working
        self._params = params
        self._rng = rng if rng is not None else random.Random()
        self._history = history if history is not None else RecallHistory()

    async def recall(
        self,
        query: str,
        *,
        agent_id: str | None,
        k: int = 5,
        pool_size: int = 32,
        include_deep: bool = False,
        auto_expand: bool = True,
    ) -> list[RecallResult]:
        """Score candidates tier-by-tier; return the top-k whose A > tau.

        When ``auto_expand`` is true (the default), a follow-up recall on
        the same topic as the previous one — defined by either result
        overlap or query-embedding cosine within the same-topic window —
        additionally surfaces spreading-reached memories ranked by the
        full activation formula. See ``recall/expand.py`` for the
        algorithm and ``recall/history.py`` for the detection rule.
        """

        top, _ = await self._recall_with_stats(
            query,
            agent_id=agent_id,
            k=k,
            pool_size=pool_size,
            include_deep=include_deep,
            auto_expand=auto_expand,
        )
        return top

    async def recall_with_stats(
        self,
        query: str,
        *,
        agent_id: str | None,
        k: int = 5,
        pool_size: int = 32,
        include_deep: bool = False,
        auto_expand: bool = True,
    ) -> tuple[list[RecallResult], CascadeStats]:
        """Same as ``recall`` but also returns cascade telemetry."""

        return await self._recall_with_stats(
            query,
            agent_id=agent_id,
            k=k,
            pool_size=pool_size,
            include_deep=include_deep,
            auto_expand=auto_expand,
        )

    async def _recall_with_stats(
        self,
        query: str,
        *,
        agent_id: str | None,
        k: int,
        pool_size: int,
        include_deep: bool,
        auto_expand: bool,
    ) -> tuple[list[RecallResult], CascadeStats]:
        query_vec = (await asyncio.to_thread(self._embedder.embed, [query]))[0]
        now = datetime.now(UTC)
        active_set = self._working.active_set(agent_id)
        tau = self._params.tau

        cos_by_id: dict[str, float] = {}
        bm25_by_id: dict[str, float] = {}
        scored: dict[str, RecallResult] = {}

        # Auto-include DEEP when the query has a temporal-distance marker.
        effective_include_deep = include_deep or has_deep_cue(query)
        tier_plan = _tier_plan(self._params, include_deep=effective_include_deep)
        tiers_scanned: list[IndexTier] = []
        stopped_at: IndexTier | None = None

        for tier, confidence in tier_plan:
            vec_task = asyncio.create_task(
                self._store.vec_search(
                    query_vec, pool_size, agent_id, tier_filter={tier}
                )
            )
            fts_task = asyncio.create_task(
                self._store.fts_search(
                    query, pool_size, agent_id, tier_filter={tier}
                )
            )
            vec_hits, fts_hits = await asyncio.gather(vec_task, fts_task)
            tiers_scanned.append(tier)

            for mid, cos in vec_hits:
                cos_by_id.setdefault(mid, cos)
            for mid, bm25 in fts_hits:
                bm25_by_id.setdefault(mid, bm25)

            new_ids = (
                {m for m, _ in vec_hits}
                | {m for m, _ in fts_hits}
            ) - scored.keys()

            if new_ids:
                await self._score_candidates(
                    new_ids=new_ids,
                    query=query,
                    now=now,
                    agent_id=agent_id,
                    active_set=active_set,
                    cos_by_id=cos_by_id,
                    bm25_by_id=bm25_by_id,
                    scored=scored,
                )

            if scored:
                top_score = max(r.score for r in scored.values())
                if top_score >= confidence:
                    stopped_at = tier
                    break

        top = sorted(scored.values(), key=lambda r: r.score, reverse=True)
        # Secondary threshold: only return candidates clearing tau.
        top = [r for r in top if r.score > tau][:k]

        # ─── auto-expand on same-topic follow-up ─────────────────────
        # Runs AFTER the direct cascade so expansion seeds from the real
        # top-k; runs BEFORE reconsolidation so expanded memories also
        # get their access_history bumped. Skipped on empty direct
        # results (nothing to spread from) and when the caller opted out.
        #
        # Two independent time scales:
        # - Same-topic detection is pure semantic (result overlap or
        #   query cosine) — no time gate. A user returning to the thread
        #   hours later still benefits.
        # - Streak (for hop-count escalation) resets after a gap longer
        #   than ``streak_reset_seconds``, so a fresh thread starts at
        #   hops=1 even when the topic is clearly the same.
        #
        #   recall()
        #      │
        #      ├─► cascade HOT→WARM→COLD, filter τ, take top-k
        #      │
        #      ├─► auto_expand AND top non-empty AND prev in history?
        #      │       │                                    │  no
        #      │       │ yes                                ▼
        #      │       ▼                              record + return
        #      │    is_same_topic(prev)?
        #      │       │                                    │  no
        #      │       │ yes                                ▼
        #      │       ▼                              record + return
        #      │    gap ≤ streak_reset?  yes → streak = prev+1
        #      │                         no  → streak = 1
        #      │       │
        #      │       ▼
        #      │    hops = hops_for_streak(streak, hops_max)
        #      │    BFS relation graph (capped by expand_candidates_max)
        #      │    score with bm25=0, cos=0, seeds in active_set
        #      │    filter τ, take limit=k, tag source="expanded"
        #      │       │
        #      └───────┴──► record history, reconsolidate, return
        streak = 1
        if auto_expand and top:
            prev = self._history.latest(agent_id)
            current_result_ids = {r.memory.id for r in top}
            if prev is not None and is_same_topic(
                prev,
                current_query_vec=query_vec,
                current_result_ids=current_result_ids,
                cosine_threshold=self._params.same_topic_cosine,
            ):
                # ``max(0, …)`` guards against wall-clock skew producing
                # a negative gap (e.g. NTP step-back, replayed timestamps
                # in tests). Without the clamp, a negative gap would
                # always pass the ≤ check and streak could grow
                # indefinitely.
                gap_seconds = max(
                    0.0, (now - prev.timestamp).total_seconds()
                )
                if gap_seconds <= self._params.streak_reset_seconds:
                    streak = prev.streak + 1
                # else: fresh thread, streak stays at 1 (shallow expansion)
                hops = hops_for_streak(streak, self._params.expand_hops_max)
                expanded = await expand_from_seeds(
                    self._store,
                    seed_memories=[r.memory for r in top],
                    query=query,
                    now=now,
                    agent_id=agent_id,
                    hops=hops,
                    limit=k,
                    params=self._params,
                    rng=self._rng,
                    exclude_ids=current_result_ids | scored.keys(),
                )
                top.extend(
                    RecallResult(
                        memory=c.memory,
                        score=c.score,
                        breakdown=c.breakdown,
                        source="expanded",
                    )
                    for c in expanded
                )

        # Record this recall for future same-topic detection. Done before
        # reconsolidation so the snapshot captures what the caller saw.
        self._history.record(
            agent_id,
            PastQuery(
                query=query,
                query_vec=query_vec,
                timestamp=now,
                result_ids={r.memory.id for r in top},
                streak=streak,
            ),
        )

        for result in top:
            await self._store.reconsolidate(result.memory.id, now)
            result.memory.access_history.append(now)
            result.memory.rehearsal_count += 1
            result.memory.last_accessed_at = now
            # Reminiscence (§1.9 footnote): a DEEP memory that is reactivated
            # jumps to WARM and bumps reminisced_count. The next rebalance
            # will recompute idx_priority exactly, but we set it to
            # mid-WARM here so the state stays consistent in the interim.
            if result.memory.index_tier is IndexTier.DEEP:
                await self._store.reminisce_to_warm(result.memory.id)
                result.memory.reminisced_count += 1
                result.memory.index_tier = IndexTier.WARM
                result.memory.idx_priority = 0.5
        self._working.extend(agent_id, (r.memory.id for r in top))

        # Lazy heuristic extraction on the returned top-k (Stage 3 §9.541).
        # Only runs once per memory — extraction_level=1 skips re-extraction
        # until Stage 4's LLM refinement upgrades to level=2.
        await self._apply_lazy_extraction(top)

        stats = CascadeStats(
            tiers_scanned=tiers_scanned,
            stopped_at=stopped_at,
            candidates_scored=len(scored),
        )
        return top, stats

    async def _apply_lazy_extraction(self, results: list[RecallResult]) -> None:
        """Fill ``extracted_*`` fields on any top-k memory at level 0."""

        pending = [r.memory for r in results if r.memory.extraction_level == 0]
        if not pending:
            return
        # Extraction is CPU-bound regex work — run the whole batch on the
        # shared thread pool so we don't block the event loop. For a Stage 3
        # k≈5, the whole pass finishes in a few milliseconds.
        fields_list = await asyncio.to_thread(
            _batch_extract, [m.content for m in pending]
        )
        for memory, fields in zip(pending, fields_list, strict=True):
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

    async def _score_candidates(
        self,
        *,
        new_ids: set[str],
        query: str,
        now: datetime,
        agent_id: str | None,
        active_set: list[str],
        cos_by_id: dict[str, float],
        bm25_by_id: dict[str, float],
        scored: dict[str, RecallResult],
    ) -> None:
        memories = await self._store.materialize_memories(list(new_ids))
        # Fan-out and out-edges need to cover both the active set (for
        # spreading *into* our candidates) and the new candidates themselves
        # (since a candidate may be a spreading source for another candidate).
        all_for_relations = list({*active_set, *new_ids, *scored.keys()})
        relations_from = await self._store.relations_from(all_for_relations)
        fan_of = await self._store.fan_out(all_for_relations)

        for memory in memories:
            pinned = await self._store.is_pinned(memory.id, agent_id)
            breakdown = compute_activation(
                memory=memory,
                query=query,
                now=now,
                active_set=active_set,
                relations_from=relations_from,
                fan_of=fan_of,
                bm25_raw=bm25_by_id.get(memory.id, 0.0),
                cos_sim=cos_by_id.get(memory.id, 0.0),
                pinned=pinned,
                rng=self._rng,
                params=self._params,
            )
            scored[memory.id] = RecallResult(
                memory=memory, score=breakdown.total, breakdown=breakdown
            )

    async def explain(
        self,
        query: str,
        memory_id: str,
        *,
        agent_id: str | None,
    ) -> ActivationBreakdown | None:
        """Return the per-term breakdown without reconsolidating.

        Scans across all tiers (including DEEP) so explain always finds
        the target memory regardless of its current tier.
        """

        memory = await self._store.get_memory(memory_id)
        if memory is None:
            return None
        query_vec = (await asyncio.to_thread(self._embedder.embed, [query]))[0]
        cos_hits = await self._store.vec_search(query_vec, k=200, agent_id=agent_id)
        fts_hits = await self._store.fts_search(query, k=200, agent_id=agent_id)
        cos_by_id = dict(cos_hits)
        bm25_by_id = dict(fts_hits)
        active_set = self._working.active_set(agent_id)
        relations_from = await self._store.relations_from([*active_set, memory_id])
        fan_of = await self._store.fan_out([*active_set, memory_id])
        pinned = await self._store.is_pinned(memory_id, agent_id)
        return compute_activation(
            memory=memory,
            query=query,
            now=datetime.now(UTC),
            active_set=active_set,
            relations_from=relations_from,
            fan_of=fan_of,
            bm25_raw=bm25_by_id.get(memory_id, 0.0),
            cos_sim=cos_by_id.get(memory_id, 0.0),
            pinned=pinned,
            rng=random.Random(0),  # deterministic for explain
            params=self._params,
        )

    async def expand_recall(
        self,
        memory_id: str,
        *,
        agent_id: str | None,
        query: str | None = None,
        hops: int = 1,
        k: int = 5,
    ) -> list[RecallResult]:
        """Explicitly expand from one memory via the relation graph.

        Bypasses the same-topic heuristic used by auto-expand. Use when
        the caller has an external signal for "dig deeper now" — an
        agent framework's own follow-up detection, a user click on
        "show related," a planner asking "what else is connected?".

        ``query`` seeds the matching term of the activation formula. If
        ``None`` (default), the seed memory's own content is used, which
        rewards candidates that are both related to the seed *and*
        textually similar. Pass an empty string to disable matching
        entirely (pure spreading + base-level). Seed memory is excluded
        from the returned list — the caller already has it.

        Reconsolidates the seed and every returned memory: calling
        ``expand()`` is itself an act of engaging with the seed, so
        access_history gets bumped just like for direct recall.
        """

        if hops < 1:
            raise ValueError(f"hops must be >= 1, got {hops}")

        seed = await self._store.get_memory(memory_id)
        if seed is None:
            return []

        # Agent scoping: caller bound to ``agent_id`` can only seed
        # expansion from memories they can see.
        if agent_id is None:
            if seed.agent_id is not None:
                return []
        else:
            if seed.agent_id not in (agent_id, None):
                return []

        effective_query = seed.content if query is None else query
        now = datetime.now(UTC)

        expanded = await expand_from_seeds(
            self._store,
            seed_memories=[seed],
            query=effective_query,
            now=now,
            agent_id=agent_id,
            hops=hops,
            limit=k,
            params=self._params,
            rng=self._rng,
            exclude_ids={memory_id},
        )
        results = [
            RecallResult(
                memory=c.memory,
                score=c.score,
                breakdown=c.breakdown,
                source="expanded",
            )
            for c in expanded
        ]

        # Reconsolidate seed + every returned memory. User explicitly
        # engaged with these; the formula treats access as rehearsal.
        await self._store.reconsolidate(seed.id, now)
        for r in results:
            await self._store.reconsolidate(r.memory.id, now)
            r.memory.access_history.append(now)
            r.memory.rehearsal_count += 1
            r.memory.last_accessed_at = now
        self._working.extend(agent_id, (r.memory.id for r in results))

        return results


def _batch_extract(contents: list[str]):
    """Module-level helper so asyncio.to_thread has something picklable-ish
    to call without reaching into the engine."""

    return [extract_heuristic(c) for c in contents]


def _tier_plan(
    params: FormulaParams, *, include_deep: bool
) -> list[tuple[IndexTier, float]]:
    """Return ``[(tier, confidence_threshold)]`` in cascade order.

    DEEP has no early-stop threshold — once the cascade reaches it we scan
    fully, then fall through to return whatever cleared ``tau``.
    """

    plan: list[tuple[IndexTier, float]] = [
        (IndexTier.HOT, params.tau + params.confidence_hot_offset),
        (IndexTier.WARM, params.tau + params.confidence_warm_offset),
        (IndexTier.COLD, params.tau + params.confidence_cold_offset),
    ]
    if include_deep:
        plan.append((IndexTier.DEEP, -math.inf))
    return plan
