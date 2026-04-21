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

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import IndexTier, Memory
from mnemoss.encoder import Embedder
from mnemoss.encoder.extraction import extract_heuristic
from mnemoss.formula.activation import ActivationBreakdown, compute_activation
from mnemoss.formula.query_bias import has_deep_cue
from mnemoss.store.sqlite_backend import SQLiteBackend
from mnemoss.working import WorkingMemory

UTC = timezone.utc


@dataclass
class RecallResult:
    """One entry in the list returned to the caller of ``recall``."""

    memory: Memory
    score: float
    breakdown: ActivationBreakdown


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
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._working = working
        self._params = params
        self._rng = rng if rng is not None else random.Random()

    async def recall(
        self,
        query: str,
        *,
        agent_id: str | None,
        k: int = 5,
        pool_size: int = 32,
        include_deep: bool = False,
    ) -> list[RecallResult]:
        """Score candidates tier-by-tier; return the top-k whose A > tau."""

        top, _ = await self._recall_with_stats(
            query,
            agent_id=agent_id,
            k=k,
            pool_size=pool_size,
            include_deep=include_deep,
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
    ) -> tuple[list[RecallResult], CascadeStats]:
        """Same as ``recall`` but also returns cascade telemetry."""

        return await self._recall_with_stats(
            query,
            agent_id=agent_id,
            k=k,
            pool_size=pool_size,
            include_deep=include_deep,
        )

    async def _recall_with_stats(
        self,
        query: str,
        *,
        agent_id: str | None,
        k: int,
        pool_size: int,
        include_deep: bool,
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
