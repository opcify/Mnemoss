"""Retrieval pipeline.

Given a query, fan out to FTS + vector search in parallel, union the
candidate pool, filter by agent scope (enforced by the store layer),
score each candidate with the full ACT-R activation formula, keep those
clearing the threshold ``tau``, sort by activation, take top-k, and
reconsolidate **only the returned top-k** (not the full candidate pool).

This is the only file that glues formula + store + working + relations
together, so it also exposes ``explain`` for debugging.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timezone

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory
from mnemoss.encoder import Embedder
from mnemoss.formula.activation import ActivationBreakdown, compute_activation
from mnemoss.store.sqlite_backend import SQLiteBackend
from mnemoss.working import WorkingMemory

UTC = timezone.utc


@dataclass
class RecallResult:
    """One entry in the list returned to the caller of ``recall``."""

    memory: Memory
    score: float
    breakdown: ActivationBreakdown


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
    ) -> list[RecallResult]:
        """Score candidates; return the top-k whose activation clears ``tau``."""

        query_vec = self._embedder.embed([query])[0]

        vec_task = asyncio.create_task(
            self._store.vec_search(query_vec, pool_size, agent_id)
        )
        fts_task = asyncio.create_task(
            self._store.fts_search(query, pool_size, agent_id)
        )
        vec_hits, fts_hits = await asyncio.gather(vec_task, fts_task)

        cos_by_id: dict[str, float] = {mid: cos for mid, cos in vec_hits}
        bm25_by_id: dict[str, float] = {mid: bm25 for mid, bm25 in fts_hits}
        candidate_ids = list({*cos_by_id.keys(), *bm25_by_id.keys()})
        if not candidate_ids:
            return []

        memories = await self._store.materialize_memories(candidate_ids)
        active_set = self._working.active_set(agent_id)
        # Include candidates in the fan-out lookup so co-occurrence parents
        # in the active set can still fire even when their out-edges include
        # candidates only.
        all_for_relations = list({*active_set, *candidate_ids})
        relations_from = await self._store.relations_from(all_for_relations)
        fan_of = await self._store.fan_out(all_for_relations)

        now = datetime.now(UTC)
        scored: list[RecallResult] = []
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
            if breakdown.total < self._params.tau:
                continue
            scored.append(
                RecallResult(memory=memory, score=breakdown.total, breakdown=breakdown)
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        top = scored[:k]

        # Reconsolidate only the returned top-k. Update WM with what the
        # caller actually got back.
        for result in top:
            await self._store.reconsolidate(result.memory.id, now)
            result.memory.access_history.append(now)
            result.memory.rehearsal_count += 1
        self._working.extend(agent_id, (r.memory.id for r in top))

        return top

    async def explain(
        self,
        query: str,
        memory_id: str,
        *,
        agent_id: str | None,
    ) -> ActivationBreakdown | None:
        """Return the per-term breakdown without reconsolidating."""

        memory = await self._store.get_memory(memory_id)
        if memory is None:
            return None
        query_vec = self._embedder.embed([query])[0]
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
