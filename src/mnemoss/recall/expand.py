"""Spreading-activation expansion over the relation graph.

When the caller issues a follow-up recall on the same topic, we extend
the result set with memories reachable through the relation graph from
the seeds. Candidates are scored by the full activation formula but
with the seeds injected into the active set, so spreading activation
does most of the work — the formula automatically favours memories
closely associated with what the user already got, even when their
literal or semantic match to the query is weak.

Intentional omission: expanded candidates are scored with
``bm25_raw = 0`` and ``cos_sim = 0``. A spreading-reached memory that
*also* matches the query directly would already show up in the primary
cascade, so we don't pay for a second vec/FTS pass here. The goal of
expansion is specifically to surface things direct retrieval missed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory
from mnemoss.formula.activation import ActivationBreakdown, compute_activation
from mnemoss.store.sqlite_backend import SQLiteBackend

_EXPAND_PREDICATES: tuple[str, ...] = (
    "similar_to",
    "derived_from",
    "co_occurs_in_session",
)


@dataclass
class ExpandedCandidate:
    """A memory surfaced via spreading activation from the seed set."""

    memory: Memory
    score: float
    breakdown: ActivationBreakdown


def hops_for_streak(streak: int, hops_max: int) -> int:
    """Map consecutive same-topic count → graph radius.

    Streak 1 is the first follow-up: 1 hop. Each additional same-topic
    query in a row widens the net by one hop, capped at
    ``hops_max``. Capping matters because relation fan-out is roughly
    geometric; 3+ hops on a well-connected store can pull in thousands
    of memories.
    """

    return max(1, min(streak, hops_max))


async def expand_from_seeds(
    store: SQLiteBackend,
    *,
    seed_memories: list[Memory],
    query: str,
    now: datetime,
    agent_id: str | None,
    hops: int,
    limit: int,
    params: FormulaParams,
    rng: random.Random,
    exclude_ids: set[str],
) -> list[ExpandedCandidate]:
    """Return up to ``limit`` candidates ranked by activation.

    ``exclude_ids`` should at minimum contain the seeds themselves plus
    anything already returned by the direct cascade, so expansion never
    duplicates a memory the caller already has.
    """

    seed_ids = [m.id for m in seed_memories]
    if not seed_ids:
        return []

    reachable = await store.expand_via_relations(
        seed_ids,
        hops=hops,
        predicates=_EXPAND_PREDICATES,
        max_candidates=params.expand_candidates_max,
    )
    candidate_ids = reachable - exclude_ids
    if not candidate_ids:
        return []

    memories = await store.materialize_memories(list(candidate_ids))
    # Agent scoping: drop anything not visible to ``agent_id``. Ambient
    # callers (agent_id is None) see only other-ambient memories.
    if agent_id is None:
        memories = [m for m in memories if m.agent_id is None]
    else:
        memories = [m for m in memories if m.agent_id in (agent_id, None)]
    if not memories:
        return []

    # ``active_set`` is the seeds only — NOT the caller's working memory.
    # Divergence from the main cascade is intentional: expansion's job is
    # specifically "spread from these hits", so we inject exactly those
    # hits as the spreading source. Mixing in unrelated working-memory
    # entries would dilute the signal that makes expansion useful.
    active_set = list(seed_ids)
    relation_ids = list({*active_set, *(m.id for m in memories)})
    relations_from = await store.relations_from(relation_ids)
    fan_of = await store.fan_out(relation_ids)
    # Batched pin lookup: 1 SQL instead of one-per-candidate.
    pinned_ids = await store.pinned_by_agent([m.id for m in memories], agent_id)

    out: list[ExpandedCandidate] = []
    tau = params.tau
    for memory in memories:
        breakdown = compute_activation(
            memory=memory,
            query=query,
            now=now,
            active_set=active_set,
            relations_from=relations_from,
            fan_of=fan_of,
            bm25_raw=0.0,
            cos_sim=0.0,
            pinned=memory.id in pinned_ids,
            rng=rng,
            params=params,
        )
        if breakdown.total > tau:
            out.append(ExpandedCandidate(memory=memory, score=breakdown.total, breakdown=breakdown))

    out.sort(key=lambda c: c.score, reverse=True)
    return out[:limit]
