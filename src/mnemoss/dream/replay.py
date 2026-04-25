"""P1 Replay — pick memories worth consolidating.

Uses the same base-level scoring as recall: memories with high
``B_i`` (recent access, high rehearsal) or recent creation are worth
running through the cluster → extract → relations pipeline.

Stage 4 scans the whole memory table, computes ``B_i`` in Python,
sorts, and takes top-``limit``. Fine up to ~50K memories; for larger
workspaces we'll paginate the scan in Stage 6+ when it matters.
"""

from __future__ import annotations

from datetime import datetime

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory
from mnemoss.formula.base_level import compute_base_level
from mnemoss.store.sqlite_backend import SQLiteBackend


async def select_replay_candidates(
    store: SQLiteBackend,
    agent_id: str | None,
    params: FormulaParams,
    *,
    now: datetime,
    limit: int = 100,
    min_base_level: float | None = None,
) -> list[Memory]:
    """Return the top-``limit`` memories ordered by ``B_i`` desc.

    Agent scoping follows the recall rule: ``agent_id=None`` returns
    ambient-only; a non-None id returns that agent's private memories
    plus ambient. ``min_base_level`` filters out memories whose B_i is
    too low to be worth the cluster/extract work; ``None`` disables
    the filter (useful for tests).
    """

    all_ids = await store.iter_memory_ids()
    if not all_ids:
        return []

    memories: list[Memory] = []
    for i in range(0, len(all_ids), 500):
        batch = all_ids[i : i + 500]
        memories.extend(await store.materialize_memories(batch))

    if agent_id is None:
        scoped = [m for m in memories if m.agent_id is None]
    else:
        scoped = [m for m in memories if m.agent_id == agent_id or m.agent_id is None]

    scored: list[tuple[Memory, float]] = []
    for m in scoped:
        # Replay is a storage-time cognition pass — pick memories for
        # Dream to rehearse based on their long-horizon activation, not
        # their moment-to-moment recall rank. Use d_storage.
        b = compute_base_level(
            m.access_history, now, m.created_at, params, d=params.d_storage
        )
        if min_base_level is not None and b < min_base_level:
            continue
        scored.append((m, b))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [m for m, _ in scored[:limit]]
