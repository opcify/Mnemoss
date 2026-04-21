"""P7 Rebalance — recompute idx_priority + tier for every memory.

This is the *metadata-only* pass described in §2.4 and the Dreaming P7
step in §2.5: walk every memory, evaluate the current base level and
protections, write back a fresh ``idx_priority`` and matching
``index_tier``. The Memory content and embedding never change.

In Stage 2 this is exposed directly on ``Mnemoss.rebalance()`` so users
can trigger it manually (e.g. at session end). Stage 4 will wrap it as
dreaming phase P7; because ``rebalance`` takes a store and params, the
same function serves both.

Pin handling: the formula's ``γ·pinned`` boost keeps a pinned memory
HOT even when its base level has decayed. Pinning is per-(memory, agent)
but ``idx_priority`` is a memory-wide property, so rebalance treats a
memory as pinned if *any* agent (including the ambient ``NULL``) has
pinned it. This matches user intent: pinning is a protection signal for
the memory itself, not a per-agent privacy signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import IndexTier
from mnemoss.formula.base_level import compute_base_level
from mnemoss.formula.idx_priority import compute_idx_priority, idx_priority_to_tier
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


@dataclass
class RebalanceStats:
    """Summary returned from a single rebalance pass."""

    scanned: int = 0
    migrated: int = 0
    tier_before: dict[IndexTier, int] = field(
        default_factory=lambda: {t: 0 for t in IndexTier}
    )
    tier_after: dict[IndexTier, int] = field(
        default_factory=lambda: {t: 0 for t in IndexTier}
    )


async def rebalance(
    store: SQLiteBackend,
    params: FormulaParams,
    *,
    now: datetime | None = None,
    batch_size: int = 500,
) -> RebalanceStats:
    """Recompute idx_priority + tier for every memory in the store.

    Returns per-tier counts before/after and the number of migrations. A
    migration is a tier change, not merely an ``idx_priority`` change —
    idx_priority is always rewritten, but migrations specifically tell
    callers how many memories crossed a threshold.
    """

    t = now if now is not None else datetime.now(UTC)
    stats = RebalanceStats()
    stats.tier_before = await store.tier_counts()

    all_ids = await store.iter_memory_ids()
    for start in range(0, len(all_ids), batch_size):
        batch = all_ids[start : start + batch_size]
        memories = await store.materialize_memories(batch)
        pinned_set = await store.pinned_any(batch)

        for memory in memories:
            stats.scanned += 1
            b = compute_base_level(
                memory.access_history, t, memory.created_at, params
            )
            ip = compute_idx_priority(
                base_level=b,
                salience=memory.salience,
                emotional_weight=memory.emotional_weight,
                pinned=memory.id in pinned_set,
                params=params,
            )
            new_tier = idx_priority_to_tier(ip)
            if new_tier is not memory.index_tier:
                stats.migrated += 1
            await store.update_idx_priority(memory.id, ip, new_tier)

    stats.tier_after = await store.tier_counts()
    return stats
