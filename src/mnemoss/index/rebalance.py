"""P7 Rebalance — recompute idx_priority and re-bucket every memory by capacity.

Two stages:

1. **Recompute** ``idx_priority`` for every memory using the current
   activation formula and current ``now`` (storage-path decay
   ``d_storage``). The Memory content, embedding, and relations are
   never touched. This stage is unchanged from the original threshold-
   based bucketing.

2. **Re-bucket** memories into HOT / WARM / COLD / DEEP by **capacity**
   instead of by threshold. Sort all memories by ``idx_priority``
   descending; pinned memories take the top of HOT regardless of rank;
   then fill HOT, WARM, COLD top-down by absolute caps from
   :class:`TierCapacityParams`. Whatever doesn't fit goes to DEEP.

Why capacity-based instead of threshold-based
----------------------------------------------

The original rule was ``HOT iff idx_priority > 0.7``. This degenerates
under any time-skewed workload: ``B_i = ln(Σ (t-t_k)^-d)`` collapses
for memories older than ~1 hour with default parameters, ``idx_priority``
falls below 0.1 for ~99% of them, and HOT empties — defeating the
"fast cascade scan" the architecture promises.

Capacity-based bucketing is structurally bounded: HOT, WARM, COLD all
stay constant-size regardless of N, formula tuning, or corpus age.
DEEP absorbs the long tail. Cascade latency is then ``O(log(hot_cap) +
log(warm_cap) + log(cold_cap))`` independent of total memory count.

This also maps directly onto cognitive architecture: working memory
has a hard biological cap (Miller 1956: 7±2; Cowan 2001: 4±1), not a
threshold. A polymath who knows 10M facts has the same ~5-item working
memory as anyone else; their advantage is in long-term retrieval, not
in working-memory size. The Mnemoss tier caps follow the same
structural property.

Pin handling
------------

Pinned memories are forced HOT, displacing lower-priority non-pinned
entries to fit. Pinning per-(memory, agent) is treated as
"if any agent pinned it, treat as pinned" — pinning expresses memory
salience, not per-agent privacy. If pin count exceeds ``hot_cap``,
HOT can temporarily exceed its cap (cap is a target, not a hard
truncation that would silently drop user pins).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from mnemoss.core.config import FormulaParams, TierCapacityParams
from mnemoss.core.types import IndexTier
from mnemoss.formula.base_level import compute_base_level
from mnemoss.formula.idx_priority import compute_idx_priority
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


@dataclass
class RebalanceStats:
    """Summary returned from a single rebalance pass."""

    scanned: int = 0
    migrated: int = 0
    tier_before: dict[IndexTier, int] = field(default_factory=lambda: {t: 0 for t in IndexTier})
    tier_after: dict[IndexTier, int] = field(default_factory=lambda: {t: 0 for t in IndexTier})


def _bucket_by_capacity(
    ranked_ids: list[str],
    pinned_ids: set[str],
    capacity: TierCapacityParams,
) -> dict[str, IndexTier]:
    """Assign each id to a tier given the priority ranking and pin set.

    ``ranked_ids`` is sorted by ``idx_priority`` descending. Pinned
    members are *promoted* to HOT regardless of their rank, occupying
    the first ``len(pinned)`` HOT seats. Remaining seats fill from the
    top of the non-pinned ranking.
    """

    out: dict[str, IndexTier] = {}

    pinned_in_order = [mid for mid in ranked_ids if mid in pinned_ids]
    non_pinned = [mid for mid in ranked_ids if mid not in pinned_ids]

    for mid in pinned_in_order:
        out[mid] = IndexTier.HOT

    cursor = 0
    remaining_hot = max(0, capacity.hot_cap - len(pinned_in_order))
    bands: list[tuple[IndexTier, int]] = [
        (IndexTier.HOT, remaining_hot),
        (IndexTier.WARM, capacity.warm_cap),
        (IndexTier.COLD, capacity.cold_cap),
    ]
    for tier, cap in bands:
        end = cursor + max(0, cap)
        for mid in non_pinned[cursor:end]:
            out[mid] = tier
        cursor = end
    for mid in non_pinned[cursor:]:
        out[mid] = IndexTier.DEEP

    return out


async def rebalance(
    store: SQLiteBackend,
    params: FormulaParams,
    capacity: TierCapacityParams,
    *,
    now: datetime | None = None,
    batch_size: int = 500,
) -> RebalanceStats:
    """Recompute idx_priority + re-bucket every memory in the store.

    Returns per-tier counts before/after and the number of migrations.
    A migration is a tier change, not merely an ``idx_priority``
    change — idx_priority is always rewritten, but migrations
    specifically tell callers how many memories crossed a bucket.
    """

    t = now if now is not None else datetime.now(UTC)
    stats = RebalanceStats()
    stats.tier_before = await store.tier_counts()

    all_ids = await store.iter_memory_ids()

    # Stage 1: recompute idx_priority for every memory. Persist the
    # value but defer tier assignment until we have the global ranking.
    priorities: dict[str, float] = {}
    prev_tier: dict[str, IndexTier] = {}
    pinned_global: set[str] = set()

    for start in range(0, len(all_ids), batch_size):
        batch = all_ids[start : start + batch_size]
        memories = await store.materialize_memories(batch)
        pinned_set = await store.pinned_any(batch)
        pinned_global.update(pinned_set)

        for memory in memories:
            stats.scanned += 1
            b = compute_base_level(
                memory.access_history, t, memory.created_at, params, d=params.d_storage
            )
            ip = compute_idx_priority(
                base_level=b,
                salience=memory.salience,
                emotional_weight=memory.emotional_weight,
                pinned=memory.id in pinned_set,
                params=params,
            )
            priorities[memory.id] = ip
            prev_tier[memory.id] = memory.index_tier

    # Stage 2: rank-and-bucket by capacity.
    ranked = sorted(priorities.keys(), key=lambda mid: priorities[mid], reverse=True)
    new_tiers = _bucket_by_capacity(ranked, pinned_global, capacity)

    # Stage 3: persist priority + tier per memory; count migrations.
    for mid, ip in priorities.items():
        new_tier = new_tiers.get(mid, IndexTier.DEEP)
        if new_tier is not prev_tier.get(mid):
            stats.migrated += 1
        await store.update_idx_priority(mid, ip, new_tier)

    stats.tier_after = await store.tier_counts()
    return stats
