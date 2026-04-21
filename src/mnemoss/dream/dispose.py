"""P8 Dispose — formula-driven memory disposal.

Zero LLM calls. Every decision is derived from ``compute_base_level``
and cluster geometry; if the criterion doesn't fire, the memory stays.
Stage 5 ships two of the three criteria from §1.8:

- ``activation_dead``: ``max A_i < τ - δ``. Once the ceiling on
  activation (``B_i + S_max + MP + ε_max``) falls below the retrieval
  threshold minus a safety margin, the memory cannot surface even
  under the best-case conditions — dispose.
- ``redundant``: part of a large cluster (``≥5``), close to its
  centroid (``similarity > 0.92``), and *not* the representative.
  Duplicative detail best summarized by the cluster's P3 fact.

The third criterion, ``fact_covered``, requires comparing each episode
against an aggregated-facts embedding; that's useful but more
expensive and lands in Stage 6+ when we have the benchmark pressure
to justify it.

Hard protections veto any disposal, per §1.8:

- Pinned by any agent.
- ``salience > 0.8`` or ``emotional_weight > 0.7``.
- ``age_days < 30`` (minimum retention period).

Disposal writes a ``Tombstone`` pointing back at the source messages
in the Raw Log, then removes the memory + its vectors + its FTS entry
+ its relations + its pins. The Raw Log itself is never touched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory, Tombstone
from mnemoss.formula.base_level import compute_base_level
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc
log = logging.getLogger(__name__)

MIN_AGE_DAYS = 30
SALIENCE_FLOOR = 0.8
EMOTIONAL_FLOOR = 0.7
REDUNDANT_CLUSTER_MIN = 5
REDUNDANT_SIMILARITY_MIN = 0.92


@dataclass
class DisposalStats:
    """Summary of one P8 pass."""

    scanned: int = 0
    disposed: int = 0
    activation_dead: int = 0
    redundant: int = 0
    protected: int = 0
    disposed_ids: list[str] = field(default_factory=list)


async def dispose_pass(
    store: SQLiteBackend,
    params: FormulaParams,
    *,
    now: datetime | None = None,
    candidates: list[Memory] | None = None,
) -> DisposalStats:
    """Scan memories and dispose those that meet §1.8's criteria.

    ``candidates`` lets a caller (like the dream runner) restrict the
    scan to the replay set. If ``None`` the whole memory table is
    scanned — O(N) + per-memory disposal cost, fine up to ~50K.
    """

    t = now if now is not None else datetime.now(UTC)
    stats = DisposalStats()

    if candidates is None:
        ids = await store.iter_memory_ids()
        if not ids:
            return stats
        candidates = []
        for i in range(0, len(ids), 500):
            candidates.extend(
                await store.materialize_memories(ids[i : i + 500])
            )

    # Fetch pin status for the whole candidate batch up front.
    pinned_set = await store.pinned_any([m.id for m in candidates])

    # Snapshot cluster sizes at pass-start so iterative deletions don't
    # change the decision for later candidates. Without this, disposing the
    # first tail member shrinks the cluster below the threshold and spares
    # the others — an order-dependent bug.
    cluster_sizes: dict[str, int] = {}
    for m in candidates:
        if m.cluster_id and m.cluster_id not in cluster_sizes:
            cluster_sizes[m.cluster_id] = await store.cluster_size(m.cluster_id)

    ceiling = params.s_max + params.mp + params.epsilon_max
    dispose_threshold = params.tau - params.delta  # max A_i strictly below → dispose

    for memory in candidates:
        stats.scanned += 1

        if _is_protected(memory, t, pinned_set):
            stats.protected += 1
            continue

        b = compute_base_level(memory.access_history, t, memory.created_at, params)

        reason: str | None = None
        if b + ceiling < dispose_threshold:
            reason = "activation_dead"
        elif _is_redundant_static(memory, cluster_sizes):
            reason = "redundant"

        if reason is None:
            continue

        tombstone = Tombstone(
            original_id=memory.id,
            workspace_id=memory.workspace_id,
            agent_id=memory.agent_id,
            dropped_at=t,
            reason=reason,
            gist_snapshot=(
                memory.extracted_gist
                or _snapshot(memory.content)
            ),
            b_at_drop=b,
            source_message_ids=list(memory.source_message_ids),
        )
        await store.write_tombstone(tombstone)
        await store.delete_memory_completely(memory.id)

        stats.disposed += 1
        stats.disposed_ids.append(memory.id)
        if reason == "activation_dead":
            stats.activation_dead += 1
        elif reason == "redundant":
            stats.redundant += 1

    return stats


def _is_protected(memory: Memory, now: datetime, pinned: set[str]) -> bool:
    if memory.id in pinned:
        return True
    if memory.salience > SALIENCE_FLOOR:
        return True
    if memory.emotional_weight > EMOTIONAL_FLOOR:
        return True
    age_days = (now - memory.created_at).total_seconds() / 86400.0
    return age_days < MIN_AGE_DAYS


def _is_redundant_static(
    memory: Memory, cluster_sizes: dict[str, int]
) -> bool:
    if memory.cluster_id is None:
        return False
    if memory.is_cluster_representative:
        return False
    if (memory.cluster_similarity or 0.0) <= REDUNDANT_SIMILARITY_MIN:
        return False
    return cluster_sizes.get(memory.cluster_id, 0) >= REDUNDANT_CLUSTER_MIN


def _snapshot(content: str, *, limit: int = 200) -> str:
    text = content.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
