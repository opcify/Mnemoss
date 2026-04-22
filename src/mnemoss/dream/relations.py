"""Relations phase — graph edges derived from P2 clusters and P3 Consolidate outputs.

Three edge types on top of the baseline ``co_occurs_in_session``:

- ``similar_to`` (symmetric): pairwise within each non-noise cluster.
  Confidence = the lower of the two members' HDBSCAN probabilities, so
  a weak outlier doesn't inflate an otherwise-strong cluster's edges.
- ``derived_from`` (directed): new consolidated memory → each source
  cluster member. Spreading activation uses this so a recalled fact
  fans into its supporting episodes.
- ``shares_entity`` (symmetric): two refined members of any cluster
  that share at least one canonical entity (as emitted by Dream P3
  Consolidate). This is how NER flows into recall without ever being
  parsed on the query side — working-memory items that share entities
  with candidate memories spread activation through this edge.
  Confidence = Jaccard(entities_a, entities_b) — 1.0 when the sets are
  identical, scaling down with disjointness.

Supersedes edges (when a new consolidation conflicts with an older fact)
are emitted by Consolidate's structured response when the LLM flags a
conflict; the conflict-resolution policy that acts on them lives here.
"""

from __future__ import annotations

from collections.abc import Iterable

from mnemoss.core.types import Memory
from mnemoss.dream.cluster import ClusterAssignment, group_by_cluster
from mnemoss.store.sqlite_backend import SQLiteBackend


async def write_similar_to_edges(
    store: SQLiteBackend,
    assignments: dict[str, ClusterAssignment],
) -> int:
    """For each non-noise cluster, add symmetric similar_to edges between
    every pair of members. Returns the number of edges written."""

    clusters = group_by_cluster(assignments)
    count = 0
    for members in clusters.values():
        for i, a_id in enumerate(members):
            a_sim = assignments[a_id].similarity or 0.0
            for b_id in members[i + 1 :]:
                b_sim = assignments[b_id].similarity or 0.0
                confidence = min(a_sim, b_sim)
                await store.write_relation(a_id, b_id, "similar_to", confidence)
                await store.write_relation(b_id, a_id, "similar_to", confidence)
                count += 2
    return count


async def write_derived_from_edges(
    store: SQLiteBackend,
    extracted: list[Memory],
) -> int:
    """For each newly-extracted Memory, write a ``derived_from`` edge
    pointing at each of its source cluster members. Returns the number
    of edges written."""

    count = 0
    for new_mem in extracted:
        for parent_id in new_mem.derived_from:
            await store.write_relation(new_mem.id, parent_id, "derived_from", 1.0)
            count += 1
    return count


def _canonical_entities(mem: Memory) -> set[str]:
    """Return the lowercased, stripped entity set for one memory.

    Case folding happens here so 'Alice' and 'ALICE' collapse to one
    edge. Dream P3 emits canonical surface forms in the source
    language; we fold only for comparison, not for storage.
    """

    if not mem.extracted_entities:
        return set()
    return {e.strip().casefold() for e in mem.extracted_entities if e and e.strip()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / len(a | b)


async def write_shares_entity_edges(
    store: SQLiteBackend,
    members: Iterable[Memory],
) -> int:
    """Write symmetric ``shares_entity`` edges between any two members
    (of the same cluster or replay set) whose refined entity sets
    intersect.

    Only memories with ``extraction_level >= 2`` participate — level-1
    heuristic entities are always ``None``, and we don't want stale
    extractions producing edges. Returns the number of edges written.
    """

    refined = [m for m in members if m.extraction_level >= 2]
    if len(refined) < 2:
        return 0

    entities_by_id = {m.id: _canonical_entities(m) for m in refined}
    count = 0
    for i, a in enumerate(refined):
        a_ents = entities_by_id[a.id]
        if not a_ents:
            continue
        for b in refined[i + 1 :]:
            b_ents = entities_by_id[b.id]
            if not b_ents:
                continue
            j = _jaccard(a_ents, b_ents)
            if j <= 0.0:
                continue
            await store.write_relation(a.id, b.id, "shares_entity", j)
            await store.write_relation(b.id, a.id, "shares_entity", j)
            count += 2
    return count
