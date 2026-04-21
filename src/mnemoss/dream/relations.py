"""P5 Relations — graph edges derived from P2 clusters and P3 extractions.

Stage 4 adds two edge types on top of the Stage 1 ``co_occurs_in_session``:

- ``similar_to`` (symmetric): pairwise within each non-noise cluster.
  Confidence = the lower of the two members' HDBSCAN probabilities, so
  a weak outlier doesn't inflate an otherwise-strong cluster's edges.
- ``derived_from`` (directed): new extracted memory → each source
  cluster member. Spreading activation uses this so a recalled fact
  fans into its supporting episodes.

Supersedes edges (when a new extraction conflicts with an older one)
arrive with Stage 5's P6 Generalize / conflict-resolution work.
"""

from __future__ import annotations

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
            await store.write_relation(
                new_mem.id, parent_id, "derived_from", 1.0
            )
            count += 1
    return count
