"""P2 Cluster — HDBSCAN over replay-set embeddings.

We use sklearn's ``HDBSCAN`` (since sklearn 1.3) because it's already
transitive through our ``sentence-transformers`` stack and doesn't need
the fragile native build of the standalone ``hdbscan`` package.

HDBSCAN requires Euclidean distance, not cosine. We L2-normalize the
embeddings first so Euclidean on the unit sphere is monotone with
cosine distance — gives the same neighborhoods for our purposes.

Cluster ids are fresh ULIDs per run. The ``Memory.cluster_id`` column
points at the *latest* P2 assignment; older cluster_ids are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ulid


@dataclass
class ClusterAssignment:
    cluster_id: str | None
    similarity: float | None
    is_representative: bool


def cluster_embeddings(
    embeddings: dict[str, np.ndarray],
    *,
    min_cluster_size: int = 3,
) -> dict[str, ClusterAssignment]:
    """Run HDBSCAN and return ``{memory_id: ClusterAssignment}``.

    Points labelled as noise (``-1`` in sklearn's output) get
    ``cluster_id=None``. Points in a real cluster get a run-local ULID
    for ``cluster_id`` and their HDBSCAN membership probability as
    ``similarity``. The highest-probability member of each cluster is
    marked ``is_representative=True``.

    Returns an empty dict if the input has fewer points than
    ``min_cluster_size`` (HDBSCAN would error on that).
    """

    if not embeddings:
        return {}

    ids = list(embeddings.keys())
    if len(ids) < min_cluster_size:
        # Too few points for HDBSCAN; treat all as noise.
        return {mid: ClusterAssignment(None, None, False) for mid in ids}

    matrix = np.stack([embeddings[i] for i in ids])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)

    # Lazy import so test environments without sklearn can still import
    # the module (they just can't call this function).
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean", copy=True
    )
    labels = clusterer.fit_predict(normalized)
    probs = clusterer.probabilities_

    label_to_cluster_id: dict[int, str] = {}
    out: dict[str, ClusterAssignment] = {}
    for i, mid in enumerate(ids):
        label = int(labels[i])
        if label < 0:
            out[mid] = ClusterAssignment(None, None, False)
            continue
        if label not in label_to_cluster_id:
            label_to_cluster_id[label] = str(ulid.new())
        out[mid] = ClusterAssignment(
            cluster_id=label_to_cluster_id[label],
            similarity=float(probs[i]),
            is_representative=False,
        )

    # One representative per cluster: the highest-probability member.
    best_per_cluster: dict[str, tuple[str, float]] = {}
    for mid, assignment in out.items():
        if assignment.cluster_id is None:
            continue
        key = assignment.cluster_id
        sim = assignment.similarity or 0.0
        if key not in best_per_cluster or sim > best_per_cluster[key][1]:
            best_per_cluster[key] = (mid, sim)
    for rep_id, _ in best_per_cluster.values():
        out[rep_id].is_representative = True

    return out


def group_by_cluster(
    assignments: dict[str, ClusterAssignment],
) -> dict[str, list[str]]:
    """Return ``{cluster_id: [memory_ids]}`` for non-noise assignments."""

    out: dict[str, list[str]] = {}
    for mid, a in assignments.items():
        if a.cluster_id is None:
            continue
        out.setdefault(a.cluster_id, []).append(mid)
    return out
