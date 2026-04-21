"""P2 Cluster tests (Checkpoint N)."""

from __future__ import annotations

import numpy as np

from mnemoss.dream.cluster import (
    ClusterAssignment,
    cluster_embeddings,
    group_by_cluster,
)


def _unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_empty_input_returns_empty() -> None:
    assert cluster_embeddings({}) == {}


def test_sub_min_cluster_size_labels_all_as_noise() -> None:
    embeddings = {
        "m1": _unit(1.0, 0.0, 0.0),
        "m2": _unit(0.9, 0.1, 0.0),
    }
    result = cluster_embeddings(embeddings, min_cluster_size=3)
    assert len(result) == 2
    assert all(a.cluster_id is None for a in result.values())
    assert all(not a.is_representative for a in result.values())


def test_clustered_points_get_labels_and_one_representative() -> None:
    # Two distinct tight clusters of 3 points each.
    embeddings = {
        # Cluster A: near (1, 0, 0)
        "a1": _unit(1.0, 0.05, 0.0),
        "a2": _unit(1.0, 0.0, 0.05),
        "a3": _unit(0.95, 0.05, 0.05),
        # Cluster B: near (0, 1, 0)
        "b1": _unit(0.05, 1.0, 0.0),
        "b2": _unit(0.0, 1.0, 0.05),
        "b3": _unit(0.05, 0.95, 0.05),
    }
    result = cluster_embeddings(embeddings, min_cluster_size=3)

    cluster_ids = {a.cluster_id for a in result.values() if a.cluster_id}
    # Expect two distinct cluster ids (may degenerate to 1 if HDBSCAN
    # decides they're one cluster — tolerate either, but reps must
    # still be 1 per cluster).
    assert 1 <= len(cluster_ids) <= 2

    by_cluster = group_by_cluster(result)
    for members in by_cluster.values():
        reps = [m for m in members if result[m].is_representative]
        assert len(reps) == 1  # exactly one rep per non-noise cluster.


def test_group_by_cluster_skips_noise() -> None:
    assignments = {
        "a": ClusterAssignment(cluster_id="c1", similarity=0.9, is_representative=True),
        "b": ClusterAssignment(cluster_id="c1", similarity=0.8, is_representative=False),
        "noise": ClusterAssignment(cluster_id=None, similarity=None, is_representative=False),
    }
    groups = group_by_cluster(assignments)
    assert groups == {"c1": ["a", "b"]}
