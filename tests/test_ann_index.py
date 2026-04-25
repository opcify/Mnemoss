"""Unit tests for the HNSW ANN index wrapper.

Covers add / remove / query semantics, shape validation, and the
rehydration-via-batch-add path used when opening an existing workspace.
"""

from __future__ import annotations

import numpy as np
import pytest

from mnemoss.store.ann_index import HNSWLIB_AVAILABLE, ANNIndex

pytestmark = pytest.mark.skipif(
    not HNSWLIB_AVAILABLE,
    reason="hnswlib not installed; install with `pip install mnemoss[ann]`",
)


def _vec(values: list[float], dim: int = 4) -> np.ndarray:
    """Normalize a small vector to unit length for cosine tests."""

    a = np.array(values, dtype=np.float32)
    assert a.shape == (dim,)
    return a / np.linalg.norm(a)


def test_empty_query_returns_empty_list() -> None:
    idx = ANNIndex(dim=4)
    assert idx.query(_vec([1.0, 0.0, 0.0, 0.0]), k=5) == []


def test_add_then_query_returns_similarity_sorted() -> None:
    idx = ANNIndex(dim=4)
    idx.add("a", _vec([1.0, 0.0, 0.0, 0.0]))
    idx.add("b", _vec([0.9, 0.1, 0.0, 0.0]))  # nearby
    idx.add("c", _vec([0.0, 0.0, 1.0, 0.0]))  # far

    results = idx.query(_vec([1.0, 0.0, 0.0, 0.0]), k=3)

    assert len(results) == 3
    ids = [mid for mid, _ in results]
    assert ids[0] == "a"
    assert ids[1] == "b"
    assert ids[2] == "c"
    # Similarities sort descending.
    sims = [s for _, s in results]
    assert sims[0] >= sims[1] >= sims[2]


def test_query_k_is_clamped_to_live_size() -> None:
    idx = ANNIndex(dim=4)
    idx.add("a", _vec([1.0, 0.0, 0.0, 0.0]))
    idx.add("b", _vec([0.0, 1.0, 0.0, 0.0]))

    results = idx.query(_vec([1.0, 0.0, 0.0, 0.0]), k=10)
    assert len(results) == 2


def test_add_duplicate_id_is_noop() -> None:
    idx = ANNIndex(dim=4)
    idx.add("a", _vec([1.0, 0.0, 0.0, 0.0]))
    idx.add("a", _vec([0.0, 0.0, 0.0, 1.0]))  # different embedding, same id
    assert idx.size() == 1


def test_wrong_shape_raises() -> None:
    idx = ANNIndex(dim=4)
    with pytest.raises(ValueError, match="shape"):
        idx.add("a", np.array([1.0, 0.0], dtype=np.float32))


def test_remove_excludes_from_future_queries() -> None:
    idx = ANNIndex(dim=4)
    idx.add("a", _vec([1.0, 0.0, 0.0, 0.0]))
    idx.add("b", _vec([0.9, 0.1, 0.0, 0.0]))
    idx.add("c", _vec([0.0, 0.0, 1.0, 0.0]))

    idx.remove("a")

    results = idx.query(_vec([1.0, 0.0, 0.0, 0.0]), k=5)
    ids = [mid for mid, _ in results]
    assert "a" not in ids
    assert ids[0] == "b"
    assert idx.size() == 2
    assert idx.stats().n_deleted == 1


def test_remove_missing_id_is_noop() -> None:
    idx = ANNIndex(dim=4)
    idx.add("a", _vec([1.0, 0.0, 0.0, 0.0]))
    idx.remove("does-not-exist")
    assert idx.size() == 1


def test_add_batch_matches_individual_adds() -> None:
    dim = 8
    rng = np.random.default_rng(seed=42)
    vectors = rng.standard_normal((20, dim)).astype(np.float32)
    # Normalize for cosine stability.
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"m{i}" for i in range(20)]

    batch_idx = ANNIndex(dim=dim)
    batch_idx.add_batch(ids, vectors)

    single_idx = ANNIndex(dim=dim)
    for mid, vec in zip(ids, vectors, strict=True):
        single_idx.add(mid, vec)

    query = vectors[0]
    batch_hits = [mid for mid, _ in batch_idx.query(query, k=10)]
    single_hits = [mid for mid, _ in single_idx.query(query, k=10)]
    assert batch_hits == single_hits


def test_add_batch_skips_duplicates() -> None:
    idx = ANNIndex(dim=4)
    idx.add("a", _vec([1.0, 0.0, 0.0, 0.0]))

    idx.add_batch(
        ["a", "b"],
        np.array(
            [_vec([0.5, 0.5, 0.0, 0.0]), _vec([0.0, 1.0, 0.0, 0.0])],
            dtype=np.float32,
        ),
    )

    # "a" was not replaced — query at its original direction still finds it.
    results = idx.query(_vec([1.0, 0.0, 0.0, 0.0]), k=2)
    ids = [mid for mid, _ in results]
    assert "a" in ids
    assert idx.size() == 2


def test_capacity_grows_on_overflow() -> None:
    # Start with capacity 4 and add 10 to exercise the resize path.
    idx = ANNIndex(dim=4, initial_capacity=4)
    for i in range(10):
        v = np.zeros(4, dtype=np.float32)
        v[i % 4] = 1.0
        idx.add(f"m{i}", v)
    assert idx.size() == 10
    # Capacity has grown (4 → 8 → 16).
    assert idx.stats().capacity >= 10


def test_approximate_recall_matches_exact_on_random_corpus() -> None:
    """Default HNSW config should hit >=0.98 recall@10 vs exhaustive NN."""

    dim = 64
    n_corpus = 500
    n_queries = 20
    k = 10
    rng = np.random.default_rng(seed=7)

    corpus = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    idx = ANNIndex(dim=dim, initial_capacity=n_corpus)
    idx.add_batch([f"m{i}" for i in range(n_corpus)], corpus)

    # Exhaustive top-k via numpy.
    sims = corpus @ queries.T  # (n_corpus, n_queries)
    exhaustive_topk: list[set[int]] = []
    for q in range(n_queries):
        topk = np.argpartition(-sims[:, q], k)[:k]
        exhaustive_topk.append(set(int(i) for i in topk))

    total_overlap = 0
    for q in range(n_queries):
        results = idx.query(queries[q], k=k)
        ann_ids = {int(mid[1:]) for mid, _ in results}
        total_overlap += len(ann_ids & exhaustive_topk[q])

    recall = total_overlap / (n_queries * k)
    assert recall >= 0.98, f"HNSW recall@{k} = {recall:.3f} below 0.98 floor"
