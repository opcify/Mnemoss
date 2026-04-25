"""Approximate-nearest-neighbor index for Mnemoss (Phase 1.2).

``sqlite-vec`` ships a vec0 virtual table that lets us store embeddings
in SQLite — but its ``MATCH`` operator does a **linear scan** internally,
so query latency grows O(N) with corpus size. For a 10K-row workspace
that's ~30ms per recall; for 100K it's ~300ms. That's the bottleneck
behind Mnemoss being 2-3x slower than ``raw_stack`` on the scale
benchmark (see docs/ROOT_CAUSE.md Phase 1.1).

This module wraps ``hnswlib`` to provide **O(log N)** vector recall via
Hierarchical Navigable Small World graphs. The index lives in memory —
on workspace open we rehydrate from the ``memory_vec`` table, on write
we add to both SQLite (persistence) and the HNSW graph (fast query).
Delete is soft: ``mark_deleted`` so the graph structure stays intact
and queries skip tombstoned labels.

Design choices
--------------

- **In-memory, not persisted.** HNSW indices do serialize to disk via
  ``save_index``, but rebuild from ``memory_vec`` is fast (O(N log N) in
  the number of add_items calls) and keeps the on-disk format pinned to
  SQLite alone. Also sidesteps version-skew questions if ``hnswlib``
  wire format changes.
- **Cosine space.** Mnemoss stores already-normalized embeddings per
  the embedder contract; hnswlib's ``space='cosine'`` treats
  ``distance = 1 - cos_sim`` — we convert back to similarity before
  returning. Matches the sqlite-vec ``distance_metric=cosine`` setting.
- **String id → int label map.** hnswlib requires integer labels; we
  keep a bidirectional dict in Python. Labels are monotonically assigned
  from a counter, not reused after delete, so a deleted memory's label
  never refers to a different memory later.
- **Optional dependency.** ``hnswlib`` lives behind the ``[ann]``
  extra. ``SQLiteBackend`` falls back to ``sqlite-vec``'s linear scan
  if the import fails — the library still works at the cost of O(N)
  vec_search. Install with ``pip install mnemoss[ann]`` for production.

Accuracy vs speed
-----------------

HNSW is *approximate*: a k-NN query may miss a small number of true
neighbors. With default config (``M=16, ef_construction=200, ef=50``),
recall@10 versus an exhaustive scan is typically >0.99 on the
dimensions Mnemoss ships with (384 / 1536 / 3072). Callers that need
exactness can force ``ef = max_elements`` at query time, which
degenerates to a linear scan. Most agent workloads don't need that.

Thread safety
-------------

Mnemoss funnels every DB call through a single worker thread (see
``SQLiteBackend._executor``). We take the same approach: every mutating
method here is only called on that worker thread, so we don't serialize
separately. Read queries from other threads would need explicit locking
if we ever widened the concurrency model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import hnswlib  # type: ignore[import-untyped]

    HNSWLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - tested in dev env
    HNSWLIB_AVAILABLE = False


@dataclass
class ANNIndexStats:
    """Snapshot of index size, useful for status() and tests."""

    n_elements: int
    n_deleted: int
    capacity: int


class ANNIndex:
    """HNSW index over workspace embeddings.

    Parameters
    ----------
    dim:
        Embedding dimension. Must match the embedder's output and the
        ``workspace_meta.embedding_dim`` pin.
    initial_capacity:
        Starting ``max_elements``. Grows by doubling when exceeded.
    m:
        HNSW ``M`` — "bidirectional links per node." Higher = better
        recall, more memory. 16 is the library default and a sensible
        balance. (Name is lowercase in Python, maps to hnswlib's uppercase
        ``M`` keyword arg internally.)
    ef_construction:
        Candidate list size during insert. Higher = better index quality,
        slower build. 200 is conservative and matches most published
        benchmarks.
    ef_query:
        Default query-time candidate list size. Can be overridden per
        query. Higher = better recall, slower query. 50 keeps recall@10
        >= 0.99 on embedders we ship with.
    """

    def __init__(
        self,
        *,
        dim: int,
        initial_capacity: int = 1024,
        m: int = 16,  # HNSW "M" parameter — lowercase to satisfy N803.
        ef_construction: int = 200,
        ef_query: int = 50,
    ) -> None:
        if not HNSWLIB_AVAILABLE:
            raise RuntimeError(
                "hnswlib is not installed. Install with `pip install mnemoss[ann]` "
                "to enable the ANN index, or unset SQLiteBackend's ann argument "
                "to fall back to sqlite-vec's linear scan."
            )
        self._dim = dim
        self._m = m
        self._ef_construction = ef_construction
        self._ef_query = ef_query
        self._capacity = initial_capacity

        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=initial_capacity,
            ef_construction=ef_construction,
            M=m,
        )
        self._index.set_ef(ef_query)

        self._id_to_label: dict[str, int] = {}
        self._label_to_id: dict[int, str] = {}
        self._next_label = 0
        self._n_deleted = 0

    # ─── writes ─────────────────────────────────────────────────────

    def add(self, memory_id: str, embedding: np.ndarray) -> None:
        """Insert a memory into the index. No-op if already present."""

        if memory_id in self._id_to_label:
            return
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.shape != (self._dim,):
            raise ValueError(
                f"ANNIndex.add: embedding shape {emb.shape} != ({self._dim},)"
            )
        # Grow capacity before adding if we're at the ceiling. hnswlib's
        # resize_index doubles cost for now but keeps the graph valid.
        if self._next_label >= self._capacity:
            self._capacity *= 2
            self._index.resize_index(self._capacity)

        label = self._next_label
        self._next_label += 1
        self._index.add_items(emb.reshape(1, -1), np.array([label], dtype=np.int64))
        self._id_to_label[memory_id] = label
        self._label_to_id[label] = memory_id

    def add_batch(
        self, memory_ids: list[str], embeddings: np.ndarray
    ) -> None:
        """Insert many memories at once (used by workspace rehydrate).

        ``embeddings`` is an (N, dim) float32 array aligned with
        ``memory_ids``. Ids already present in the index are skipped
        with their corresponding embedding row dropped from the batch.
        """

        if len(memory_ids) != len(embeddings):
            raise ValueError(
                f"ANNIndex.add_batch: len(memory_ids)={len(memory_ids)} != "
                f"len(embeddings)={len(embeddings)}"
            )
        if len(memory_ids) == 0:
            return
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self._dim:
            raise ValueError(
                f"ANNIndex.add_batch: embeddings shape {emb.shape} "
                f"!= (N, {self._dim})"
            )

        # Filter out any ids already present (rehydrate on an
        # already-populated index should be a no-op on the duplicates).
        keep_mask = np.array(
            [mid not in self._id_to_label for mid in memory_ids], dtype=bool
        )
        if not keep_mask.any():
            return
        fresh_ids = [mid for mid, keep in zip(memory_ids, keep_mask, strict=True) if keep]
        fresh_emb = emb[keep_mask]

        # Ensure capacity — one resize round-trip for the whole batch.
        required = self._next_label + len(fresh_ids)
        if required > self._capacity:
            while self._capacity < required:
                self._capacity *= 2
            self._index.resize_index(self._capacity)

        labels = np.arange(
            self._next_label, self._next_label + len(fresh_ids), dtype=np.int64
        )
        self._index.add_items(fresh_emb, labels)
        for mid, label in zip(fresh_ids, labels.tolist(), strict=True):
            self._id_to_label[mid] = label
            self._label_to_id[label] = mid
        self._next_label += len(fresh_ids)

    def remove(self, memory_id: str) -> None:
        """Soft-delete ``memory_id``. No-op if not present.

        hnswlib's ``mark_deleted`` keeps the graph edges intact so
        neighbors of the deleted node remain reachable; the label is
        simply skipped during query. Labels are never reused.
        """

        label = self._id_to_label.pop(memory_id, None)
        if label is None:
            return
        self._label_to_id.pop(label, None)
        self._index.mark_deleted(label)
        self._n_deleted += 1

    # ─── reads ──────────────────────────────────────────────────────

    def query(
        self,
        embedding: np.ndarray,
        k: int,
        *,
        ef: int | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``[(memory_id, cosine_similarity)]`` sorted by similarity desc.

        Returns empty list if the index is empty. ``k`` is clamped to the
        number of live elements — asking for more than exist doesn't
        raise, just returns fewer results.
        """

        n_live = len(self._id_to_label)
        if n_live == 0:
            return []
        effective_k = min(k, n_live)
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.shape != (self._dim,):
            raise ValueError(
                f"ANNIndex.query: embedding shape {emb.shape} != ({self._dim},)"
            )
        if ef is not None:
            self._index.set_ef(ef)
        try:
            labels, distances = self._index.knn_query(
                emb.reshape(1, -1), k=effective_k
            )
        finally:
            if ef is not None:
                self._index.set_ef(self._ef_query)

        out: list[tuple[str, float]] = []
        for label, dist in zip(labels[0].tolist(), distances[0].tolist(), strict=True):
            mid = self._label_to_id.get(int(label))
            if mid is None:
                # Deleted concurrently between knn_query and iteration;
                # shouldn't happen with our thread model but belt-and-braces.
                continue
            similarity = 1.0 - float(dist)
            out.append((mid, similarity))
        return out

    # ─── introspection ──────────────────────────────────────────────

    def size(self) -> int:
        return len(self._id_to_label)

    def stats(self) -> ANNIndexStats:
        return ANNIndexStats(
            n_elements=len(self._id_to_label),
            n_deleted=self._n_deleted,
            capacity=self._capacity,
        )
