"""Smoke tests for ``bench/backends/raw_stack_backend.py``.

``RawStackBackend`` is the baseline the blog post's Chart 1 compares
Mnemoss against — the "markdown + SQLite + embedding" stack every
builder has either built or thought about building. These tests cover
the protocol contract; the recall-quality comparison happens in the
full launch-comparison harness with a real embedder.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from bench.backends import MemoryBackend, RecallHit
from bench.backends.raw_stack_backend import RawStackBackend, _normalize
from mnemoss import FakeEmbedder

# ─── contract ──────────────────────────────────────────────────────


async def test_satisfies_memory_backend_protocol() -> None:
    async with RawStackBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        assert isinstance(be, MemoryBackend)
        assert be.backend_id == "raw_stack"


# ─── roundtrip ─────────────────────────────────────────────────────


async def test_observe_then_recall_returns_the_memory() -> None:
    async with RawStackBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        mid = await be.observe("alice joined the quarterly kickoff", ts=0.0)
        assert mid
        hits = await be.recall("alice joined the quarterly kickoff", k=5)
        assert hits
        assert hits[0].memory_id == mid
        assert hits[0].rank == 1
        # Self-match cosine with FakeEmbedder's normalized hash vectors
        # must be exactly 1.0 (same input → identical vector → dot = 1).
        assert hits[0].score is not None
        assert abs(hits[0].score - 1.0) < 1e-5


async def test_ranks_are_contiguous_and_1_indexed() -> None:
    async with RawStackBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        for i in range(5):
            await be.observe(f"memory {i}", ts=float(i))
        hits = await be.recall("memory 0", k=5)
        assert len(hits) == 5
        assert [h.rank for h in hits] == [1, 2, 3, 4, 5]


async def test_recall_on_empty_workspace_returns_empty_list() -> None:
    async with RawStackBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        hits = await be.recall("anything", k=10)
        assert hits == []


# ─── cosine ordering ──────────────────────────────────────────────


async def test_recall_scores_are_descending_cosine() -> None:
    """The ranked list must be sorted by descending cosine similarity."""

    async with RawStackBackend(embedding_model=FakeEmbedder(dim=32)) as be:
        await be.observe("the quick brown fox", ts=0.0)
        await be.observe("the lazy dog sleeps", ts=1.0)
        await be.observe("quantum entanglement", ts=2.0)
        hits = await be.recall("the quick brown fox", k=3)
        assert len(hits) == 3
        scores = [h.score for h in hits]
        # strict=False because scores[1:] is shorter by construction.
        for prev, cur in zip(scores, scores[1:], strict=False):
            assert prev is not None and cur is not None
            assert prev >= cur, f"scores not monotonically decreasing: {scores}"


async def test_top_hit_has_higher_score_than_tail() -> None:
    async with RawStackBackend(embedding_model=FakeEmbedder(dim=32)) as be:
        target_id = await be.observe("meeting with alice on thursday", ts=0.0)
        for i in range(5):
            await be.observe(f"unrelated topic {i}", ts=float(i + 1))
        hits = await be.recall("meeting with alice on thursday", k=6)
        assert hits[0].memory_id == target_id
        assert hits[0].score is not None
        assert hits[-1].score is not None
        assert hits[0].score > hits[-1].score


# ─── markdown log contents ────────────────────────────────────────


async def test_markdown_log_records_every_observation() -> None:
    """The markdown log is part of the baseline's identity — every
    observe must land a line in it."""

    async with RawStackBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        await be.observe("first memory", ts=0.0)
        await be.observe("second memory", ts=123.0)
        body = be._md_path.read_text()
        lines = [line for line in body.splitlines() if line.strip()]
        assert len(lines) == 2
        assert "first memory" in lines[0]
        assert "second memory" in lines[1]
        # ts renders as integer seconds for readability.
        assert "[0]" in lines[0]
        assert "[123]" in lines[1]


async def test_markdown_log_is_human_readable_append_only() -> None:
    """Pattern is `- [ts] text` per line. No JSON, no escaping — it's
    meant to look like MEMORY.md."""

    async with RawStackBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        await be.observe("alice: hey, quick question", ts=100.0)
        body = be._md_path.read_text()
        # Leading dash + space + bracketed ts + space + text.
        assert body.startswith("- [100]")
        assert "alice: hey, quick question" in body


# ─── sqlite side persists vectors ─────────────────────────────────


async def test_each_observe_inserts_one_sqlite_row() -> None:
    async with RawStackBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        for i in range(4):
            await be.observe(f"memory {i}", ts=float(i))
        # Open a separate read-only handle to prove the row is persisted.
        conn = sqlite3.connect(str(be._db_path))
        n = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        assert n == 4


async def test_stored_vectors_are_unit_normalized() -> None:
    """``RawStackBackend`` stores L2-normalized vectors so recall's
    dot product == cosine similarity."""

    async with RawStackBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        await be.observe("hello world", ts=0.0)
        conn = sqlite3.connect(str(be._db_path))
        row = conn.execute("SELECT vec FROM memories LIMIT 1").fetchone()
        conn.close()
        vec = np.frombuffer(row[0], dtype=np.float32)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


# ─── cleanup ──────────────────────────────────────────────────────


async def test_tempdir_cleanup_on_close() -> None:
    be = RawStackBackend(embedding_model=FakeEmbedder(dim=16))
    td = be._tempdir
    assert td.exists()
    await be.observe("memory", ts=0.0)
    await be.close()
    assert not td.exists()


async def test_close_is_idempotent() -> None:
    be = RawStackBackend(embedding_model=FakeEmbedder(dim=16))
    await be.observe("memory", ts=0.0)
    await be.close()
    await be.close()  # must not raise


# ─── tiny unit for the normalize helper ───────────────────────────


def test_normalize_unit_vector_is_identity() -> None:
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out = _normalize(v)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6


def test_normalize_scales_to_unit() -> None:
    v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    out = _normalize(v)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6


def test_normalize_zero_vector_returns_zero_vector() -> None:
    v = np.zeros(5, dtype=np.float32)
    out = _normalize(v)
    assert np.allclose(out, 0.0)


# ─── RecallHit shape ──────────────────────────────────────────────


async def test_recall_hit_fields_are_populated() -> None:
    async with RawStackBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        await be.observe("alpha", ts=0.0)
        hits = await be.recall("alpha", k=1)
        assert len(hits) == 1
        hit = hits[0]
        assert isinstance(hit, RecallHit)
        assert isinstance(hit.memory_id, str) and hit.memory_id
        assert hit.rank == 1
        assert hit.score is not None
        assert isinstance(hit.score, float)


# ─── instance isolation ──────────────────────────────────────────


async def test_two_instances_do_not_share_data() -> None:
    be1 = RawStackBackend(embedding_model=FakeEmbedder(dim=16))
    be2 = RawStackBackend(embedding_model=FakeEmbedder(dim=16))
    try:
        assert be1._tempdir != be2._tempdir
        id1 = await be1.observe("private to instance 1", ts=0.0)
        id2 = await be2.observe("private to instance 2", ts=0.0)
        hits1 = await be1.recall("private to instance 1", k=5)
        hits2 = await be2.recall("private to instance 2", k=5)
        assert hits1 and hits1[0].memory_id == id1
        assert hits2 and hits2[0].memory_id == id2
        ids_seen_by_be1 = {h.memory_id for h in hits1}
        assert id2 not in ids_seen_by_be1
    finally:
        await be1.close()
        await be2.close()
