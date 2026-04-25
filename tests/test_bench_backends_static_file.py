"""Smoke tests for ``bench/backends/static_file_backend.py``.

The StaticFileBackend is the honest baseline for Chart 1 in the launch
post — what Hermes (``MEMORY.md``) and Claude Code (``CLAUDE.md``) ship
with out of the box. These tests cover the behavior contract; the
quality comparison against MnemossBackend happens in the full harness,
not here.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.backends import MemoryBackend, RecallHit
from bench.backends.static_file_backend import StaticFileBackend

# ─── contract ──────────────────────────────────────────────────────


async def test_satisfies_memory_backend_protocol() -> None:
    async with StaticFileBackend() as be:
        assert isinstance(be, MemoryBackend)
        assert be.backend_id == "static_file"


# ─── roundtrip ─────────────────────────────────────────────────────


async def test_observe_then_recall_returns_the_memory() -> None:
    async with StaticFileBackend() as be:
        mid = await be.observe("alice joined the quarterly kickoff", ts=0.0)
        assert mid, "observe returns a non-empty id"
        hits = await be.recall("alice kickoff", k=5)
        assert hits
        assert hits[0].memory_id == mid
        assert hits[0].rank == 1
        assert hits[0].score == 2.0  # 'alice' + 'kickoff'


async def test_recall_drops_zero_score_memories() -> None:
    """A keyword grep returns nothing when no tokens match."""

    async with StaticFileBackend() as be:
        await be.observe("alice meeting today", ts=0.0)
        await be.observe("bob coffee order", ts=1.0)
        hits = await be.recall("quantum cryptography", k=10)
        assert hits == []


async def test_ranks_are_contiguous_and_1_indexed() -> None:
    async with StaticFileBackend() as be:
        await be.observe("alpha beta gamma", ts=0.0)
        await be.observe("alpha delta", ts=1.0)
        await be.observe("epsilon", ts=2.0)
        hits = await be.recall("alpha beta", k=5)
        assert len(hits) == 2
        assert [h.rank for h in hits] == [1, 2]


# ─── scoring: token overlap + recency tiebreak ─────────────────────


async def test_higher_overlap_beats_lower_overlap() -> None:
    async with StaticFileBackend() as be:
        id_two = await be.observe("alice meeting today", ts=0.0)  # overlap 2
        id_one = await be.observe("bob meeting", ts=1.0)  # overlap 1
        hits = await be.recall("alice meeting", k=5)
        assert hits[0].memory_id == id_two
        assert hits[1].memory_id == id_one
        assert hits[0].score == 2.0
        assert hits[1].score == 1.0


async def test_tied_overlap_breaks_by_recency_newer_first() -> None:
    """Two memories with identical token overlap — newer one ranks first.
    This matches 'read the file from the end' style recall."""

    async with StaticFileBackend() as be:
        older = await be.observe("alice meeting", ts=0.0)
        newer = await be.observe("alice meeting", ts=1000.0)
        hits = await be.recall("alice meeting", k=5)
        assert len(hits) == 2
        assert hits[0].memory_id == newer
        assert hits[1].memory_id == older


# ─── tokenizer edge cases ──────────────────────────────────────────


async def test_case_insensitive_token_match() -> None:
    async with StaticFileBackend() as be:
        mid = await be.observe("ALICE at the KICKOFF", ts=0.0)
        hits = await be.recall("alice kickoff", k=5)
        assert hits
        assert hits[0].memory_id == mid


async def test_punctuation_doesnt_create_ghost_tokens() -> None:
    async with StaticFileBackend() as be:
        mid = await be.observe("alice, joined: the kickoff!", ts=0.0)
        hits = await be.recall("alice kickoff", k=5)
        assert hits
        assert hits[0].memory_id == mid
        assert hits[0].score == 2.0  # just 'alice' + 'kickoff', nothing else


async def test_empty_query_returns_empty() -> None:
    async with StaticFileBackend() as be:
        await be.observe("alice", ts=0.0)
        assert await be.recall("", k=5) == []
        assert await be.recall("   ", k=5) == []
        assert await be.recall("!!!", k=5) == []


# ─── persistence shape (file actually exists and is append-only) ───


async def test_observe_writes_jsonl_line_per_call() -> None:
    async with StaticFileBackend() as be:
        await be.observe("first", ts=0.0)
        await be.observe("second", ts=1.0)
        lines = be._path.read_text().strip().splitlines()
        assert len(lines) == 2
        first, second = [json.loads(line) for line in lines]
        assert first["text"] == "first"
        assert second["text"] == "second"
        assert first["ts"] == 0.0
        assert second["ts"] == 1.0


async def test_custom_path_is_honored() -> None:
    """Supplying path= uses that file (and the directory isn't deleted
    on close — only auto-allocated tempdirs are)."""

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "subdir" / "memory.jsonl"
        be = StaticFileBackend(path=target)
        try:
            await be.observe("hello", ts=0.0)
            assert target.exists()
            hits = await be.recall("hello", k=5)
            assert hits
        finally:
            await be.close()
        # Custom path: file still exists after close (caller-owned).
        assert target.exists()


# ─── cleanup ───────────────────────────────────────────────────────


async def test_tempdir_cleanup_on_close() -> None:
    be = StaticFileBackend()
    td = be._tempdir
    assert td is not None
    assert td.exists()
    await be.observe("memory", ts=0.0)
    await be.close()
    assert not td.exists()


async def test_close_is_idempotent() -> None:
    async with StaticFileBackend() as be:
        await be.observe("m", ts=0.0)
    # __aexit__ already called close(); a second explicit call must not raise.
    # Note: `async with` closed `be`, so this is the second call.
    await be.close()


# ─── RecallHit shape ───────────────────────────────────────────────


async def test_recall_hit_fields() -> None:
    async with StaticFileBackend() as be:
        await be.observe("alpha beta", ts=0.0)
        hits = await be.recall("alpha", k=1)
        assert len(hits) == 1
        hit = hits[0]
        assert isinstance(hit, RecallHit)
        assert hit.rank == 1
        assert isinstance(hit.score, float)
        assert hit.score == 1.0
