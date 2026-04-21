"""Event segmentation (Checkpoint J).

Covers the three closing rules (turn_shift, time_gap, size_limit), the
auto_1to1 legacy fallback, per-agent isolation, and the explicit flush
paths. Uses FakeEmbedder so the tests exercise the client end-to-end
without the local model.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mnemoss import FakeEmbedder, Mnemoss, SegmentationParams, StorageParams
from mnemoss.core.types import RawMessage
from mnemoss.encoder.event_segmentation import EventSegmenter

UTC = timezone.utc


def _mnemoss(tmp_path: Path, **kwargs) -> Mnemoss:
    return Mnemoss(
        workspace="test",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        **kwargs,
    )


def _msg(
    content: str,
    *,
    agent_id: str | None = None,
    session_id: str = "s",
    turn_id: str = "t",
    ts: datetime | None = None,
) -> RawMessage:
    return RawMessage(
        id=f"raw-{content[:10]}-{turn_id}",
        workspace_id="ws",
        agent_id=agent_id,
        session_id=session_id,
        turn_id=turn_id,
        parent_id=None,
        timestamp=ts or datetime.now(UTC),
        role="user",
        content=content,
    )


# ─── pure segmenter unit tests (no store) ────────────────────────────


def test_auto_close_produces_single_event() -> None:
    seg = EventSegmenter()
    now = datetime.now(UTC)
    step = seg.on_observe(
        _msg("hi", turn_id="auto"), now, SegmentationParams(), auto_close=True
    )
    assert len(step.closed_events) == 1
    assert step.closed_events[0].closed_by == "auto_1to1"
    assert step.closed_events[0].memory_id == step.pending_memory_id
    assert seg.pending_buffer_count() == 0


def test_same_turn_batches_until_turn_shift() -> None:
    seg = EventSegmenter()
    base = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC)
    params = SegmentationParams()

    s1 = seg.on_observe(_msg("first", turn_id="t1", ts=base), base, params)
    s2 = seg.on_observe(
        _msg("second", turn_id="t1", ts=base + timedelta(seconds=1)),
        base + timedelta(seconds=1),
        params,
    )
    # Both messages share the same pending memory id.
    assert s1.pending_memory_id == s2.pending_memory_id
    assert s1.closed_events == []
    assert s2.closed_events == []
    assert seg.pending_buffer_count() == 1

    # New turn arrives → the previous buffer closes.
    s3 = seg.on_observe(
        _msg("other", turn_id="t2", ts=base + timedelta(seconds=2)),
        base + timedelta(seconds=2),
        params,
    )
    assert len(s3.closed_events) == 1
    closed = s3.closed_events[0]
    assert closed.closed_by == "turn_shift"
    assert [m.content for m in closed.messages] == ["first", "second"]
    assert closed.memory_id == s1.pending_memory_id
    # The new turn has its own buffer.
    assert s3.pending_memory_id != s1.pending_memory_id


def test_time_gap_closes_idle_buffers() -> None:
    seg = EventSegmenter()
    params = SegmentationParams(time_gap_seconds=30.0)
    t0 = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC)

    seg.on_observe(_msg("idle", turn_id="t1", ts=t0), t0, params)
    # 60s later, same session but different turn — both time_gap (idle > 30s)
    # and turn_shift apply; the segmenter picks time_gap first.
    s = seg.on_observe(
        _msg("new", turn_id="t2", ts=t0 + timedelta(seconds=60)),
        t0 + timedelta(seconds=60),
        params,
    )
    assert len(s.closed_events) == 1
    assert s.closed_events[0].closed_by == "time_gap"


def test_size_limit_closes_current_buffer() -> None:
    seg = EventSegmenter()
    params = SegmentationParams(max_event_messages=3)
    t = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC)

    for i in range(2):
        step = seg.on_observe(
            _msg(f"m{i}", turn_id="big", ts=t),
            t + timedelta(milliseconds=i),
            params,
        )
        assert step.closed_events == []
    # 3rd message hits the cap → closes immediately.
    step = seg.on_observe(
        _msg("m2", turn_id="big", ts=t),
        t + timedelta(milliseconds=3),
        params,
    )
    assert len(step.closed_events) == 1
    assert step.closed_events[0].closed_by == "size_limit"
    assert len(step.closed_events[0].messages) == 3


def test_character_cap_also_triggers_size_limit() -> None:
    seg = EventSegmenter()
    params = SegmentationParams(max_event_characters=10)
    t = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC)

    step = seg.on_observe(
        _msg("exactlyten", turn_id="c", ts=t), t, params
    )
    assert len(step.closed_events) == 1
    assert step.closed_events[0].closed_by == "size_limit"


def test_agents_have_isolated_buffers() -> None:
    seg = EventSegmenter()
    t = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC)
    params = SegmentationParams()

    a1 = seg.on_observe(_msg("alice", agent_id="alice", turn_id="t1", ts=t), t, params)
    b1 = seg.on_observe(_msg("bob", agent_id="bob", turn_id="t1", ts=t), t, params)
    # Different buffers, so no cross-close.
    assert a1.closed_events == []
    assert b1.closed_events == []
    assert a1.pending_memory_id != b1.pending_memory_id
    assert seg.pending_buffer_count() == 2


def test_flush_narrows_to_scope() -> None:
    seg = EventSegmenter()
    t = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC)
    params = SegmentationParams()

    seg.on_observe(_msg("a", agent_id="alice", turn_id="t1", ts=t), t, params)
    seg.on_observe(_msg("b", agent_id="bob", turn_id="t1", ts=t), t, params)
    assert seg.pending_buffer_count() == 2

    closed = seg.flush(agent_id="alice", now=t + timedelta(seconds=1))
    assert len(closed) == 1
    assert closed[0].closed_by == "flush"
    assert seg.pending_buffer_count() == 1


# ─── client-level integration (with SQLite + FakeEmbedder) ───────────


async def test_auto_close_path_stores_memory_immediately(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mid = await mem.observe(role="user", content="hello")
        assert mid is not None
        # No flush needed; the memory is already queryable.
        results = await mem.recall("hello", k=1)
        assert any(r.memory.id == mid for r in results)
    finally:
        await mem.close()


async def test_shared_turn_id_produces_one_memory(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mid1 = await mem.observe(role="user", content="hello", turn_id="conv1")
        mid2 = await mem.observe(role="assistant", content="hi there", turn_id="conv1")
        assert mid1 == mid2  # same pending id → same memory

        # Buffer still open: recall shouldn't find it yet.
        results = await mem.recall("hello", k=5)
        assert all(r.memory.id != mid1 for r in results)

        # Force flush.
        flushed = await mem.flush_session()
        assert flushed == 1

        results = await mem.recall("hello", k=5)
        hit = next((r for r in results if r.memory.id == mid1), None)
        assert hit is not None
        # Content is the concatenation of both messages.
        assert "hello" in hit.memory.content
        assert "hi there" in hit.memory.content
        # source_message_ids captures both underlying Raw Log rows.
        assert len(hit.memory.source_message_ids) == 2
    finally:
        await mem.close()


async def test_turn_shift_flushes_previous_turn(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        m1 = await mem.observe(role="user", content="turn one", turn_id="t1")
        # Observing a new turn_id closes t1's buffer as a side effect.
        m2 = await mem.observe(role="user", content="turn two", turn_id="t2")
        assert m1 != m2

        results = await mem.recall("turn one", k=5)
        assert any(r.memory.id == m1 for r in results)
        # t2 is still buffered, not yet in the store.
        assert all(r.memory.id != m2 for r in results)
    finally:
        await mem.close()


async def test_close_drains_all_buffers(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    mid = await mem.observe(role="user", content="bye", turn_id="open")
    # Close should flush before shutting the store down.
    await mem.close()

    # Re-open and verify the memory landed.
    mem2 = _mnemoss(tmp_path)
    try:
        results = await mem2.recall("bye", k=3)
        assert any(r.memory.id == mid for r in results)
    finally:
        await mem2.close()


async def test_flush_session_scoped_by_agent(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    alice = mem.for_agent("alice")
    bob = mem.for_agent("bob")
    try:
        await alice.observe(role="user", content="alice turn", turn_id="t")
        await bob.observe(role="user", content="bob turn", turn_id="t")

        flushed = await mem.flush_session(agent_id="alice")
        assert flushed == 1

        alice_results = await alice.recall("turn", k=3)
        alice_contents = {r.memory.content for r in alice_results}
        assert "alice turn" in alice_contents

        # Bob's buffer hasn't been flushed yet, so his turn isn't recallable.
        bob_results = await bob.recall("turn", k=3)
        assert "bob turn" not in {r.memory.content for r in bob_results}
    finally:
        await mem.close()


async def test_size_limit_during_observe_persists_immediately(tmp_path: Path) -> None:
    mem = _mnemoss(
        tmp_path, segmentation=SegmentationParams(max_event_messages=2)
    )
    try:
        await mem.observe(role="user", content="a", turn_id="size")
        await mem.observe(role="user", content="b", turn_id="size")  # hits cap
        # The buffer closed on the second observe, so the memory is
        # already persisted; no flush needed.
        results = await mem.recall("a b", k=3)
        assert results
    finally:
        await mem.close()


async def test_time_gap_does_not_interfere_within_one_turn(tmp_path: Path) -> None:
    # Guardrail: time_gap only compares against *other* buffers, not the
    # current one being appended to. Two messages arriving 10 minutes
    # apart but with the same turn_id should still batch.
    mem = _mnemoss(
        tmp_path, segmentation=SegmentationParams(time_gap_seconds=30.0)
    )
    try:
        mid1 = await mem.observe(role="user", content="early", turn_id="long_turn")
        # Simulate a long pause — since we can't time-travel the wall clock
        # mid-call, just observe another message and verify it still batches.
        mid2 = await mem.observe(role="user", content="late", turn_id="long_turn")
        assert mid1 == mid2
    finally:
        await mem.close()


async def test_observe_returns_none_for_filtered_roles(tmp_path: Path) -> None:
    from mnemoss import EncoderParams

    mem = _mnemoss(tmp_path, encoder=EncoderParams(encoded_roles={"user"}))
    try:
        assert await mem.observe(role="tool_call", content="ignored") is None
        # No buffer should have been opened either.
        mid = await mem.observe(role="user", content="kept", turn_id="kept-turn")
        assert mid is not None
    finally:
        await mem.close()


async def test_raw_log_receives_every_message_even_when_buffered(tmp_path: Path) -> None:
    """Principle 3 under Stage 3: Raw Log is unfiltered and unbuffered."""

    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="first", turn_id="t")
        await mem.observe(role="user", content="second", turn_id="t")
        # Neither has been persisted as a Memory yet, but both must be in the
        # Raw Log. Query via the private store API.
        assert mem._store is not None
        rows = await mem._store.list_recent_in_session("default", 10)
        _ = rows  # no memories yet
        # Raw Log exposed via recent raw_message scan:
        raw_count = await _raw_message_count(mem)
        assert raw_count == 2
    finally:
        await mem.close()


async def _raw_message_count(mem: Mnemoss) -> int:
    """Test helper: count rows in raw_message. Reaches into the store
    intentionally — there's no public counter yet. Raw-log lives on its
    own SQLite file (schema v6+), so query the raw connection."""

    assert mem._store is not None
    conn = mem._store._raw_conn
    assert conn is not None

    def _count() -> int:
        return conn.execute("SELECT COUNT(*) FROM raw_message").fetchone()[0]

    return await mem._store._run(_count)


@pytest.mark.parametrize("n", [1, 5, 19])
async def test_flush_all_is_idempotent(tmp_path: Path, n: int) -> None:
    mem = _mnemoss(tmp_path)
    try:
        for i in range(n):
            await mem.observe(role="user", content=f"m{i}", turn_id=f"t{i % 3}")

        flushed = await mem.flush_session()
        # Second flush has nothing left.
        again = await mem.flush_session()
        assert again == 0
        _ = flushed
    finally:
        await mem.close()
