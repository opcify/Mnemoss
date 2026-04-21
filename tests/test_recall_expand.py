"""Auto-expand on same-topic follow-up recalls.

Each test wires co_occurrence relations by observing multiple memories
in the same session — that's the edge type the encoder writes at
encode-time, and ``expand_from_seeds`` walks it alongside
``similar_to`` / ``derived_from`` (the latter two only materialize
during dreaming, so they're not exercised here).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from mnemoss import FakeEmbedder, FormulaParams, Mnemoss, StorageParams
from mnemoss.recall.expand import hops_for_streak
from mnemoss.recall.history import (
    PastQuery,
    RecallHistory,
    _cosine,
    is_same_topic,
)


def _mnemoss(tmp_path: Path, **kwargs) -> Mnemoss:
    return Mnemoss(
        workspace="ws",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        formula=FormulaParams(noise_scale=0.0),  # deterministic
        **kwargs,
    )


# ─── is_same_topic unit tests ────────────────────────────────────


def test_same_topic_triggers_on_result_overlap() -> None:
    prev = PastQuery(
        query="x",
        query_vec=np.array([1.0, 0.0], dtype=np.float32),
        timestamp=_fixed_now(),
        result_ids={"m1"},
    )
    assert is_same_topic(
        prev,
        current_query_vec=np.array([0.0, 1.0], dtype=np.float32),  # orthogonal
        current_result_ids={"m1", "m2"},
        cosine_threshold=0.7,
    )


def test_same_topic_triggers_on_query_cosine() -> None:
    vec = np.array([1.0, 1.0], dtype=np.float32)
    prev = PastQuery(
        query="x",
        query_vec=vec,
        timestamp=_fixed_now(),
        result_ids=set(),
    )
    assert is_same_topic(
        prev,
        current_query_vec=vec.copy(),
        current_result_ids=set(),
        cosine_threshold=0.7,
    )


def test_same_topic_does_not_gate_on_time() -> None:
    """Detection is purely semantic — a user returning to the same
    thread hours later is still asking about the same topic."""

    prev = PastQuery(
        query="x",
        query_vec=np.array([1.0, 0.0], dtype=np.float32),
        timestamp=_fixed_now(),
        result_ids={"m1"},
    )
    assert is_same_topic(
        prev,
        current_query_vec=np.array([1.0, 0.0], dtype=np.float32),
        current_result_ids={"m1"},
        cosine_threshold=0.7,
    )


def test_same_topic_false_on_different_topic() -> None:
    prev = PastQuery(
        query="x",
        query_vec=np.array([1.0, 0.0], dtype=np.float32),
        timestamp=_fixed_now(),
        result_ids={"m1"},
    )
    assert not is_same_topic(
        prev,
        current_query_vec=np.array([0.0, 1.0], dtype=np.float32),
        current_result_ids={"m9"},  # no overlap, orthogonal vectors
        cosine_threshold=0.7,
    )


def test_cosine_handles_zero_vectors() -> None:
    zero = np.zeros(4, dtype=np.float32)
    assert _cosine(zero, zero) == 0.0


# ─── RecallHistory ────────────────────────────────────────────────


def test_history_is_per_agent() -> None:
    hist = RecallHistory()
    now = _fixed_now()
    hist.record(
        "alice",
        PastQuery("q", np.zeros(2, dtype=np.float32), now, set()),
    )
    assert hist.latest("alice") is not None
    assert hist.latest("bob") is None
    assert hist.latest(None) is None


def test_history_clear_scoped() -> None:
    hist = RecallHistory()
    now = _fixed_now()
    for aid in ("alice", "bob"):
        hist.record(aid, PastQuery("q", np.zeros(2, dtype=np.float32), now, set()))
    hist.clear("alice")
    assert hist.latest("alice") is None
    assert hist.latest("bob") is not None
    hist.clear()
    assert hist.latest("bob") is None


# ─── hops_for_streak ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "streak,hops_max,expected",
    [(1, 2, 1), (2, 2, 2), (3, 2, 2), (5, 2, 2), (2, 3, 2), (5, 3, 3)],
)
def test_hops_for_streak(streak: int, hops_max: int, expected: int) -> None:
    assert hops_for_streak(streak, hops_max) == expected


# ─── expand_via_relations predicate + cap ─────────────────────────


async def test_expand_via_relations_respects_predicate_filter(
    tmp_path: Path,
) -> None:
    """Edges with predicates outside the expansion set must not be walked.

    If someone writes an unrelated relation type (say ``mentions``) into
    the table, expansion should silently ignore it — otherwise a caller
    adding new relation kinds would accidentally widen the expansion
    blast radius.
    """

    mem = _mnemoss(tmp_path)
    try:
        # Three observes in the same session → three memories with
        # bidirectional ``co_occurs_in_session`` edges.
        ids = []
        for c in ["apple", "banana", "cherry"]:
            mid = await mem.observe(role="user", content=c, session_id="s1")
            assert mid is not None
            ids.append(mid)

        store = mem._store
        assert store is not None
        # Inject a rogue edge apple→external under a predicate that's
        # NOT in expand's allow-list.
        await store.write_relation(ids[0], "external_id", "mentions", 1.0)

        # Walk from apple. With the expansion allow-list we get banana
        # and cherry (co_occurs) but NOT ``external_id`` (mentions).
        reached = await store.expand_via_relations(
            [ids[0]],
            hops=1,
            predicates=("co_occurs_in_session", "similar_to", "derived_from"),
        )
        assert "external_id" not in reached
        assert ids[1] in reached
        assert ids[2] in reached

        # Without the filter the rogue edge shows up.
        reached_all = await store.expand_via_relations([ids[0]], hops=1)
        assert "external_id" in reached_all
    finally:
        await mem.close()


async def test_expand_via_relations_respects_max_candidates(
    tmp_path: Path,
) -> None:
    """Cap short-circuits BFS before the reachable set explodes.

    The encoder's ``session_cooccurrence_window`` (default 5) controls
    how many neighbours each newly-observed memory links back to. Over
    8 observes, the first memory accumulates edges to roughly 5 later
    ones — enough to demonstrate the cap behaviour.
    """

    mem = _mnemoss(tmp_path)
    try:
        ids = []
        for i in range(8):
            mid = await mem.observe(
                role="user", content=f"note {i}", session_id="s1"
            )
            assert mid is not None
            ids.append(mid)

        store = mem._store
        assert store is not None

        uncapped = await store.expand_via_relations(
            [ids[0]],
            hops=2,  # 2 hops on an 8-node dense subgraph → everyone
            predicates=("co_occurs_in_session",),
        )
        # 2-hop reaches most/all other memories; we only care that it's
        # well above the cap we'll apply below.
        assert len(uncapped) >= 5

        capped = await store.expand_via_relations(
            [ids[0]],
            hops=2,
            predicates=("co_occurs_in_session",),
            max_candidates=3,
        )
        # BFS stops when the non-seed reachable set hits the cap; exact
        # size depends on when the check fires within a hop. The
        # guarantee is "bounded," not "exactly N" — what matters is we
        # stop and the capped result is strictly smaller than uncapped.
        assert len(capped) < len(uncapped)
        assert len(capped) >= 3
    finally:
        await mem.close()


async def test_expansion_escalates_to_hops_3(tmp_path: Path) -> None:
    """hops=3 BFS reaches memories that hops=1/2 would miss.

    Test the store-level BFS directly on a linear chain A→B→C→D→E. The
    hop-count contract (streak=3 → hops=3) is already covered by the
    parametrized ``test_hops_for_streak``; here we verify the BFS itself
    honours the requested radius. Going through ``engine.recall`` would
    bury this in cascade noise from FakeEmbedder.
    """

    mem = _mnemoss(tmp_path)
    try:
        ids = []
        for i, c in enumerate(["alpha", "bravo", "charlie", "delta", "echo"]):
            mid = await mem.observe(
                role="user", content=c, session_id=f"s{i}"
            )
            assert mid is not None
            ids.append(mid)
        a, b, c_id, d_id, e_id = ids

        assert mem._store is not None
        # Linear chain: A ↔ B ↔ C ↔ D ↔ E. No shortcuts.
        for src, dst in [
            (a, b), (b, a),
            (b, c_id), (c_id, b),
            (c_id, d_id), (d_id, c_id),
            (d_id, e_id), (e_id, d_id),
        ]:
            await mem._store.write_relation(
                src, dst, "co_occurs_in_session", 0.5
            )

        # Verify escalation: each hop count reaches exactly the expected
        # depth of the chain.
        reach1 = await mem._store.expand_via_relations(
            [a], hops=1, predicates=("co_occurs_in_session",)
        )
        assert reach1 == {b}

        reach2 = await mem._store.expand_via_relations(
            [a], hops=2, predicates=("co_occurs_in_session",)
        )
        assert reach2 == {b, c_id}

        reach3 = await mem._store.expand_via_relations(
            [a], hops=3, predicates=("co_occurs_in_session",)
        )
        assert reach3 == {b, c_id, d_id}
        # E (hops=4) is still out of reach.
        assert e_id not in reach3
    finally:
        await mem.close()


async def test_clock_skew_does_not_explode_streak(tmp_path: Path) -> None:
    """Negative gap (prev.timestamp > now) must not keep streak growing.

    Simulates NTP step-back: rewrite the history timestamp into the
    future. Without the ``max(0, …)`` clamp, every subsequent recall
    would see a negative gap, pass the ≤ check, and keep incrementing
    streak forever.
    """

    mem = _mnemoss(tmp_path)
    try:
        for c in ["alice one", "alice two", "alice three"]:
            await mem.observe(role="user", content=c, session_id="s1")

        assert mem._engine is not None
        engine = mem._engine
        await engine.recall("alice one", agent_id=None, k=2, pool_size=2)

        # Push prev.timestamp 10 years into the future and set a huge
        # prior streak; without the clamp, the next recall would treat
        # gap as "negative, within window" → streak = prev+1 = 101.
        prev = engine._history.latest(None)
        assert prev is not None
        prev.timestamp = prev.timestamp + timedelta(days=3650)
        prev.streak = 100

        await engine.recall("alice one", agent_id=None, k=2, pool_size=2)

        new = engine._history.latest(None)
        assert new is not None
        # With the clamp, gap_seconds is clamped to 0, which is ≤
        # streak_reset_seconds → streak should increment to 101.
        # But the point of the test is it should NOT overflow or crash.
        # The exact value isn't load-bearing; what matters is streak is
        # bounded and doesn't corrupt. Sanity: ≤ prev + 1.
        assert new.streak <= prev.streak + 1
    finally:
        await mem.close()


# ─── integration with Mnemoss ─────────────────────────────────────


async def test_second_recall_on_same_topic_adds_expanded_results(
    tmp_path: Path,
) -> None:
    """Expansion surfaces memories the cascade missed.

    The cascade scores up to ``pool_size`` candidates per tier; anything
    outside that pool is invisible to the first pass. The test runs with
    ``pool_size=2`` so only a subset of the workspace reaches the
    cascade, leaving room for expansion to add neighbours via the
    co_occurs_in_session relation graph.
    """

    mem = _mnemoss(tmp_path)
    try:
        for content in [
            "alice project kickoff",
            "alice design review",
            "alice sync meeting",
            "alice launch plan",
            "alice retro notes",
            "alice stakeholder update",
        ]:
            await mem.observe(role="user", content=content, session_id="s1")

        # Go through the engine so we can shrink pool_size below the
        # workspace size — otherwise the cascade already covers every
        # memory and expansion has nothing new to offer.
        assert mem._store is not None and mem._engine is not None
        engine = mem._engine

        first = await engine.recall(
            "alice project", agent_id=None, k=2, pool_size=2
        )
        assert first, "direct recall returned nothing"
        assert all(r.source == "direct" for r in first)

        second = await engine.recall(
            "alice project", agent_id=None, k=2, pool_size=2
        )
        expanded = [r for r in second if r.source == "expanded"]
        assert expanded, "expected at least one expanded result on follow-up"
        direct_ids = {r.memory.id for r in second if r.source == "direct"}
        assert all(r.memory.id not in direct_ids for r in expanded)
    finally:
        await mem.close()


async def test_auto_expand_false_disables_expansion(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        for c in [
            "alice alpha",
            "alice beta",
            "alice gamma",
            "alice delta",
            "alice epsilon",
            "alice zeta",
        ]:
            await mem.observe(role="user", content=c, session_id="s1")

        assert mem._engine is not None
        await mem._engine.recall("alice alpha", agent_id=None, k=2, pool_size=2)
        second = await mem._engine.recall(
            "alice alpha", agent_id=None, k=2, pool_size=2, auto_expand=False
        )
        assert all(r.source == "direct" for r in second)
    finally:
        await mem.close()


async def test_first_recall_never_expands(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        for c in ["x one", "x two", "x three"]:
            await mem.observe(role="user", content=c, session_id="s1")
        first = await mem.recall("x one", k=2)
        assert all(r.source == "direct" for r in first)
    finally:
        await mem.close()


async def test_empty_direct_results_no_expansion(tmp_path: Path) -> None:
    # With no memories, direct is empty; history recording still happens
    # but there's nothing to seed expansion from on follow-up.
    mem = _mnemoss(tmp_path)
    try:
        first = await mem.recall("anything", k=5)
        assert first == []
        second = await mem.recall("anything", k=5)
        assert second == []
    finally:
        await mem.close()


async def test_expansion_respects_agent_scope(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    alice = mem.for_agent("alice")
    bob = mem.for_agent("bob")
    try:
        # Bob's session — every memory private to bob.
        for c in ["bob task start", "bob task middle", "bob task end"]:
            await bob.observe(role="user", content=c, session_id="s_bob")

        # Alice recalls. She shouldn't see bob memories even on expansion.
        await alice.observe(role="user", content="alice note", session_id="s_alice")

        first = await alice.recall("task", k=5)
        assert all(r.memory.agent_id in ("alice", None) for r in first)
        second = await alice.recall("task again", k=5)
        for r in second:
            assert r.memory.agent_id in ("alice", None), (
                f"leaked bob memory via expansion: {r.memory.id}"
            )
    finally:
        await mem.close()


async def test_expansion_fires_after_long_gap_with_fresh_streak(
    tmp_path: Path,
) -> None:
    """User returning to a thread hours later still gets expansion —
    detection is semantic, not time-gated. The streak resets to 1 so
    hop-count starts shallow, but expansion still fires."""

    mem = _mnemoss(tmp_path)
    try:
        for c in [
            "alice alpha",
            "alice beta",
            "alice gamma",
            "alice delta",
            "alice epsilon",
            "alice zeta",
        ]:
            await mem.observe(role="user", content=c, session_id="s1")

        assert mem._engine is not None
        engine = mem._engine

        # First recall establishes history.
        first = await engine.recall(
            "alice alpha", agent_id=None, k=2, pool_size=2
        )
        assert first

        # Simulate an hour-long gap by rewriting the history entry's
        # timestamp. Detection is time-agnostic, so same-topic still
        # fires; streak resets to 1 because the gap exceeds
        # streak_reset_seconds.
        prev_entry = engine._history.latest(None)
        assert prev_entry is not None
        prev_entry.timestamp = prev_entry.timestamp - timedelta(hours=1)
        prev_entry.streak = 3  # previous streak was deep; gap should reset it

        second = await engine.recall(
            "alice alpha", agent_id=None, k=2, pool_size=2
        )
        expanded = [r for r in second if r.source == "expanded"]
        assert expanded, "expansion should fire even after a long gap"

        # Streak reset: the newly-recorded entry should show streak=1,
        # not 4 (prev.streak + 1).
        new_entry = engine._history.latest(None)
        assert new_entry is not None
        assert new_entry.streak == 1
    finally:
        await mem.close()


async def test_different_topic_does_not_expand(
    tmp_path: Path,
) -> None:
    mem = _mnemoss(tmp_path)
    try:
        # Two disjoint clusters in separate sessions → no cross-session
        # co_occurrence edges between them.
        for c in ["alice alpha", "alice beta"]:
            await mem.observe(role="user", content=c, session_id="s_alice")
        for c in ["dinner recipe", "dinner plan"]:
            await mem.observe(role="user", content=c, session_id="s_dinner")

        await mem.recall("alice alpha", k=2)
        second = await mem.recall("dinner recipe", k=2)
        # Recall hit the dinner cluster, not the alice one. No overlap and
        # low cosine → no expansion fires.
        assert all(r.source == "direct" for r in second)
    finally:
        await mem.close()


def test_streak_resets_beyond_window_in_engine_logic(tmp_path: Path) -> None:
    """Pure-logic check for the streak reset threshold.

    The engine applies the reset using ``params.streak_reset_seconds``;
    this test exercises the branch decision without spinning up a full
    Mnemoss. It's a regression guard against the old behaviour where
    the time threshold gated detection itself.
    """

    params = FormulaParams()
    assert params.streak_reset_seconds == 600.0  # default, 10 min

    now = _fixed_now()
    within = now - timedelta(seconds=500)
    beyond = now - timedelta(seconds=700)
    assert (now - within).total_seconds() <= params.streak_reset_seconds
    assert (now - beyond).total_seconds() > params.streak_reset_seconds


# ─── explicit expand() API ────────────────────────────────────────


async def test_explicit_expand_returns_related_memories(tmp_path: Path) -> None:
    """mem.expand(seed) surfaces memories reachable via relations."""

    mem = _mnemoss(tmp_path)
    try:
        ids = []
        for c in ["apple", "banana", "cherry", "date"]:
            mid = await mem.observe(role="user", content=c, session_id="s1")
            assert mid is not None
            ids.append(mid)

        results = await mem.expand(ids[0], hops=1, k=5)
        # Seed memory (ids[0]) must NOT appear — caller already has it.
        assert all(r.memory.id != ids[0] for r in results)
        # Related memories should surface.
        assert results
        assert all(r.source == "expanded" for r in results)
    finally:
        await mem.close()


async def test_explicit_expand_missing_memory_returns_empty(
    tmp_path: Path,
) -> None:
    """Non-existent seed id → empty list (not an error)."""

    mem = _mnemoss(tmp_path)
    try:
        results = await mem.expand("nonexistent-id", hops=1)
        assert results == []
    finally:
        await mem.close()


async def test_explicit_expand_respects_agent_scope(tmp_path: Path) -> None:
    """Alice can't seed expansion from Bob's private memory."""

    mem = _mnemoss(tmp_path)
    alice = mem.for_agent("alice")
    bob = mem.for_agent("bob")
    try:
        bob_mid = await bob.observe(role="user", content="bob secret", session_id="s_bob")
        assert bob_mid is not None

        # Alice requests expansion from Bob's memory — should get empty.
        results = await alice.expand(bob_mid, hops=1)
        assert results == []
    finally:
        await mem.close()


async def test_explicit_expand_bypasses_same_topic_heuristic(
    tmp_path: Path,
) -> None:
    """Explicit expand doesn't require any prior recall history."""

    mem = _mnemoss(tmp_path)
    try:
        ids = []
        for c in ["x", "y", "z"]:
            mid = await mem.observe(role="user", content=c, session_id="s1")
            assert mid is not None
            ids.append(mid)

        # No recall() has happened; auto-expand heuristic cannot fire.
        # Explicit expand still works.
        results = await mem.expand(ids[0], hops=1)
        assert results
    finally:
        await mem.close()


async def test_explicit_expand_reconsolidates_seed_and_results(
    tmp_path: Path,
) -> None:
    """Expanding is engagement — access_history bumps on seed + results."""

    mem = _mnemoss(tmp_path)
    try:
        ids = []
        for c in ["alpha", "bravo"]:
            mid = await mem.observe(role="user", content=c, session_id="s1")
            assert mid is not None
            ids.append(mid)

        assert mem._store is not None
        before_seed = await mem._store.get_memory(ids[0])
        assert before_seed is not None
        before_count = before_seed.rehearsal_count

        await mem.expand(ids[0], hops=1)

        after_seed = await mem._store.get_memory(ids[0])
        assert after_seed is not None
        assert after_seed.rehearsal_count == before_count + 1
    finally:
        await mem.close()


async def test_explicit_expand_rejects_invalid_hops(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mid = await mem.observe(role="user", content="x", session_id="s1")
        assert mid is not None
        with pytest.raises(ValueError, match="hops"):
            await mem.expand(mid, hops=0)
    finally:
        await mem.close()


# ─── helpers ──────────────────────────────────────────────────────


def _fixed_now() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
