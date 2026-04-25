"""Cascade-retrieval tests (Checkpoint F).

Exercises the tier-by-tier scan with early stopping and the DEEP gating
rules. Uses a real SQLite backend + FakeEmbedder so the cascade logic is
verified end-to-end without model I/O.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import IndexTier, Memory, RawMessage
from mnemoss.encoder import FakeEmbedder
from mnemoss.encoder.event_encoder import encode_message
from mnemoss.recall import RecallEngine
from mnemoss.store.sqlite_backend import SQLiteBackend
from mnemoss.working import WorkingMemory

UTC = timezone.utc


async def _setup(tmp_path: Path, dim: int = 16):
    embedder = FakeEmbedder(dim=dim)
    store = SQLiteBackend(
        db_path=tmp_path / "mem.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=dim,
        embedder_id=embedder.embedder_id,
    )
    await store.open()
    wm = WorkingMemory(capacity=10)
    engine = RecallEngine(
        store=store,
        embedder=embedder,
        working=wm,
        # These tests assert legacy ACT-R cascade semantics (confidence-
        # threshold early-stop, FTS+activation scoring). Production
        # default switched to ``use_tier_cascade_recall`` in 2026-04;
        # opt out here so the suite continues to exercise the path it
        # was written for.
        params=FormulaParams(
            noise_scale=0.0, use_tier_cascade_recall=False
        ),
        rng=random.Random(0),
    )
    return store, engine, embedder


async def _observe_at_tier(
    store: SQLiteBackend,
    embedder: FakeEmbedder,
    content: str,
    tier: IndexTier,
    *,
    agent_id: str | None = None,
) -> Memory:
    msg = RawMessage(
        id=f"raw-{content[:20]}-{tier.value}",
        workspace_id="ws",
        agent_id=agent_id,
        session_id="s1",
        turn_id="t",
        parent_id=None,
        timestamp=datetime.now(UTC),
        role="user",
        content=content,
    )
    memory = encode_message(msg)
    memory.index_tier = tier
    memory.idx_priority = {
        IndexTier.HOT: 0.9,
        IndexTier.WARM: 0.5,
        IndexTier.COLD: 0.2,
        IndexTier.DEEP: 0.05,
    }[tier]
    emb = embedder.embed([content])[0]
    await store.write_memory(memory, emb)
    return memory


async def test_cascade_stops_at_hot_when_fresh_hit(tmp_path: Path) -> None:
    store, engine, embedder = await _setup(tmp_path)
    # Only HOT memory present, fresh enough to exceed CONFIDENCE_HOT=1.0.
    await _observe_at_tier(store, embedder, "Alice meeting 4:20", IndexTier.HOT)

    _, stats = await engine.recall_with_stats("Alice meeting", agent_id=None, k=3)
    assert stats.stopped_at is IndexTier.HOT
    assert stats.tiers_scanned == [IndexTier.HOT]
    await store.close()


async def test_cascade_falls_through_when_hot_empty(tmp_path: Path) -> None:
    store, engine, embedder = await _setup(tmp_path)
    # Stash the content only in COLD; HOT and WARM are empty.
    m = await _observe_at_tier(store, embedder, "ancient record about Alice", IndexTier.COLD)

    results, stats = await engine.recall_with_stats("Alice", agent_id=None, k=3)
    # Cold hit gets returned because it clears tau even if no early-stop.
    assert m.id in {r.memory.id for r in results}
    # All three default tiers scanned; no early stop.
    assert stats.tiers_scanned == [IndexTier.HOT, IndexTier.WARM, IndexTier.COLD]
    await store.close()


async def test_deep_excluded_from_default_recall(tmp_path: Path) -> None:
    store, engine, embedder = await _setup(tmp_path)
    deep = await _observe_at_tier(store, embedder, "lost memory about Alice", IndexTier.DEEP)

    results, stats = await engine.recall_with_stats("Alice", agent_id=None, k=3)
    assert deep.id not in {r.memory.id for r in results}
    # DEEP never scanned without opt-in.
    assert IndexTier.DEEP not in stats.tiers_scanned
    await store.close()


async def test_deep_included_with_opt_in(tmp_path: Path) -> None:
    store, engine, embedder = await _setup(tmp_path)
    deep = await _observe_at_tier(store, embedder, "lost memory about Alice", IndexTier.DEEP)

    results, stats = await engine.recall_with_stats("Alice", agent_id=None, k=3, include_deep=True)
    assert deep.id in {r.memory.id for r in results}
    assert IndexTier.DEEP in stats.tiers_scanned
    await store.close()


async def test_deep_auto_included_on_temporal_cue(tmp_path: Path) -> None:
    store, engine, embedder = await _setup(tmp_path)
    deep = await _observe_at_tier(store, embedder, "original plan about Alice", IndexTier.DEEP)

    # No include_deep=True, but the query contains "long ago" → auto-include.
    results, stats = await engine.recall_with_stats(
        "what did we plan about Alice long ago", agent_id=None, k=3
    )
    assert deep.id in {r.memory.id for r in results}
    assert IndexTier.DEEP in stats.tiers_scanned
    await store.close()


async def test_reminiscence_promotes_deep_hit_to_warm(tmp_path: Path) -> None:
    store, engine, embedder = await _setup(tmp_path)
    deep = await _observe_at_tier(store, embedder, "forgotten Alice note", IndexTier.DEEP)

    results, _ = await engine.recall_with_stats("Alice", agent_id=None, k=3, include_deep=True)
    assert deep.id in {r.memory.id for r in results}

    # After recall, the memory should have jumped to WARM and bumped
    # reminisced_count. Verify against fresh store state (not the in-memory
    # result object).
    got = await store.get_memory(deep.id)
    assert got is not None
    assert got.index_tier is IndexTier.WARM
    assert got.reminisced_count == 1
    await store.close()


async def test_non_deep_hits_do_not_reminisce(tmp_path: Path) -> None:
    store, engine, embedder = await _setup(tmp_path)
    m = await _observe_at_tier(store, embedder, "active note", IndexTier.HOT)

    await engine.recall_with_stats("active note", agent_id=None, k=3)
    got = await store.get_memory(m.id)
    assert got is not None
    assert got.reminisced_count == 0
    assert got.index_tier is IndexTier.HOT
    await store.close()


async def test_stats_are_well_formed(tmp_path: Path) -> None:
    store, engine, embedder = await _setup(tmp_path)
    await _observe_at_tier(store, embedder, "Alice HOT", IndexTier.HOT)
    await _observe_at_tier(store, embedder, "Alice WARM", IndexTier.WARM)
    await _observe_at_tier(store, embedder, "Alice COLD", IndexTier.COLD)

    _, stats = await engine.recall_with_stats("Alice", agent_id=None, k=10, include_deep=True)
    assert stats.candidates_scored >= 1
    assert stats.tiers_scanned
    # Whatever cascade did, stopped_at (if set) must be one of the scanned tiers.
    if stats.stopped_at is not None:
        assert stats.stopped_at in stats.tiers_scanned
    await store.close()


async def test_scoring_is_not_duplicated_across_tiers(tmp_path: Path) -> None:
    """No memory is scored twice even when it is surfaced by both FTS and
    vec search in the same tier."""

    # We only need the embedder from _setup; the engine is rebuilt below
    # against a dedicated store so we can instrument scoring without
    # the default engine's working-memory state in the picture.
    embedder = FakeEmbedder(dim=16)
    sub = tmp_path / "scoring_test"
    sub.mkdir()
    store2 = SQLiteBackend(
        db_path=sub / "mem2.sqlite",
        raw_log_path=sub / "raw_log2.sqlite",
        workspace_id="ws",
        embedding_dim=embedder.dim,
        embedder_id=embedder.embedder_id,
    )
    await store2.open()
    engine = RecallEngine(
        store=store2,
        embedder=embedder,
        working=WorkingMemory(capacity=10),
        params=FormulaParams(noise_scale=0.0, use_tier_cascade_recall=False),
        rng=random.Random(0),
    )

    # Write a batch of memories across three tiers so the cascade fall-through
    # is exercised.
    ids: list[str] = []
    for tier in (IndexTier.HOT, IndexTier.WARM, IndexTier.COLD):
        for i in range(3):
            m = await _observe_at_tier(store2, embedder, f"alice note {tier.value}-{i}", tier)
            ids.append(m.id)

    import mnemoss.recall.engine as engine_mod

    call_counter: dict[str, int] = {}
    original = engine_mod.compute_activation

    def counting(**kwargs):
        mid = kwargs["memory"].id
        call_counter[mid] = call_counter.get(mid, 0) + 1
        return original(**kwargs)

    engine_mod.compute_activation = counting
    try:
        _, _stats = await engine.recall_with_stats(
            "alice note", agent_id=None, k=10, include_deep=True
        )
    finally:
        engine_mod.compute_activation = original

    # Every memory scored exactly once.
    assert all(c == 1 for c in call_counter.values())
    await store2.close()


# A tier-aware vec/fts search is exercised end-to-end in the cascade tests
# above, but we also verify the tier_filter parameter in isolation.
async def test_vec_and_fts_respect_tier_filter(tmp_path: Path) -> None:
    store, _, embedder = await _setup(tmp_path)
    m_hot = await _observe_at_tier(store, embedder, "alice HOT story", IndexTier.HOT)
    m_warm = await _observe_at_tier(store, embedder, "alice WARM story", IndexTier.WARM)

    hot_only = await store.vec_search(
        embedder.embed(["alice"])[0],
        k=10,
        agent_id=None,
        tier_filter={IndexTier.HOT},
    )
    warm_only = await store.vec_search(
        embedder.embed(["alice"])[0],
        k=10,
        agent_id=None,
        tier_filter={IndexTier.WARM},
    )
    hot_ids = {mid for mid, _ in hot_only}
    warm_ids = {mid for mid, _ in warm_only}
    assert m_hot.id in hot_ids
    assert m_hot.id not in warm_ids
    assert m_warm.id in warm_ids
    assert m_warm.id not in hot_ids

    fts_hot = await store.fts_search(
        "alice HOT story", k=10, agent_id=None, tier_filter={IndexTier.HOT}
    )
    fts_warm = await store.fts_search(
        "alice HOT story", k=10, agent_id=None, tier_filter={IndexTier.WARM}
    )
    assert any(mid == m_hot.id for mid, _ in fts_hot)
    assert all(mid != m_hot.id for mid, _ in fts_warm)
    await store.close()


# ─── skip_fts_when_no_literal_markers ────────────────────────────


async def _setup_with_params(tmp_path: Path, params: FormulaParams, dim: int = 16):
    """Mirror ``_setup`` but with caller-supplied ``FormulaParams``."""

    embedder = FakeEmbedder(dim=dim)
    store = SQLiteBackend(
        db_path=tmp_path / "mem.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=dim,
        embedder_id=embedder.embedder_id,
    )
    await store.open()
    wm = WorkingMemory(capacity=10)
    engine = RecallEngine(
        store=store,
        embedder=embedder,
        working=wm,
        params=params,
        rng=random.Random(0),
    )
    return store, engine, embedder


async def test_skip_fts_on_plain_query_when_knob_on(tmp_path: Path) -> None:
    """With the knob on + a plain query (``b_F == 1.0``), FTS is not called.

    We verify by wrapping ``store.fts_search`` to count invocations — a
    plain query with the knob on should record zero FTS calls while
    still returning results via vec_search.
    """

    params = FormulaParams(
        noise_scale=0.0,
        skip_fts_when_no_literal_markers=True,
        use_tier_cascade_recall=False,
    )
    store, engine, embedder = await _setup_with_params(tmp_path, params)
    await _observe_at_tier(store, embedder, "alice meeting notes", IndexTier.HOT)

    fts_calls = 0
    orig_fts = store.fts_search

    async def _spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal fts_calls
        fts_calls += 1
        return await orig_fts(*args, **kwargs)

    store.fts_search = _spy  # type: ignore[method-assign]
    try:
        results, _ = await engine.recall_with_stats("alice", agent_id=None, k=3)
        assert len(results) > 0  # vec_search still finds it
    finally:
        store.fts_search = orig_fts  # type: ignore[method-assign]
        await store.close()

    assert fts_calls == 0, (
        f"Expected FTS to be skipped on plain query with knob on; "
        f"got {fts_calls} call(s)"
    )


async def test_skip_fts_still_runs_on_literal_query(tmp_path: Path) -> None:
    """The knob only skips when ``b_F(query) == 1.0``. Quoted queries
    (``b_F = 1.5``) must still hit FTS."""

    params = FormulaParams(
        noise_scale=0.0,
        skip_fts_when_no_literal_markers=True,
        use_tier_cascade_recall=False,
    )
    store, engine, embedder = await _setup_with_params(tmp_path, params)
    await _observe_at_tier(store, embedder, "alice meeting notes", IndexTier.HOT)

    fts_calls = 0
    orig_fts = store.fts_search

    async def _spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal fts_calls
        fts_calls += 1
        return await orig_fts(*args, **kwargs)

    store.fts_search = _spy  # type: ignore[method-assign]
    try:
        # Quoted query → b_F(q) = 1.5 → do not skip FTS.
        await engine.recall_with_stats('"alice"', agent_id=None, k=3)
    finally:
        store.fts_search = orig_fts  # type: ignore[method-assign]
        await store.close()

    assert fts_calls >= 1, (
        f"Literal (quoted) query should hit FTS at least once; got {fts_calls}"
    )


async def test_skip_fts_default_off_preserves_hybrid(tmp_path: Path) -> None:
    """Default params → FTS always runs, even on plain queries."""

    store, engine, embedder = await _setup(tmp_path)  # default params
    await _observe_at_tier(store, embedder, "alice meeting notes", IndexTier.HOT)

    fts_calls = 0
    orig_fts = store.fts_search

    async def _spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal fts_calls
        fts_calls += 1
        return await orig_fts(*args, **kwargs)

    store.fts_search = _spy  # type: ignore[method-assign]
    try:
        await engine.recall_with_stats("alice", agent_id=None, k=3)
    finally:
        store.fts_search = orig_fts  # type: ignore[method-assign]
        await store.close()

    assert fts_calls >= 1


# ─── skip_empty_tiers ────────────────────────────────────────────


async def test_skip_empty_tiers_drops_round_trips(tmp_path: Path) -> None:
    """With the knob on, only non-empty tiers appear in ``tiers_scanned``.

    Seed HOT only (typical bulk-ingest state) and confirm the cascade
    doesn't bother with WARM/COLD.
    """

    params = FormulaParams(
        noise_scale=0.0,
        skip_empty_tiers=True,
        use_tier_cascade_recall=False,
    )
    store, engine, embedder = await _setup_with_params(tmp_path, params)
    await _observe_at_tier(store, embedder, "alice notes", IndexTier.HOT)

    _, stats = await engine.recall_with_stats("alice", agent_id=None, k=3)
    assert stats.tiers_scanned == [IndexTier.HOT]
    await store.close()


async def test_skip_empty_tiers_default_off_scans_all(tmp_path: Path) -> None:
    """With the knob off (default), cascade scans HOT/WARM/COLD even
    when WARM and COLD are empty — preserves existing behaviour."""

    store, engine, embedder = await _setup(tmp_path)  # default params
    # Put content only in COLD so the HOT / WARM scan returns empty and
    # the cascade must fall through to COLD. This matches
    # test_cascade_falls_through_when_hot_empty's expectations.
    await _observe_at_tier(store, embedder, "old alice note", IndexTier.COLD)

    _, stats = await engine.recall_with_stats("alice", agent_id=None, k=3)
    assert stats.tiers_scanned == [IndexTier.HOT, IndexTier.WARM, IndexTier.COLD]
    await store.close()
