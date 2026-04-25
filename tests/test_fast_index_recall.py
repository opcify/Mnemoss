"""Tests for ``FormulaParams.use_fast_index_recall``.

Fast-index recall is Mnemoss's architectural bet: cognition (ACT-R
formula) runs async (observe, reconsolidate, dream); the user-facing
recall path is pure ANN top-K + cached ``idx_priority``. These tests
verify the read path behaves as a pure index lookup and skips the
FTS scan / tier cascade / spreading work that the full ACT-R path
does.
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


async def _setup(
    tmp_path: Path,
    params: FormulaParams,
    dim: int = 16,
) -> tuple[SQLiteBackend, RecallEngine, FakeEmbedder]:
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


async def _observe_at_tier(
    store: SQLiteBackend,
    embedder: FakeEmbedder,
    content: str,
    tier: IndexTier,
    *,
    idx_priority: float | None = None,
    agent_id: str | None = None,
    tag: str = "",
) -> Memory:
    msg = RawMessage(
        id=f"raw-{content[:20]}-{tier.value}-{tag}",
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
    memory.idx_priority = idx_priority if idx_priority is not None else {
        IndexTier.HOT: 0.8,
        IndexTier.WARM: 0.5,
        IndexTier.COLD: 0.2,
        IndexTier.DEEP: 0.05,
    }[tier]
    emb = embedder.embed([content])[0]
    await store.write_memory(memory, emb)
    return memory


# ─── recall path ────────────────────────────────────────────────


async def test_fast_index_returns_cos_sim_ranked_with_zero_priority_weight(
    tmp_path: Path,
) -> None:
    """priority_weight=0 → rank is pure cosine via ANN.

    Plant three memories, run a query that matches the first strongly
    and the others weakly, verify the order.
    """

    params = FormulaParams(
        noise_scale=0.0,
        use_fast_index_recall=True,
        fast_index_semantic_weight=1.0,
        fast_index_priority_weight=0.0,
    )
    store, engine, embedder = await _setup(tmp_path, params)
    try:
        m_close = await _observe_at_tier(store, embedder, "alpha beta gamma", IndexTier.HOT)
        m_mid = await _observe_at_tier(store, embedder, "alpha delta epsilon", IndexTier.HOT)
        m_far = await _observe_at_tier(store, embedder, "omega zeta theta", IndexTier.HOT)

        results, stats = await engine.recall_with_stats(
            "alpha beta gamma",
            agent_id=None,
            k=3,
            reconsolidate=False,
        )
        ids = [r.memory.id for r in results]
        assert m_close.id == ids[0]
        # Fast-index path reports no tier scans.
        assert stats.tiers_scanned == []
        # All three memories got scored.
        assert stats.candidates_scored >= 3
        # With k=3 and three memories, all should appear.
        assert set(ids) == {m_close.id, m_mid.id, m_far.id}
    finally:
        await store.close()


async def test_fast_index_priority_weight_breaks_ties(tmp_path: Path) -> None:
    """With identical cosine, higher ``idx_priority`` wins."""

    params = FormulaParams(
        noise_scale=0.0,
        use_fast_index_recall=True,
        fast_index_semantic_weight=1.0,
        fast_index_priority_weight=1.0,
    )
    store, engine, embedder = await _setup(tmp_path, params)
    try:
        # FakeEmbedder deterministically hashes content. To isolate
        # priority's effect we have to make two memories embed to the
        # SAME vector and differ only on priority. Duplicate content
        # does that — both embed identically, identical cos_sim to
        # any query. The one with higher idx_priority should rank
        # higher in fast-index mode.
        m_low = await _observe_at_tier(
            store, embedder, "duplicate content", IndexTier.HOT,
            idx_priority=0.2, tag="low",
        )
        m_high = await _observe_at_tier(
            store, embedder, "duplicate content", IndexTier.HOT,
            idx_priority=0.9, tag="high",
        )

        results, _ = await engine.recall_with_stats(
            "duplicate content",
            agent_id=None,
            k=2,
            reconsolidate=False,
        )
        ids = [r.memory.id for r in results]
        assert ids[0] == m_high.id
        assert ids[1] == m_low.id
    finally:
        await store.close()


async def test_fast_index_skips_fts(tmp_path: Path) -> None:
    """Fast-index never touches FTS regardless of query content.

    Even a quoted query (``b_F = 1.5``, which ``skip_fts_when_no_literal_markers``
    would not skip) must not hit FTS in fast-index mode.
    """

    params = FormulaParams(
        noise_scale=0.0,
        use_fast_index_recall=True,
    )
    store, engine, embedder = await _setup(tmp_path, params)
    await _observe_at_tier(store, embedder, "alice meeting notes", IndexTier.HOT)

    fts_calls = 0
    orig = store.fts_search

    async def _spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal fts_calls
        fts_calls += 1
        return await orig(*args, **kwargs)

    store.fts_search = _spy  # type: ignore[method-assign]
    try:
        # Quoted query — in the ACT-R path this would force FTS.
        await engine.recall_with_stats('"alice"', agent_id=None, k=3, reconsolidate=False)
    finally:
        store.fts_search = orig  # type: ignore[method-assign]
        await store.close()

    assert fts_calls == 0, f"Fast-index should never call FTS; got {fts_calls}"


async def test_fast_index_skips_spreading_and_noise(tmp_path: Path) -> None:
    """With the same memory + params, fast-index gives deterministic
    scores while the ACT-R path adds noise.

    We seed the engine with a non-zero noise_scale and verify the
    fast-index returned breakdown has ``noise == 0.0`` and
    ``spreading == 0.0``.
    """

    params = FormulaParams(
        noise_scale=0.5,  # would materially shift ACT-R scores
        use_fast_index_recall=True,
    )
    store, engine, embedder = await _setup(tmp_path, params)
    try:
        await _observe_at_tier(store, embedder, "alpha beta gamma", IndexTier.HOT)
        results, _ = await engine.recall_with_stats(
            "alpha beta gamma",
            agent_id=None,
            k=1,
            reconsolidate=False,
        )
        assert len(results) == 1
        b = results[0].breakdown
        assert b.noise == 0.0
        assert b.spreading == 0.0
    finally:
        await store.close()


async def test_default_params_still_use_act_r_path(tmp_path: Path) -> None:
    """Default ``FormulaParams`` keeps the full ACT-R recall.

    Regression guard: if someone flips the default, this test surfaces
    it (full path scans at least the HOT tier).
    """

    store, engine, embedder = await _setup(tmp_path, FormulaParams(noise_scale=0.0))
    try:
        await _observe_at_tier(store, embedder, "alice meeting", IndexTier.HOT)
        _, stats = await engine.recall_with_stats(
            "alice",
            agent_id=None,
            k=3,
            reconsolidate=False,
        )
        assert IndexTier.HOT in stats.tiers_scanned
    finally:
        await store.close()


# ─── validation ─────────────────────────────────────────────────


async def test_fast_index_requires_positive_weight(tmp_path: Path) -> None:
    """use_fast_index_recall=True with both weights == 0 is a
    configuration error — ranking would collapse to arbitrary."""

    import pytest

    with pytest.raises(ValueError, match="fast_index"):
        FormulaParams(
            use_fast_index_recall=True,
            fast_index_semantic_weight=0.0,
            fast_index_priority_weight=0.0,
        )
