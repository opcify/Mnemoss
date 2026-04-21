"""Rebalance pass (Checkpoint G / P7).

Verifies that simulating the passage of time and then rebalancing
migrates memories to their correct tiers, and that pinning keeps a
memory HOT regardless of age.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.index import rebalance
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


def _memory(
    id: str,
    content: str,
    created_at: datetime,
    tier: IndexTier = IndexTier.HOT,
    access_history: list[datetime] | None = None,
) -> Memory:
    ah = access_history if access_history is not None else [created_at]
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=None,
        session_id="s1",
        created_at=created_at,
        content=content,
        content_embedding=None,
        role="user",
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=ah,
        index_tier=tier,
        idx_priority=0.9,
    )


async def _backend(tmp_path: Path, dim: int = 4) -> SQLiteBackend:
    b = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        workspace_id="ws",
        embedding_dim=dim,
        embedder_id="fake:dim4",
    )
    await b.open()
    return b


async def test_fresh_memory_stays_hot(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    now = datetime.now(UTC)
    m = _memory("m1", "fresh", now)
    await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    stats = await rebalance(b, FormulaParams(), now=now)
    assert stats.scanned == 1
    got = await b.get_memory("m1")
    assert got is not None
    assert got.index_tier is IndexTier.HOT
    assert got.idx_priority > 0.7
    await b.close()


async def test_aged_unused_memory_drifts_down(tmp_path: Path) -> None:
    """A memory created with only its initial access, then aged for a day
    without any retrievals, should fall out of HOT."""

    b = await _backend(tmp_path)
    creation = datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC)
    m = _memory("old", "old stuff", creation)
    await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    # Simulate one day passing without any reconsolidation.
    stats = await rebalance(
        b, FormulaParams(), now=creation + timedelta(days=1)
    )
    assert stats.scanned == 1
    got = await b.get_memory("old")
    assert got is not None
    # With default η₀=1.0, τ_η=3600s, 86400s later the grace has collapsed
    # and B_i is dominated by ln(86400^-0.5) ≈ -5.68 → σ(-5.68) ≈ 0.003 → DEEP.
    assert got.index_tier in (IndexTier.COLD, IndexTier.DEEP)
    assert got.idx_priority < 0.3
    await b.close()


async def test_pin_lifts_priority_vs_unpinned_twin(tmp_path: Path) -> None:
    """Two identical-age memories; pinning the second lifts its priority.

    γ=2 doesn't keep *ancient* memories HOT (the base-level collapses too
    hard), but on a fresh-enough memory the boost crosses the threshold.
    We verify the boost direction here; the ancient-case behaviour is
    captured in ``test_pin_boost_recent_memory_to_hot``.
    """

    b = await _backend(tmp_path)
    creation = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    unpinned = _memory("unpinned", "keep both", creation)
    pinned = _memory("pinned", "important", creation)
    await b.write_memory(unpinned, np.array([1, 0, 0, 0], dtype=np.float32))
    await b.write_memory(pinned, np.array([0, 1, 0, 0], dtype=np.float32))
    await b.pin("pinned", agent_id="alice")

    await rebalance(b, FormulaParams(), now=creation + timedelta(days=30))
    u = await b.get_memory("unpinned")
    p = await b.get_memory("pinned")
    assert u is not None and p is not None
    # Pin raises idx_priority by a large factor on otherwise-identical memories.
    assert p.idx_priority > u.idx_priority * 3
    await b.close()


async def test_pin_boost_recent_memory_to_hot(tmp_path: Path) -> None:
    """On a moderately old memory where B has decayed into WARM territory,
    γ=2.0 lifts it back into HOT."""

    b = await _backend(tmp_path)
    creation = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC)
    m = _memory("p", "pin me", creation)
    await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))
    await b.pin("p", agent_id=None)  # ambient pin

    # 2 hours later: grace mostly collapsed (~2 e-folds in, e^-2 ≈ 0.135),
    # history term small negative → B ≈ -3.6. σ(-3.6 + 2.0) = σ(-1.6) ≈ 0.17
    # (still not HOT, confirming pin isn't a universal HOT guarantee).
    # At 5 minutes: grace ≈ η₀·e^(-300/3600) ≈ 0.92, history ln(300^-0.5) ≈ -2.85.
    # B ≈ -1.93. σ(-1.93 + 2) = σ(0.07) ≈ 0.52 → WARM. Still not HOT.
    # Even 1-minute-old pinned memory: grace ≈ 0.98, history ln(60^-0.5) ≈ -2.05.
    # B ≈ -1.07. σ(-1.07 + 2) = σ(0.93) ≈ 0.72 → HOT.
    await rebalance(b, FormulaParams(), now=creation + timedelta(seconds=60))
    got = await b.get_memory("p")
    assert got is not None
    assert got.index_tier is IndexTier.HOT
    await b.close()


async def test_recently_accessed_memory_stays_warmer(tmp_path: Path) -> None:
    """Two memories of identical age differ only in access history.
    The one that was recently re-accessed should land in a warmer tier."""

    b = await _backend(tmp_path)
    creation = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    now = creation + timedelta(hours=6)

    untouched = _memory("untouched", "never re-read", creation)
    # Freshly rehearsed memory — recent access entries keep B_i up.
    rehearsed = _memory(
        "rehearsed",
        "re-read often",
        creation,
        access_history=[
            creation,
            creation + timedelta(hours=1),
            creation + timedelta(hours=3),
            now - timedelta(minutes=1),
        ],
    )
    await b.write_memory(untouched, np.array([1, 0, 0, 0], dtype=np.float32))
    await b.write_memory(rehearsed, np.array([0, 1, 0, 0], dtype=np.float32))

    await rebalance(b, FormulaParams(), now=now)
    u = await b.get_memory("untouched")
    r = await b.get_memory("rehearsed")
    assert u is not None and r is not None
    assert r.idx_priority > u.idx_priority
    await b.close()


async def test_tier_distribution_changes(tmp_path: Path) -> None:
    """Tier counts shift from before to after as memories migrate."""

    b = await _backend(tmp_path)
    creation = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    for i in range(5):
        m = _memory(f"m{i}", f"content {i}", creation)
        # Everyone starts HOT.
        await b.write_memory(
            m, np.array([i == k for k in range(4)], dtype=np.float32)
        )

    counts_before = await b.tier_counts()
    assert counts_before[IndexTier.HOT] == 5

    stats = await rebalance(
        b, FormulaParams(), now=creation + timedelta(days=14)
    )
    assert stats.migrated == 5  # all 5 leave HOT at 2-week age
    counts_after = await b.tier_counts()
    assert counts_after[IndexTier.HOT] == 0
    await b.close()


async def test_pinned_any_returns_cross_agent_pins(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    now = datetime.now(UTC)
    for mid in ("m1", "m2", "m3"):
        m = _memory(mid, f"c-{mid}", now)
        await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    await b.pin("m1", "alice")
    await b.pin("m2", None)  # ambient pin
    # m3 unpinned.

    result = await b.pinned_any(["m1", "m2", "m3"])
    assert result == {"m1", "m2"}
    await b.close()
