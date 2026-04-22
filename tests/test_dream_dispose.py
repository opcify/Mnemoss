"""P8 Dispose tests (Checkpoint Q)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from mnemoss import FakeEmbedder, Mnemoss, StorageParams
from mnemoss.core.config import FormulaParams
from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.dream.dispose import dispose_pass
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


async def _backend(tmp_path: Path, dim: int = 4) -> SQLiteBackend:
    b = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=dim,
        embedder_id="fake",
    )
    await b.open()
    return b


def _mem(
    id: str,
    content: str,
    created_at: datetime,
    *,
    access_history: list[datetime] | None = None,
    salience: float = 0.0,
    emotional_weight: float = 0.0,
    cluster_id: str | None = None,
    cluster_similarity: float | None = None,
    is_representative: bool = False,
    agent_id: str | None = None,
    memory_type: MemoryType = MemoryType.EPISODE,
) -> Memory:
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=agent_id,
        session_id="s",
        created_at=created_at,
        content=content,
        content_embedding=None,
        role="user",
        memory_type=memory_type,
        abstraction_level=0.0,
        access_history=access_history or [created_at],
        index_tier=IndexTier.HOT,
        salience=salience,
        emotional_weight=emotional_weight,
        cluster_id=cluster_id,
        cluster_similarity=cluster_similarity,
        is_cluster_representative=is_representative,
    )


async def test_activation_dead_memory_disposed(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    # 2-year-old, never re-accessed → B_i ≈ ln(years^-0.5) well below the
    # -6.25 threshold.
    ancient = _mem(
        "ancient",
        "forgotten content",
        created_at=datetime(2024, 4, 21, tzinfo=UTC),
    )
    await b.write_memory(ancient, np.array([1, 0, 0, 0], dtype=np.float32))

    stats = await dispose_pass(b, FormulaParams(), now=datetime(2026, 4, 21, tzinfo=UTC))
    assert stats.disposed == 1
    assert stats.activation_dead == 1
    assert "ancient" in stats.disposed_ids

    tomb = await b.list_tombstones(limit=5)
    assert len(tomb) == 1
    assert tomb[0].reason == "activation_dead"
    assert tomb[0].original_id == "ancient"
    # Memory is physically gone.
    assert await b.get_memory("ancient") is None
    await b.close()


async def test_minimum_age_protection_spares_new_memory(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    now = datetime(2026, 4, 21, tzinfo=UTC)
    # Freshly created (< 30 days) memory with a dead activation profile
    # (should be spared by the min-age guard).
    fresh = _mem(
        "fresh",
        "x",
        created_at=now - timedelta(days=2),
        access_history=[now - timedelta(days=2)],
    )
    await b.write_memory(fresh, np.array([1, 0, 0, 0], dtype=np.float32))

    stats = await dispose_pass(b, FormulaParams(), now=now)
    assert stats.disposed == 0
    assert stats.protected == 1
    assert await b.get_memory("fresh") is not None
    await b.close()


async def test_high_salience_memory_protected(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    old = _mem(
        "important",
        "critical intel",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        salience=0.95,
    )
    await b.write_memory(old, np.array([1, 0, 0, 0], dtype=np.float32))

    stats = await dispose_pass(b, FormulaParams(), now=datetime(2026, 4, 21, tzinfo=UTC))
    assert stats.disposed == 0
    assert stats.protected == 1
    await b.close()


async def test_pinned_memory_protected(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    old = _mem(
        "pinned",
        "pinned content",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    await b.write_memory(old, np.array([1, 0, 0, 0], dtype=np.float32))
    await b.pin("pinned", agent_id=None)

    stats = await dispose_pass(b, FormulaParams(), now=datetime(2026, 4, 21, tzinfo=UTC))
    assert stats.disposed == 0
    assert stats.protected == 1
    await b.close()


async def test_redundant_cluster_member_disposed(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    creation = datetime(2024, 4, 21, tzinfo=UTC)  # old enough to bypass age guard
    now = datetime(2026, 4, 21, tzinfo=UTC)
    # Recent accesses keep B_i above the activation_dead floor so the
    # *redundant* branch is what fires. Without these, the long idle time
    # would make every member activation_dead first.
    recent_access = [now - timedelta(minutes=1)]

    for i in range(5):
        m = _mem(
            f"m{i}",
            f"content {i}",
            created_at=creation,
            access_history=[creation, *recent_access],
            cluster_id="c1",
            cluster_similarity=0.95,
            is_representative=(i == 0),
        )
        emb = np.zeros(4, dtype=np.float32)
        emb[i % 4] = 1.0
        await b.write_memory(m, emb)

    stats = await dispose_pass(b, FormulaParams(), now=now)
    # The 4 non-representatives should be disposed via redundant; m0 stays.
    assert stats.redundant == 4
    assert stats.activation_dead == 0
    assert await b.get_memory("m0") is not None  # rep kept
    for i in range(1, 5):
        assert await b.get_memory(f"m{i}") is None
    await b.close()


async def test_dispose_writes_gist_snapshot(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    m = _mem(
        "m1",
        "This is the full content of the memory.",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    m.extracted_gist = "a short gist"
    await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    await dispose_pass(b, FormulaParams(), now=datetime(2026, 4, 21, tzinfo=UTC))
    tomb = await b.list_tombstones(limit=5)
    assert len(tomb) == 1
    # Prefers extracted_gist over raw content snapshot.
    assert tomb[0].gist_snapshot == "a short gist"
    await b.close()


async def test_client_dispose_standalone(tmp_path: Path) -> None:
    mem = Mnemoss(
        workspace="t",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
    )
    try:
        # observe() creates a fresh memory → protected by age guard.
        await mem.observe(role="user", content="recent note")
        stats = await mem.dispose()
        assert stats.disposed == 0
        assert stats.protected == 1
    finally:
        await mem.close()
