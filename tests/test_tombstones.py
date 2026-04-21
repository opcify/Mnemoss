"""Tombstone storage + client.tombstones() tests (Checkpoint Q)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from mnemoss import FakeEmbedder, Mnemoss, StorageParams, Tombstone
from mnemoss.core.types import IndexTier, Memory, MemoryType
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


def _memory(id: str, content: str, *, agent_id: str | None = None) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=agent_id,
        session_id="s",
        created_at=now,
        content=content,
        content_embedding=None,
        role="user",
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[now],
        index_tier=IndexTier.HOT,
    )


async def test_write_and_read_tombstone(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    t = Tombstone(
        original_id="m1",
        workspace_id="ws",
        agent_id=None,
        dropped_at=datetime(2026, 4, 21, tzinfo=UTC),
        reason="activation_dead",
        gist_snapshot="the gist of the disposed memory",
        b_at_drop=-7.5,
        source_message_ids=["r1", "r2"],
    )
    await b.write_tombstone(t)
    got = await b.list_tombstones(limit=10)
    assert len(got) == 1
    assert got[0].original_id == "m1"
    assert got[0].reason == "activation_dead"
    assert got[0].source_message_ids == ["r1", "r2"]
    assert got[0].b_at_drop == pytest.approx(-7.5)
    await b.close()


async def test_list_tombstones_agent_scope(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    now = datetime.now(UTC)
    await b.write_tombstone(
        Tombstone("mA", "ws", "alice", now, "activation_dead", "gA", -7, [])
    )
    await b.write_tombstone(
        Tombstone("mB", "ws", "bob", now, "redundant", "gB", -3, [])
    )
    await b.write_tombstone(
        Tombstone("mX", "ws", None, now, "activation_dead", "gX", -7, [])
    )

    ambient = await b.list_tombstones(agent_id=None)
    assert {t.original_id for t in ambient} == {"mX"}

    alice_view = await b.list_tombstones(agent_id="alice")
    assert {t.original_id for t in alice_view} == {"mA", "mX"}
    await b.close()


async def test_delete_memory_completely_removes_from_all_tables(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    m = _memory("m1", "sample content")
    await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))
    await b.write_relation("m1", "m1", "self_ref", 1.0)
    await b.pin("m1", agent_id="alice")

    await b.delete_memory_completely("m1")

    assert await b.get_memory("m1") is None
    vec_hits = await b.vec_search(
        np.array([1, 0, 0, 0], dtype=np.float32), k=5, agent_id=None
    )
    assert all(mid != "m1" for mid, _ in vec_hits)
    fts_hits = await b.fts_search("sample", k=5, agent_id=None)
    assert all(mid != "m1" for mid, _ in fts_hits)
    assert await b.is_pinned("m1", agent_id="alice") is False
    rels = await b.relations_from(["m1"])
    assert rels["m1"] == set()
    await b.close()


async def test_cluster_size_counts_cluster_members(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    for i in range(3):
        m = _memory(f"m{i}", f"c{i}")
        m.cluster_id = "cluster-xyz"
        await b.write_memory(
            m, np.array([1 if i == 0 else 0, 0, 0, 0], dtype=np.float32)
        )
    # Memory in a different cluster shouldn't count.
    other = _memory("m3", "other")
    other.cluster_id = "cluster-abc"
    await b.write_memory(other, np.array([0, 1, 0, 0], dtype=np.float32))

    assert await b.cluster_size("cluster-xyz") == 3
    assert await b.cluster_size("cluster-abc") == 1
    assert await b.cluster_size("missing") == 0
    await b.close()


# ─── client-level accessor ──────────────────────────────────────


def _mnemoss(tmp_path: Path) -> Mnemoss:
    return Mnemoss(
        workspace="t",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
    )


async def test_client_tombstones_returns_empty_by_default(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="x")
        ts = await mem.tombstones()
        assert ts == []
    finally:
        await mem.close()
