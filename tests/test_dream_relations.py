"""P5 Relations tests (Checkpoint N)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.dream.cluster import ClusterAssignment
from mnemoss.dream.relations import (
    write_derived_from_edges,
    write_shares_entity_edges,
    write_similar_to_edges,
)
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


def _memory(id: str, content: str) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=None,
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


async def test_similar_to_edges_are_symmetric(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    for mid in ("m1", "m2", "m3"):
        await b.write_memory(_memory(mid, mid), np.array([1, 0, 0, 0], dtype=np.float32))

    assignments = {
        "m1": ClusterAssignment("c1", 0.9, True),
        "m2": ClusterAssignment("c1", 0.8, False),
        "m3": ClusterAssignment("c1", 0.7, False),
    }
    count = await write_similar_to_edges(b, assignments)
    # 3 members → C(3,2) = 3 pairs × 2 directions = 6 edges.
    assert count == 6
    out = await b.relations_from(["m1", "m2", "m3"])
    assert "m2" in out["m1"]
    assert "m1" in out["m2"]
    await b.close()


async def test_similar_to_skips_noise(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    for mid in ("m1", "m2"):
        await b.write_memory(_memory(mid, mid), np.array([1, 0, 0, 0], dtype=np.float32))

    assignments = {
        "m1": ClusterAssignment(None, None, False),
        "m2": ClusterAssignment(None, None, False),
    }
    count = await write_similar_to_edges(b, assignments)
    assert count == 0
    await b.close()


async def test_derived_from_edges_are_directed(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    parent1 = _memory("p1", "parent1")
    parent2 = _memory("p2", "parent2")
    child = _memory("c", "child")
    child.derived_from = ["p1", "p2"]

    for m in (parent1, parent2, child):
        await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    count = await write_derived_from_edges(b, [child])
    assert count == 2
    # Edges go child → parent, not the other way.
    out = await b.relations_from(["c", "p1"])
    assert "p1" in out["c"]
    assert "c" not in out["p1"]
    await b.close()


# ─── shares_entity (Dream P4 edge from P3 NER output) ──────────────


def _refined(id: str, content: str, entities: list[str]) -> Memory:
    m = _memory(id, content)
    m.extracted_entities = entities
    m.extraction_level = 2
    return m


async def test_shares_entity_writes_symmetric_edges(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    a = _refined("a", "Alice worked with Bob", ["Alice", "Bob"])
    b_mem = _refined("b", "Bob met Carol", ["Bob", "Carol"])
    c = _refined("c", "totally unrelated", ["Dave"])
    for m in (a, b_mem, c):
        await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    count = await write_shares_entity_edges(b, [a, b_mem, c])
    # a∩b_mem = {bob}, |a∪b_mem|=3 → one symmetric pair = 2 edges.
    # c shares nothing → 0 edges.
    assert count == 2
    out = await b.relations_from(["a", "b", "c"])
    assert "b" in out["a"]
    assert "a" in out["b"]
    assert "c" not in out["a"]
    await b.close()


async def test_shares_entity_is_case_insensitive(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    a = _refined("a", "Alice here", ["Alice"])
    b_mem = _refined("b", "alice again", ["ALICE"])
    for m in (a, b_mem):
        await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    count = await write_shares_entity_edges(b, [a, b_mem])
    assert count == 2  # case-folded equality → one pair, symmetric
    await b.close()


async def test_shares_entity_skips_level_below_2(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    # Level-1 memories never get shares_entity edges even if their
    # (stale) entities overlap — Dream P3 is the only authoritative source.
    a = _memory("a", "Alice came by")
    a.extracted_entities = ["Alice"]
    a.extraction_level = 1
    b_mem = _memory("b", "Alice left")
    b_mem.extracted_entities = ["Alice"]
    b_mem.extraction_level = 1
    for m in (a, b_mem):
        await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    count = await write_shares_entity_edges(b, [a, b_mem])
    assert count == 0
    await b.close()


async def test_shares_entity_cjk_entities(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    a = _refined("a", "小明去了北京", ["小明", "北京"])
    b_mem = _refined("b", "小明回来了", ["小明"])
    for m in (a, b_mem):
        await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    count = await write_shares_entity_edges(b, [a, b_mem])
    assert count == 2
    await b.close()
