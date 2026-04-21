"""Relation-graph tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mnemoss.core.config import EncoderParams
from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.relations import write_cooccurrence_edges
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


def _memory(id: str, content: str, session_id: str = "s1") -> Memory:
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=None,
        session_id=session_id,
        created_at=datetime.now(UTC),
        content=content,
        content_embedding=None,
        role="user",
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[datetime.now(UTC)],
        index_tier=IndexTier.HOT,
    )


async def _backend(tmp_path: Path) -> SQLiteBackend:
    b = SQLiteBackend(
        db_path=tmp_path / "mem.sqlite",
        workspace_id="ws",
        embedding_dim=4,
        embedder_id="fake:4",
    )
    await b.open()
    return b


async def test_cooccurrence_links_recent_session_memories(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    params = EncoderParams(session_cooccurrence_window=3)
    emb = np.array([1, 0, 0, 0], dtype=np.float32)
    # Write three memories in the same session, then call the edge-writer
    # on a fourth as though it had just been encoded.
    for mid in ["m1", "m2", "m3"]:
        await b.write_memory(_memory(mid, mid), emb)
    await b.write_memory(_memory("m4", "m4"), emb)
    await write_cooccurrence_edges(b, "m4", "s1", params)

    rels = await b.relations_from(["m4"])
    assert {"m1", "m2", "m3"} <= rels["m4"]
    fan = await b.fan_out(["m4"])
    assert fan["m4"] >= 3

    # Edges are bidirectional.
    back = await b.relations_from(["m1", "m2", "m3"])
    for mid in ["m1", "m2", "m3"]:
        assert "m4" in back[mid]
    await b.close()


async def test_cooccurrence_respects_session_boundary(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    params = EncoderParams(session_cooccurrence_window=5)
    emb = np.array([1, 0, 0, 0], dtype=np.float32)
    await b.write_memory(_memory("m1", "m1", session_id="sA"), emb)
    await b.write_memory(_memory("m2", "m2", session_id="sB"), emb)
    await write_cooccurrence_edges(b, "m2", "sB", params)

    rels = await b.relations_from(["m2"])
    assert "m1" not in rels["m2"]  # different session
    await b.close()


async def test_no_session_no_edges(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    params = EncoderParams()
    emb = np.array([1, 0, 0, 0], dtype=np.float32)
    await b.write_memory(_memory("solo", "hi", session_id="s1"), emb)
    # Caller passes session_id=None → noop
    await write_cooccurrence_edges(b, "solo", None, params)
    rels = await b.relations_from(["solo"])
    assert rels["solo"] == set()
    await b.close()
