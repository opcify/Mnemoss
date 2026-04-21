"""Store-layer integration tests (real SQLite + sqlite-vec + FTS5)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from mnemoss.core.types import IndexTier, Memory, MemoryType, RawMessage
from mnemoss.store.sqlite_backend import (
    SchemaMismatchError,
    SQLiteBackend,
    build_trigram_query,
)

UTC = timezone.utc


def _memory(
    id: str, content: str, agent_id: str | None = None, session_id: str = "s1"
) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=agent_id,
        session_id=session_id,
        created_at=now,
        content=content,
        content_embedding=None,
        role="user",
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[now],
        index_tier=IndexTier.HOT,
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


async def test_round_trip_memory(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    m = _memory("m1", "hello world")
    emb = np.array([1, 0, 0, 0], dtype=np.float32)
    await b.write_memory(m, emb)

    got = await b.get_memory("m1")
    assert got is not None
    assert got.content == "hello world"
    assert got.agent_id is None
    await b.close()


async def test_schema_mismatch_raises(tmp_path: Path) -> None:
    b = await _backend(tmp_path, dim=4)
    await b.close()

    wrong = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        workspace_id="ws",
        embedding_dim=8,
        embedder_id="fake:dim4",
    )
    with pytest.raises(SchemaMismatchError):
        await wrong.open()


async def test_vec_search_returns_nearest(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    await b.write_memory(_memory("m1", "a"), np.array([1, 0, 0, 0], dtype=np.float32))
    await b.write_memory(_memory("m2", "b"), np.array([0, 1, 0, 0], dtype=np.float32))
    await b.write_memory(_memory("m3", "c"), np.array([0.9, 0.1, 0, 0], dtype=np.float32))

    hits = await b.vec_search(np.array([1, 0, 0, 0], dtype=np.float32), k=2, agent_id=None)
    ids = [h[0] for h in hits]
    assert "m1" in ids
    # m3 is almost identical to m1; should also appear.
    assert "m3" in ids
    # Similarity is in [0,1]-ish; exact m1 should have highest.
    sim = dict(hits)
    assert sim["m1"] >= sim["m3"]
    await b.close()


async def test_fts_search_finds_chinese_via_trigram(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    await b.write_memory(
        _memory("m1", "我明天下午 4:20 和 Alice 见面"),
        np.array([1, 0, 0, 0], dtype=np.float32),
    )
    await b.write_memory(
        _memory("m2", "见面地点在悉尼歌剧院旁边"),
        np.array([0, 1, 0, 0], dtype=np.float32),
    )
    # Query includes "Alice" which is in m1 only.
    hits = await b.fts_search("什么时候见 Alice?", k=5, agent_id=None)
    ids = [h[0] for h in hits]
    assert "m1" in ids
    # SQLite BM25 convention: negative, lower = better match.
    assert all(bm25 <= 0 for _, bm25 in hits)
    await b.close()


async def test_agent_scope_filters(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    await b.write_memory(
        _memory("priv_alice", "alice-only text", agent_id="alice"),
        np.array([1, 0, 0, 0], dtype=np.float32),
    )
    await b.write_memory(
        _memory("priv_bob", "bob-only text", agent_id="bob"),
        np.array([0, 1, 0, 0], dtype=np.float32),
    )
    await b.write_memory(
        _memory("ambient", "shared text"),
        np.array([0, 0, 1, 0], dtype=np.float32),
    )

    # Alice sees her own + ambient, not Bob's.
    hits = await b.vec_search(
        np.array([1, 1, 1, 0], dtype=np.float32), k=10, agent_id="alice"
    )
    ids = {h[0] for h in hits}
    assert "priv_alice" in ids
    assert "ambient" in ids
    assert "priv_bob" not in ids

    # Ambient-only caller sees only ambient.
    hits_amb = await b.vec_search(
        np.array([0, 0, 1, 0], dtype=np.float32), k=10, agent_id=None
    )
    ids_amb = {h[0] for h in hits_amb}
    assert ids_amb == {"ambient"}
    await b.close()


async def test_raw_log_and_relations(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    now = datetime.now(UTC)
    msg = RawMessage(
        id="raw1",
        workspace_id="ws",
        agent_id="alice",
        session_id="s1",
        turn_id="t1",
        parent_id=None,
        timestamp=now,
        role="user",
        content="hello",
    )
    await b.write_raw_message(msg)
    await b.write_memory(_memory("m1", "hello"), np.array([1, 0, 0, 0], dtype=np.float32))
    await b.write_memory(_memory("m2", "world"), np.array([0, 1, 0, 0], dtype=np.float32))
    await b.write_relation("m1", "m2", "co_occurs_in_session", 0.5)

    fan = await b.fan_out(["m1", "m2"])
    assert fan["m1"] == 1
    assert fan["m2"] == 0
    rels = await b.relations_from(["m1"])
    assert rels["m1"] == {"m2"}
    await b.close()


async def test_pin_is_per_agent(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    await b.write_memory(_memory("m1", "shared"), np.array([1, 0, 0, 0], dtype=np.float32))
    await b.pin("m1", "alice")
    assert await b.is_pinned("m1", "alice") is True
    assert await b.is_pinned("m1", "bob") is False
    assert await b.is_pinned("m1", None) is False
    await b.pin("m1", "bob")  # now multiple agents pin the same memory
    assert await b.is_pinned("m1", "bob") is True
    await b.close()


async def test_reconsolidate_appends_access(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    m = _memory("m1", "hi")
    await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))
    before = await b.get_memory("m1")
    assert before is not None
    assert len(before.access_history) == 1

    await b.reconsolidate("m1", datetime.now(UTC))
    after = await b.get_memory("m1")
    assert after is not None
    assert len(after.access_history) == 2
    assert after.rehearsal_count == 1
    await b.close()


def test_build_trigram_query_handles_short_input() -> None:
    assert build_trigram_query("") is None
    assert build_trigram_query("ab") is None  # < 3 chars


def test_build_trigram_query_ors_grams() -> None:
    q = build_trigram_query("Alice")
    assert q is not None
    assert '"Ali"' in q
    assert '"lic"' in q
    assert '"ice"' in q
    assert " OR " in q


def test_build_trigram_query_strips_fts_metachars() -> None:
    # ? : ( ) " are stripped because they're FTS5 syntax.
    q = build_trigram_query("4:20 PM?")
    assert q is not None
    # colon is stripped; should contain "4 2" no it's trigrams of "4 20 PM"
    # after strip → "4 20 PM" — trigrams: "4 2", " 20", "20 ", "0 P", " PM"
    assert '"4' not in q or '"4:' not in q  # ensure no raw colons survive
