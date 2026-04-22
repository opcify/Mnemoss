"""Tests for the ``entities`` FTS column on ``memory_fts``.

This is the machinery that lets Dream P3 NER output flow into recall
without any query-side entity parsing. When P3 refines a memory to
level=2 with canonical entities, those entities become BM25-searchable
via the secondary FTS column; a query that happens to contain those
tokens hits the memory even if the memory's content never literally
mentioned them the same way.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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


async def test_entities_column_empty_until_refined(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    m = _memory("m1", "an oblique reference to their visit")
    await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    # Before refinement the entities column is empty — a query for
    # "Alice" should not hit this memory via FTS.
    hits = await b.fts_search("Alice", k=5, agent_id=None)
    assert not hits
    await b.close()


async def test_refined_entities_become_fts_searchable(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    # Content deliberately doesn't literally contain the canonical
    # entity name — we want to prove the entities column carries BM25
    # signal on its own.
    m = _memory("m1", "one of the team came by around noon")
    await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    # Simulate Dream P3 Consolidate lifting to level=2 with a canonical entity.
    await b.update_extraction(
        "m1",
        gist="team member dropped in",
        entities=["Alice Smith"],
        time=None,
        location=None,
        participants=None,
        level=2,
    )

    hits = await b.fts_search("Alice Smith", k=5, agent_id=None)
    assert any(h[0] == "m1" for h in hits)
    await b.close()


async def test_refined_entities_support_cjk_search(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    m = _memory("m1", "同事下午过来一趟")
    await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    # Canonical entity ≥3 chars so the trigram tokenizer has something
    # to match on (``build_trigram_query`` returns None for <3-char
    # inputs). In practice Dream P3 emits longer canonical forms —
    # "小明老师", "北京大学", etc.
    await b.update_extraction(
        "m1",
        gist="同事下午过来",
        entities=["小明老师"],
        time=None,
        location=None,
        participants=None,
        level=2,
    )

    hits = await b.fts_search("小明老师", k=5, agent_id=None)
    assert any(h[0] == "m1" for h in hits)
    await b.close()


async def test_re_refinement_replaces_entities_column(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    m = _memory("m1", "content")
    await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    # First refinement.
    await b.update_extraction(
        "m1",
        gist=None,
        entities=["Alice"],
        time=None,
        location=None,
        participants=None,
        level=2,
    )
    assert any(h[0] == "m1" for h in await b.fts_search("Alice", k=5, agent_id=None))

    # Overwrite with a different entity; old one should no longer match.
    await b.update_extraction(
        "m1",
        gist=None,
        entities=["Bob"],
        time=None,
        location=None,
        participants=None,
        level=2,
    )
    assert any(h[0] == "m1" for h in await b.fts_search("Bob", k=5, agent_id=None))
    assert not any(h[0] == "m1" for h in await b.fts_search("Alice", k=5, agent_id=None))
    await b.close()
