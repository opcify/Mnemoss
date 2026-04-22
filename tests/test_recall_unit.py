"""Unit tests for the recall pipeline with a real SQLite + FakeEmbedder."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory, RawMessage
from mnemoss.encoder import FakeEmbedder
from mnemoss.encoder.event_encoder import encode_message
from mnemoss.recall import RecallEngine
from mnemoss.store.sqlite_backend import SQLiteBackend
from mnemoss.working import WorkingMemory

UTC = timezone.utc


async def _setup(
    tmp_path: Path, dim: int = 16
) -> tuple[SQLiteBackend, RecallEngine, FakeEmbedder, WorkingMemory]:
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
        params=FormulaParams(),
        rng=random.Random(0),
    )
    return store, engine, embedder, wm


async def _observe(
    store: SQLiteBackend,
    embedder: FakeEmbedder,
    content: str,
    *,
    agent_id: str | None = None,
    session_id: str = "s1",
    role: str = "user",
) -> Memory:
    msg = RawMessage(
        id=f"raw-{content[:20]}",
        workspace_id="ws",
        agent_id=agent_id,
        session_id=session_id,
        turn_id="t",
        parent_id=None,
        timestamp=datetime.now(UTC),
        role=role,
        content=content,
    )
    memory = encode_message(msg)
    emb = embedder.embed([content])[0]
    await store.write_memory(memory, emb)
    return memory


async def test_recall_returns_candidates_above_threshold(tmp_path: Path) -> None:
    store, engine, embedder, _wm = await _setup(tmp_path)
    m1 = await _observe(store, embedder, "我明天下午 4:20 和 Alice 见面")
    m2 = await _observe(store, embedder, "见面地点在悉尼歌剧院旁边")

    results = await engine.recall("什么时候见 Alice?", agent_id=None, k=3)
    assert results, "recall returned nothing"
    ids = {r.memory.id for r in results}
    # Both memories should clear the tau threshold and be returned.
    # (FakeEmbedder gives random semantic scores, and with only 2 documents
    # SQLite's BM25 magnitudes are sub-microscopic so FTS barely discriminates
    # here — the real ranking test is the LocalEmbedder integration test in
    # Checkpoint D.)
    assert m1.id in ids
    assert m2.id in ids

    # All scores above tau.
    assert all(r.score > FormulaParams().tau for r in results)
    await store.close()


async def test_reconsolidation_only_on_returned_top_k(tmp_path: Path) -> None:
    store, engine, embedder, _wm = await _setup(tmp_path)
    # Create 5 candidate memories that all share "Alice".
    ids = []
    for i in range(5):
        m = await _observe(store, embedder, f"Alice note {i}")
        ids.append(m.id)

    results = await engine.recall("Alice", agent_id=None, k=2)
    returned = {r.memory.id for r in results}
    assert len(returned) == 2

    # Check access_history only grew for returned memories.
    for mid in ids:
        got = await store.get_memory(mid)
        assert got is not None
        expected_len = 2 if mid in returned else 1  # creation + one retrieval
        assert len(got.access_history) == expected_len, mid
    await store.close()


async def test_wm_active_set_grows_with_observations_and_recalls(tmp_path: Path) -> None:
    store, engine, embedder, wm = await _setup(tmp_path)
    m1 = await _observe(store, embedder, "Alice note")
    # Observe does NOT touch WM here (client does that). Simulate it:
    wm.append(None, m1.id)
    assert m1.id in wm.active_set(None)

    m2 = await _observe(store, embedder, "Bob note")
    results = await engine.recall("Alice note", agent_id=None, k=1)
    assert results[0].memory.id == m1.id
    # After recall, the returned id was appended to ambient WM.
    assert m1.id in wm.active_set(None)
    _ = m2
    await store.close()


async def test_agent_isolation_in_recall(tmp_path: Path) -> None:
    store, engine, embedder, _wm = await _setup(tmp_path)
    await _observe(store, embedder, "alice secret plan", agent_id="alice")
    await _observe(store, embedder, "bob secret plan", agent_id="bob")
    await _observe(store, embedder, "shared announcement", agent_id=None)

    alice_hits = await engine.recall("secret plan", agent_id="alice", k=5)
    alice_contents = {r.memory.content for r in alice_hits}
    assert "alice secret plan" in alice_contents
    assert "bob secret plan" not in alice_contents

    ambient_hits = await engine.recall("secret plan", agent_id=None, k=5)
    ambient_contents = {r.memory.content for r in ambient_hits}
    # Ambient caller sees only ambient memories.
    assert ambient_contents <= {"shared announcement"}
    await store.close()


async def test_explain_returns_breakdown(tmp_path: Path) -> None:
    store, engine, embedder, _wm = await _setup(tmp_path)
    m = await _observe(store, embedder, "Alice meeting 4:20")
    br = await engine.explain("4:20", memory_id=m.id, agent_id=None)
    assert br is not None
    assert br.total == pytest.approx(br.base_level + br.spreading + br.matching + br.noise)
    await store.close()


async def test_empty_recall_returns_empty_list(tmp_path: Path) -> None:
    store, engine, _embedder, _wm = await _setup(tmp_path)
    results = await engine.recall("nothing observed yet", agent_id=None, k=5)
    assert results == []
    await store.close()
