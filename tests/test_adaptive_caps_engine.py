"""Adaptive caps — recall-engine instrumentation tests.

Verifies _tier_cascade_recall records telemetry when the flag is on,
and is a true no-op when it is off.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.index.adaptive_caps import TierTelemetryLedger
from mnemoss.recall.engine import RecallEngine
from mnemoss.store.sqlite_backend import SQLiteBackend
from mnemoss.working import WorkingMemory

UTC = timezone.utc


def _memory(id: str, content: str, now: datetime) -> Memory:
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=None,
        session_id="s1",
        created_at=now,
        content=content,
        content_embedding=None,
        role="user",
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[now],
        index_tier=IndexTier.HOT,
        idx_priority=0.9,
    )


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


class _StubEmbedder:
    """Deterministic 4-dim embedder — enough to drive vec_search."""

    dim = 4
    embedder_id = "fake"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            rng = random.Random(t)
            out.append(np.array([rng.random() for _ in range(4)], dtype=np.float32))
        return np.stack(out)


async def _seed(b: SQLiteBackend, emb: _StubEmbedder, now: datetime) -> None:
    for i in range(5):
        m = _memory(f"m{i}", f"content number {i}", now)
        vec = emb.embed([m.content])[0]
        await b.write_memory(m, vec)


async def test_engine_records_telemetry_when_flag_on(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    try:
        emb = _StubEmbedder()
        now = datetime.now(UTC)
        await _seed(b, emb, now)
        ledger = TierTelemetryLedger(b._require_conn())
        params = FormulaParams(adaptive_tier_caps=True)
        engine = RecallEngine(
            store=b,
            embedder=emb,
            working=WorkingMemory(capacity=10),
            params=params,
            tier_ledger=ledger,
        )
        await engine.recall("content number 2", agent_id=None, k=3)
        tel = ledger.read()
        assert tel.queries == 1
        # 5 memories, all in HOT → any winners came from HOT.
        assert tel.winners_hot > 0
        assert tel.winners_cold == 0
        assert tel.elapsed_ms_sum >= 0.0
    finally:
        await b.close()


async def test_engine_no_op_when_flag_off(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    try:
        emb = _StubEmbedder()
        now = datetime.now(UTC)
        await _seed(b, emb, now)
        ledger = TierTelemetryLedger(b._require_conn())
        params = FormulaParams(adaptive_tier_caps=False)
        engine = RecallEngine(
            store=b,
            embedder=emb,
            working=WorkingMemory(capacity=10),
            params=params,
            tier_ledger=ledger,
        )
        await engine.recall("content number 2", agent_id=None, k=3)
        assert ledger.read().queries == 0
    finally:
        await b.close()


async def test_engine_no_op_when_ledger_absent(tmp_path: Path) -> None:
    # Flag on but no ledger wired (e.g. legacy construction) → no crash.
    b = await _backend(tmp_path)
    try:
        emb = _StubEmbedder()
        now = datetime.now(UTC)
        await _seed(b, emb, now)
        params = FormulaParams(adaptive_tier_caps=True)
        engine = RecallEngine(
            store=b,
            embedder=emb,
            working=WorkingMemory(capacity=10),
            params=params,
        )
        results = await engine.recall("content number 2", agent_id=None, k=3)
        assert isinstance(results, list)
    finally:
        await b.close()


async def test_engine_provenance_captured_before_reminiscence(tmp_path: Path) -> None:
    # A DEEP-tier memory that wins recall must be attributed to DEEP
    # (scan-time provenance) — even though reconsolidation may promote
    # it DEEP->WARM during the same call.
    b = await _backend(tmp_path)
    try:
        emb = _StubEmbedder()
        now = datetime.now(UTC)
        m = _memory("d0", "deep memory content", now)
        m.index_tier = IndexTier.DEEP
        await b.write_memory(m, emb.embed([m.content])[0])
        ledger = TierTelemetryLedger(b._require_conn())
        params = FormulaParams(adaptive_tier_caps=True)
        engine = RecallEngine(
            store=b,
            embedder=emb,
            working=WorkingMemory(capacity=10),
            params=params,
            tier_ledger=ledger,
        )
        await engine.recall(
            "deep memory content", agent_id=None, k=3, include_deep=True
        )
        tel = ledger.read()
        assert tel.queries == 1
        assert tel.winners_deep == 1
        assert tel.winners_warm == 0
    finally:
        await b.close()
