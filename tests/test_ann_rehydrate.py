"""Integration tests for the HNSW index's lifecycle inside SQLiteBackend.

Covers:
- ANN is populated on ``write_memory`` and queryable immediately.
- ANN is rebuilt from ``memory_vec`` when the workspace is reopened.
- ``delete_memory_completely`` removes the id from the ANN index.
- ``use_ann_index=False`` falls back to sqlite-vec linear scan.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.store.ann_index import HNSWLIB_AVAILABLE
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


pytestmark = pytest.mark.skipif(
    not HNSWLIB_AVAILABLE,
    reason="hnswlib not installed; install with `pip install mnemoss[ann]`",
)


DIM = 4


def _memory(id: str, content: str = "x") -> Memory:
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


def _vec(values: list[float]) -> np.ndarray:
    a = np.array(values, dtype=np.float32)
    return a / np.linalg.norm(a)


async def _open(tmp_path: Path, *, use_ann: bool = True) -> SQLiteBackend:
    b = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=DIM,
        embedder_id="fake",
        use_ann_index=use_ann,
    )
    await b.open()
    return b


async def test_ann_populated_on_write(tmp_path: Path) -> None:
    b = await _open(tmp_path)
    try:
        await b.write_memory(_memory("a"), _vec([1.0, 0.0, 0.0, 0.0]))
        await b.write_memory(_memory("b"), _vec([0.9, 0.1, 0.0, 0.0]))
        await b.write_memory(_memory("c"), _vec([0.0, 0.0, 1.0, 0.0]))

        results = await b.vec_search(_vec([1.0, 0.0, 0.0, 0.0]), k=3, agent_id=None)
        ids = [mid for mid, _ in results]
        # Nearest-to-query order: a, b, c.
        assert ids[:2] == ["a", "b"]
        assert "c" in ids
    finally:
        await b.close()


async def test_ann_rehydrated_from_memory_vec_on_reopen(tmp_path: Path) -> None:
    b = await _open(tmp_path)
    try:
        await b.write_memory(_memory("a"), _vec([1.0, 0.0, 0.0, 0.0]))
        await b.write_memory(_memory("b"), _vec([0.0, 1.0, 0.0, 0.0]))
        await b.write_memory(_memory("c"), _vec([0.0, 0.0, 1.0, 0.0]))
    finally:
        await b.close()

    # Reopen — ANN should rehydrate from memory_vec.
    b2 = await _open(tmp_path)
    try:
        results = await b2.vec_search(_vec([1.0, 0.0, 0.0, 0.0]), k=3, agent_id=None)
        ids = [mid for mid, _ in results]
        assert ids[0] == "a"
        assert set(ids) == {"a", "b", "c"}
    finally:
        await b2.close()


async def test_delete_removes_from_ann(tmp_path: Path) -> None:
    b = await _open(tmp_path)
    try:
        await b.write_memory(_memory("a"), _vec([1.0, 0.0, 0.0, 0.0]))
        await b.write_memory(_memory("b"), _vec([0.9, 0.1, 0.0, 0.0]))

        await b.delete_memory_completely("a")

        results = await b.vec_search(_vec([1.0, 0.0, 0.0, 0.0]), k=5, agent_id=None)
        ids = [mid for mid, _ in results]
        assert "a" not in ids
        assert "b" in ids
    finally:
        await b.close()


async def test_use_ann_index_false_falls_back(tmp_path: Path) -> None:
    b = await _open(tmp_path, use_ann=False)
    try:
        await b.write_memory(_memory("a"), _vec([1.0, 0.0, 0.0, 0.0]))
        await b.write_memory(_memory("b"), _vec([0.0, 1.0, 0.0, 0.0]))

        # Fallback path (sqlite-vec linear scan) still returns correct order.
        results = await b.vec_search(_vec([1.0, 0.0, 0.0, 0.0]), k=2, agent_id=None)
        ids = [mid for mid, _ in results]
        assert ids[0] == "a"
        # Importantly, the _ann attribute stayed None.
        assert b._ann is None
    finally:
        await b.close()
