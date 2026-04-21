"""Mnemoss client tests (FakeEmbedder, fast)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemoss import EncoderParams, FakeEmbedder, FormulaParams, Mnemoss, StorageParams


def _mnemoss(tmp_path: Path, **kwargs) -> Mnemoss:
    return Mnemoss(
        workspace="test",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        **kwargs,
    )


async def test_observe_and_recall_end_to_end(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mid = await mem.observe(role="user", content="Alice meeting at 4:20")
        assert mid is not None

        results = await mem.recall("Alice", k=3)
        assert results
        assert mid in {r.memory.id for r in results}
    finally:
        await mem.close()


async def test_role_filter_skips_memory_write(tmp_path: Path) -> None:
    mem = _mnemoss(
        tmp_path,
        encoder=EncoderParams(encoded_roles={"user"}),  # only user messages
    )
    try:
        # 'tool_call' is not in encoded_roles, so no Memory row is created.
        skipped = await mem.observe(role="tool_call", content="ignored content")
        assert skipped is None

        # But a user message still encodes.
        mid = await mem.observe(role="user", content="keep this")
        assert mid is not None

        results = await mem.recall("keep this", k=5)
        # Only the user message can show up.
        contents = [r.memory.content for r in results]
        assert "keep this" in contents
        assert "ignored content" not in contents
    finally:
        await mem.close()


async def test_for_agent_binds_agent_id(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        alice = mem.for_agent("alice")
        bob = mem.for_agent("bob")
        await alice.observe(role="user", content="alice-note")
        await bob.observe(role="user", content="bob-note")
        await mem.observe(role="user", content="shared-note")

        alice_hits = await alice.recall("note", k=5)
        contents = {r.memory.content for r in alice_hits}
        assert "alice-note" in contents
        assert "shared-note" in contents
        assert "bob-note" not in contents

        ambient_hits = await mem.recall("note", k=5)
        assert {r.memory.content for r in ambient_hits} <= {"shared-note"}
    finally:
        await mem.close()


async def test_pin_is_per_agent(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        shared = await mem.observe(role="user", content="shared memory")
        assert shared is not None
        alice = mem.for_agent("alice")
        await alice.pin(shared)
        # Internal plumbing — use the store directly to assert.
        assert mem._store is not None
        assert await mem._store.is_pinned(shared, "alice") is True
        assert await mem._store.is_pinned(shared, "bob") is False
    finally:
        await mem.close()


async def test_reopen_validates_schema(tmp_path: Path) -> None:
    mem1 = _mnemoss(tmp_path)
    try:
        await mem1.observe(role="user", content="hello")
    finally:
        await mem1.close()

    # Re-open with the same embedder: fine.
    mem2 = _mnemoss(tmp_path)
    try:
        results = await mem2.recall("hello", k=1)
        assert results
    finally:
        await mem2.close()

    # Re-open with a mismatched embedder dim raises.
    mem3 = Mnemoss(
        workspace="test",
        embedding_model=FakeEmbedder(dim=32),  # was 16
        storage=StorageParams(root=tmp_path),
    )
    with pytest.raises(Exception, match="mismatch"):
        await mem3.observe(role="user", content="whoops")


async def test_auto_create_on_first_observe(tmp_path: Path) -> None:
    # Constructor does no I/O; no DB file created yet.
    mem = _mnemoss(tmp_path)
    db = tmp_path / "workspaces" / "test" / "memory.sqlite"
    assert not db.exists()
    try:
        await mem.observe(role="user", content="bootstrap")
        assert db.exists()
    finally:
        await mem.close()


async def test_stubs_raise_not_implemented(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        # dream() landed in Stage 4 Checkpoint M.
        # status() and export_markdown() arrive in Stage 4 Checkpoint O.
        with pytest.raises(NotImplementedError):
            await mem.status()
        with pytest.raises(NotImplementedError):
            await mem.export_markdown()
    finally:
        await mem.close()


async def test_explain_recall_returns_breakdown(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path, formula=FormulaParams(noise_scale=0.0))
    try:
        mid = await mem.observe(role="user", content="Alice meeting")
        assert mid is not None
        br = await mem.explain_recall("Alice", mid)
        assert br is not None
        assert br.base_level > 0
        assert 0.0 <= br.w_f <= 1.0
    finally:
        await mem.close()


async def test_tier_counts_and_rebalance(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        for i in range(3):
            await mem.observe(role="user", content=f"content {i}")

        counts = await mem.tier_counts()
        assert counts["hot"] == 3
        assert counts["warm"] == 0
        assert counts["cold"] == 0
        assert counts["deep"] == 0

        stats = await mem.rebalance()
        assert stats.scanned == 3
        # Fresh memories stay HOT after rebalance.
        assert stats.migrated == 0
        assert stats.tier_after[next(iter(stats.tier_after))] >= 0  # valid dict
        assert sum(stats.tier_after.values()) == 3
    finally:
        await mem.close()
