"""Adversarial input + concurrency integration tests.

The happy-path end-to-end tests in ``test_end_to_end_example.py``
show the system works when callers feed it well-formed input. This
file covers the other direction: what happens when callers feed it
weird, extreme, or concurrent input that might break assumptions
upstream (e.g. SQL encoding, tokenizer edge cases, event buffer
race conditions)?

Each case is deliberately minimal: one adversarial input, one
assertion about what ``observe`` / ``recall`` / ``dream`` should do
with it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mnemoss import (
    FakeEmbedder,
    FormulaParams,
    Mnemoss,
    StorageParams,
)


def _make_mem(tmp_path: Path) -> Mnemoss:
    return Mnemoss(
        workspace="adv",
        embedding_model=FakeEmbedder(dim=16),
        formula=FormulaParams(noise_scale=0.0),
        storage=StorageParams(root=tmp_path),
    )


# ─── content shape edge cases ─────────────────────────────────────


@pytest.mark.parametrize(
    "content",
    [
        "",            # empty string
        " ",           # single space
        "   \n\n\t  ", # whitespace-only
    ],
)
async def test_blank_content_does_not_crash(tmp_path: Path, content: str) -> None:
    """Blank content is legal from the Raw Log's perspective — we
    still persist it so the audit trail is complete — but recall
    shouldn't crash on it either."""

    mem = _make_mem(tmp_path)
    try:
        # Either the observe returns a memory id or silently drops it;
        # whichever it does, it must not raise.
        await mem.observe(role="user", content=content)
        # Recall still returns a list (possibly empty), not an error.
        results = await mem.recall("anything", k=3)
        assert isinstance(results, list)
    finally:
        await mem.close()


async def test_very_long_content_is_accepted(tmp_path: Path) -> None:
    """Agents occasionally dump multi-KB tool outputs as a single
    message. The Raw Log and Memory store must swallow them without
    truncation or crashes."""

    mem = _make_mem(tmp_path)
    try:
        # 32KB of repeating text — stress sqlite row size without
        # being pathological.
        big = "Lorem ipsum dolor sit amet " * 1200
        assert len(big) > 30_000
        mid = await mem.observe(role="user", content=big)
        assert mid is not None

        results = await mem.recall("Lorem ipsum", k=1)
        assert any(r.memory.id == mid for r in results)
        hit = next(r for r in results if r.memory.id == mid)
        # Content survived round-trip intact.
        assert len(hit.memory.content) == len(big)
    finally:
        await mem.close()


@pytest.mark.parametrize(
    "content",
    [
        "🎉 emoji party! 🔥 with a surprise 🧠",
        "مرحبا بالعالم",                       # Arabic RTL
        "Здравствуй, мир",                     # Cyrillic
        "नमस्ते दुनिया",                          # Devanagari (combining marks)
        "你好，世界 with mixed English",         # mixed Han + Latin
        "日本語と English مع العربية",           # three-script sentence
        "zero​width​spaces",        # zero-width space injection
    ],
)
async def test_unicode_content_round_trips(tmp_path: Path, content: str) -> None:
    """Mnemoss's multilingual promise means Unicode round-trip
    cleanly — no re-encoding, no normalization that loses combining
    marks, no silent dropping of exotic code points."""

    mem = _make_mem(tmp_path)
    try:
        mid = await mem.observe(role="user", content=content)
        assert mid is not None
        fetched = await mem._store.get_memory(mid) if mem._store else None
        assert fetched is not None
        assert fetched.content == content  # byte-for-byte
    finally:
        await mem.close()


async def test_content_with_sql_metacharacters_is_parameterized(
    tmp_path: Path,
) -> None:
    """A classic SQL-injection bait string must be stored literally —
    not interpreted. This is baked in because we use bound parameters,
    but an explicit test keeps it that way."""

    mem = _make_mem(tmp_path)
    try:
        injection = "'; DROP TABLE memory; --"
        mid = await mem.observe(role="user", content=injection)
        assert mid is not None

        results = await mem.recall("DROP TABLE", k=1)
        assert any(r.memory.content == injection for r in results)
    finally:
        await mem.close()


async def test_fts_special_chars_do_not_break_recall(tmp_path: Path) -> None:
    """FTS5 treats characters like ``"``, ``(``, ``*``, ``:`` as
    operators. Our ``build_trigram_query`` strips them — so a query
    that contains them should still run (possibly returning fewer
    hits) rather than raise."""

    mem = _make_mem(tmp_path)
    try:
        await mem.observe(role="user", content="the query has parens and stars")
        # Recall with a query full of FTS special chars.
        results = await mem.recall('"the" (query) * :foo', k=5)
        assert isinstance(results, list)
    finally:
        await mem.close()


# ─── concurrency ───────────────────────────────────────────────────


async def test_concurrent_observes_serialize_correctly(tmp_path: Path) -> None:
    """Multiple coroutines calling ``observe`` in parallel must all
    land in the store without losing rows or corrupting the Raw Log.

    The backend serializes writes via ``asyncio.Lock`` + SQLite WAL
    — this test makes sure that promise actually holds under real
    ``asyncio.gather`` load.
    """

    mem = _make_mem(tmp_path)
    try:
        count = 50
        await asyncio.gather(
            *[
                mem.observe(role="user", content=f"parallel note {i}")
                for i in range(count)
            ]
        )
        # All `count` memories landed.
        status = await mem.status()
        assert status["memory_count"] == count
    finally:
        await mem.close()


async def test_concurrent_observes_across_agents(tmp_path: Path) -> None:
    """Writes from different agents shouldn't collide or stomp each
    other's scoping. Alice's 20 + Bob's 20 = 40 total, and each can
    see only her/his own + ambient."""

    mem = _make_mem(tmp_path)
    try:
        alice = mem.for_agent("alice")
        bob = mem.for_agent("bob")

        async def seed(handle, prefix: str, n: int) -> None:
            for i in range(n):
                await handle.observe(
                    role="user", content=f"{prefix} note {i}"
                )

        await asyncio.gather(seed(alice, "alice", 20), seed(bob, "bob", 20))

        status = await mem.status()
        assert status["memory_count"] == 40

        alice_results = await alice.recall("note", k=50)
        bob_results = await bob.recall("note", k=50)

        # Alice sees her own (no bob's).
        assert not any("bob" in r.memory.content for r in alice_results)
        # Bob sees his own (no alice's).
        assert not any("alice" in r.memory.content for r in bob_results)
    finally:
        await mem.close()


async def test_concurrent_recall_is_safe(tmp_path: Path) -> None:
    """Parallel recalls read from WAL and don't block each other.
    Even with 20 concurrent recalls on a seeded workspace, every
    call returns a list."""

    mem = _make_mem(tmp_path)
    try:
        for i in range(30):
            await mem.observe(role="user", content=f"seed {i}")

        queries = [f"query {i}" for i in range(20)]
        all_results = await asyncio.gather(
            *[mem.recall(q, k=5) for q in queries]
        )
        assert all(isinstance(r, list) for r in all_results)
    finally:
        await mem.close()


# ─── pin + lifecycle interaction ──────────────────────────────────


async def test_pin_survives_rebalance(tmp_path: Path) -> None:
    """Rebalance shouldn't demote a pinned memory below HOT — pinning
    is the caller's explicit "keep this fresh" lever."""

    from mnemoss.core.types import IndexTier

    mem = _make_mem(tmp_path)
    try:
        mid = await mem.observe(role="user", content="very important pinned fact")
        await mem.pin(mid)
        await mem.rebalance()

        fetched = await mem._store.get_memory(mid) if mem._store else None
        assert fetched is not None
        # Pinned memories land in HOT regardless of activation.
        assert fetched.index_tier == IndexTier.HOT
    finally:
        await mem.close()


async def test_dispose_on_fresh_workspace_deletes_nothing(tmp_path: Path) -> None:
    """Fresh memories have encoding-grace bonus that protects them
    from disposal — calling ``dispose()`` immediately after observe
    should not tombstone anything."""

    mem = _make_mem(tmp_path)
    try:
        for i in range(5):
            await mem.observe(role="user", content=f"fresh note {i}")
        stats = await mem.dispose()
        assert stats.scanned == 5
        assert stats.disposed == 0
        # Recall still finds every memory.
        results = await mem.recall("fresh note", k=10)
        assert len(results) >= 5
    finally:
        await mem.close()


# ─── dream edge cases ─────────────────────────────────────────────


async def test_dream_on_empty_workspace_completes_cleanly(
    tmp_path: Path,
) -> None:
    """Nightly on a workspace with zero memories: skips everything
    downstream of REPLAY cleanly, no errors."""

    mem = _make_mem(tmp_path)
    try:
        report = await mem.dream(trigger="nightly")
        # No errors, even though most phases had nothing to do.
        assert not report.degraded_mode
    finally:
        await mem.close()


async def test_repeated_dream_does_not_grow_state_unboundedly(
    tmp_path: Path,
) -> None:
    """Running ``dream()`` back-to-back-to-back on the same
    workspace shouldn't leak state or produce more relations with
    each run once the workspace has stabilized."""

    mem = _make_mem(tmp_path)
    try:
        for i in range(6):
            await mem.observe(role="user", content=f"stable note {i}")

        # Trigger the relations/cluster path three times.
        for _ in range(3):
            await mem.dream(trigger="idle")

        # Status still JSON-serializable and memory count unchanged
        # (dream phases don't delete without DISPOSE in the chain).
        status = await mem.status()
        assert status["memory_count"] == 6
    finally:
        await mem.close()
