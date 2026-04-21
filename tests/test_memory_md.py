"""memory.md export tests (Checkpoint O)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mnemoss import FakeEmbedder, Mnemoss, StorageParams
from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.export.markdown import render_memory_md

UTC = timezone.utc


def _mem(
    id: str,
    content: str,
    *,
    memory_type: MemoryType = MemoryType.EPISODE,
    idx_priority: float = 0.8,
    salience: float = 0.0,
    abstraction: float = 0.0,
    gist: str | None = None,
    agent_id: str | None = None,
) -> Memory:
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
        memory_type=memory_type,
        abstraction_level=abstraction,
        access_history=[now],
        idx_priority=idx_priority,
        index_tier=IndexTier.HOT if idx_priority > 0.7 else IndexTier.WARM,
        salience=salience,
        extracted_gist=gist,
    )


def test_empty_memory_list_produces_header_only() -> None:
    md = render_memory_md([])
    assert "# Memory — ambient" in md
    assert "no memories" in md.lower()


def test_priority_threshold_filters_out_dim_memories() -> None:
    memories = [
        _mem("m1", "important", idx_priority=0.9),
        _mem("m2", "dim", idx_priority=0.2),
    ]
    md = render_memory_md(memories, min_idx_priority=0.5)
    assert "important" in md
    assert "dim" not in md


def test_pinned_bypasses_threshold() -> None:
    memories = [_mem("pinned_but_dim", "should appear", idx_priority=0.1)]
    md = render_memory_md(
        memories, pinned_ids={"pinned_but_dim"}, min_idx_priority=0.5
    )
    assert "should appear" in md
    assert "📌" in md


def test_grouping_order_is_deterministic() -> None:
    memories = [
        _mem("ep", "an episode", memory_type=MemoryType.EPISODE, idx_priority=0.9),
        _mem("fact", "a fact", memory_type=MemoryType.FACT, idx_priority=0.9),
        _mem("ent", "an entity", memory_type=MemoryType.ENTITY, idx_priority=0.9),
        _mem("pat", "a pattern", memory_type=MemoryType.PATTERN, idx_priority=0.9),
    ]
    md = render_memory_md(memories)
    # Facts first, then Entities, Patterns, Episodes.
    assert md.index("## Facts") < md.index("## Entities")
    assert md.index("## Entities") < md.index("## Patterns")
    assert md.index("## Patterns") < md.index("## Episodes")


def test_gist_preferred_over_raw_content() -> None:
    long = "a very long raw content " * 20
    memories = [
        _mem("m1", long, gist="Short gist of the content", idx_priority=0.9),
    ]
    md = render_memory_md(memories)
    assert "Short gist of the content" in md
    assert long not in md  # full text not included


def test_content_truncated_when_no_gist() -> None:
    long_text = "a" * 500
    memories = [_mem("m1", long_text, idx_priority=0.9)]
    md = render_memory_md(memories)
    # 120-char cap with ellipsis.
    assert "…" in md
    assert "a" * 500 not in md


def test_per_agent_title() -> None:
    md_ambient = render_memory_md([], agent_id=None)
    md_alice = render_memory_md([], agent_id="alice")
    assert "# Memory — ambient" in md_ambient
    assert "# Memory — agent `alice`" in md_alice


def test_sort_order_pinned_then_idx_priority() -> None:
    memories = [
        _mem("a", "high priority", idx_priority=0.95),
        _mem("b", "mid priority", idx_priority=0.7),
        _mem("c", "dim but pinned", idx_priority=0.3),
    ]
    md = render_memory_md(memories, pinned_ids={"c"})
    # Pinned comes first.
    assert md.index("dim but pinned") < md.index("high priority")


# ─── end-to-end via Mnemoss.export_markdown ──────────────────────


def _mnemoss(tmp_path: Path) -> Mnemoss:
    return Mnemoss(
        workspace="t",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
    )


async def test_export_markdown_from_client(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="Alice met Bob at the cafe")
        md = await mem.export_markdown()
        # Fresh memories have idx_priority ≈ σ(η_0) ≈ 0.73 → above 0.5 default.
        assert "Alice met Bob at the cafe" in md or "Alice" in md
    finally:
        await mem.close()


async def test_export_markdown_per_agent(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    alice = mem.for_agent("alice")
    try:
        await alice.observe(role="user", content="alice private")
        await mem.observe(role="user", content="ambient note")

        alice_md = await alice.export_markdown()
        ambient_md = await mem.export_markdown()

        assert "alice private" in alice_md
        assert "ambient note" in alice_md  # ambient visible to agent
        assert "alice private" not in ambient_md  # ambient caller doesn't see agent's
        assert "ambient note" in ambient_md
    finally:
        await mem.close()
