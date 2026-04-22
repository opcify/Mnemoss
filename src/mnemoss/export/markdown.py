"""memory.md renderer.

Per §6.5 the output is a deterministic Markdown view of the memory
store — "what matters right now" — suitable for system-prompt
injection or just humans reading it.

Selection rule (strictly non-LLM):

- In-scope: the caller passes a pre-filtered list of memories so
  agent-scoping has already been applied upstream.
- Kept: ``idx_priority >= min_idx_priority`` **or** pinned (per-agent)
  **or** ``salience > 0.5``.
- Grouped by ``memory_type`` (facts first, then entities, then
  patterns, then episodes).
- Within each group, sorted by (pinned, idx_priority, abstraction_level).

Each line shows the most useful preview we have: ``extracted_gist`` if
the lazy-extraction pass has run, otherwise the first 120 characters
of ``content``. A ``📌`` marker flags pinned memories.
"""

from __future__ import annotations

from mnemoss.core.types import Memory, MemoryType

_PREVIEW_CHARS = 120
_GROUP_ORDER: tuple[MemoryType, ...] = (
    MemoryType.FACT,
    MemoryType.ENTITY,
    MemoryType.PATTERN,
    MemoryType.EPISODE,
)
_GROUP_TITLES: dict[MemoryType, str] = {
    MemoryType.FACT: "Facts",
    MemoryType.ENTITY: "Entities",
    MemoryType.PATTERN: "Patterns",
    MemoryType.EPISODE: "Episodes",
}


def render_memory_md(
    memories: list[Memory],
    *,
    pinned_ids: set[str] | None = None,
    agent_id: str | None = None,
    min_idx_priority: float = 0.5,
    salience_floor: float = 0.5,
) -> str:
    """Render the provided memories as a ``memory.md`` Markdown document.

    ``pinned_ids`` is the set of memory ids pinned in the caller's scope
    (``None`` counts as empty). Pin membership overrides the priority
    threshold so a pinned memory always appears.
    """

    pinned = pinned_ids or set()

    def _keep(m: Memory) -> bool:
        return m.idx_priority >= min_idx_priority or m.id in pinned or m.salience > salience_floor

    kept = [m for m in memories if _keep(m)]
    kept.sort(
        key=lambda m: (
            0 if m.id in pinned else 1,
            -m.idx_priority,
            -m.abstraction_level,
        )
    )

    title = _title_for(agent_id)
    lines: list[str] = [f"# {title}", ""]

    if not kept:
        lines.append("_(no memories exceed the export threshold yet)_")
        return "\n".join(lines) + "\n"

    grouped: dict[MemoryType, list[Memory]] = {}
    for m in kept:
        grouped.setdefault(m.memory_type, []).append(m)

    for mt in _GROUP_ORDER:
        bucket = grouped.get(mt)
        if not bucket:
            continue
        lines.append(f"## {_GROUP_TITLES[mt]} ({len(bucket)})")
        lines.append("")
        for m in bucket:
            lines.append(_render_line(m, pinned))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _title_for(agent_id: str | None) -> str:
    if agent_id is None:
        return "Memory — ambient"
    return f"Memory — agent `{agent_id}`"


def _render_line(memory: Memory, pinned_ids: set[str]) -> str:
    marker = "📌 " if memory.id in pinned_ids else "- "
    preview = memory.extracted_gist or _snippet(memory.content)
    suffix = f"  _(idx {memory.idx_priority:.2f})_"
    return f"{marker}{preview}{suffix}"


def _snippet(content: str) -> str:
    text = content.strip().replace("\n", " ")
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[: _PREVIEW_CHARS - 1].rstrip() + "…"
