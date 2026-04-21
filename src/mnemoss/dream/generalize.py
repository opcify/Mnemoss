"""P6 Generalize — discover cross-episode patterns.

Where P3 Extract produces one fact/entity per cluster, P6 looks at
*all* the facts P3 produced this run and asks the LLM to identify
higher-level patterns that span clusters — recurring behaviors,
preferences, or regularities.

Output: zero or more ``memory_type=PATTERN`` memories with
``abstraction_level ≈ 0.85`` and ``derived_from`` pointing at the
facts they generalize. Like P3 these are cross-agent-aware: a pattern
derived from facts that span agents is promoted to ``agent_id=None``.

When there are fewer than ``min_input_facts`` P3 outputs (default 2)
P6 returns empty — there isn't enough signal to generalize.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import ulid

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory, MemoryType
from mnemoss.formula.idx_priority import idx_priority_to_tier, initial_idx_priority
from mnemoss.llm.client import LLMClient

UTC = timezone.utc
log = logging.getLogger(__name__)

DEFAULT_ABSTRACTION_LEVEL = 0.85


def build_generalize_prompt(facts: list[Memory]) -> str:
    lines = [
        "The following facts and entities were just consolidated from recent "
        "memory activity. Identify higher-level patterns that span multiple "
        "of them — recurring behaviors, preferences, or regularities that "
        "generalize beyond any single fact.",
        "",
    ]
    for i, m in enumerate(facts, 1):
        label = m.memory_type.value
        lines.append(f"{i}. [{label}] {m.content}")
    lines.extend(
        [
            "",
            "Return a JSON object with this shape:",
            "{",
            '  "patterns": [',
            "    {",
            '      "content": "pattern description, one sentence",',
            '      "derived_from": [1, 2]  // 1-indexed references to facts above',
            "    }",
            "  ]",
            "}",
            "",
            "Guidelines:",
            "- 0 patterns is a valid answer — don't invent.",
            "- Each pattern must reference ≥ 2 facts to count as cross-episode.",
            "- Use the same language as the input.",
        ]
    )
    return "\n".join(lines)


def _pattern_agent_id(sources: list[Memory]) -> str | None:
    agents = {m.agent_id for m in sources}
    return next(iter(agents)) if len(agents) == 1 else None


async def generalize_facts(
    facts: list[Memory],
    llm: LLMClient,
    params: FormulaParams,
    *,
    now: datetime | None = None,
    min_input_facts: int = 2,
) -> list[Memory]:
    """Return zero or more PATTERN Memory rows."""

    if len(facts) < min_input_facts:
        return []

    prompt = build_generalize_prompt(facts)
    try:
        response = await llm.complete_json(prompt)
    except Exception as e:  # pragma: no cover - LLM errors vary by provider
        log.warning("generalize_facts: LLM failure: %s", e)
        return []

    patterns = response.get("patterns", [])
    if not isinstance(patterns, list):
        return []

    t = now if now is not None else datetime.now(UTC)
    created: list[Memory] = []
    for entry in patterns:
        if not isinstance(entry, dict):
            continue
        content = (entry.get("content") or "").strip()
        if not content:
            continue
        raw_refs = entry.get("derived_from", [])
        if not isinstance(raw_refs, list):
            continue
        # Map 1-indexed LLM references back to fact ids; skip out-of-range.
        sources: list[Memory] = []
        for ref in raw_refs:
            if not isinstance(ref, int):
                continue
            idx = ref - 1
            if 0 <= idx < len(facts):
                sources.append(facts[idx])
        if len(sources) < 2:
            continue  # Pattern must span ≥ 2 facts.

        ip = initial_idx_priority(params)
        created.append(
            Memory(
                id=str(ulid.new()),
                workspace_id=facts[0].workspace_id,
                agent_id=_pattern_agent_id(sources),
                session_id=None,
                created_at=t,
                content=content,
                content_embedding=None,
                role=None,
                memory_type=MemoryType.PATTERN,
                abstraction_level=DEFAULT_ABSTRACTION_LEVEL,
                access_history=[t],
                last_accessed_at=None,
                salience=max((s.salience for s in sources), default=0.0),
                idx_priority=ip,
                index_tier=idx_priority_to_tier(ip),
                derived_from=[s.id for s in sources],
                derived_to=[],
                source_message_ids=[],
                source_context=_build_source_context(sources),
            )
        )
    return created


def _build_source_context(sources: list[Memory]) -> dict[str, Any]:
    return {
        "extracted_by": "dream_p6",
        "source_fact_count": len(sources),
    }
