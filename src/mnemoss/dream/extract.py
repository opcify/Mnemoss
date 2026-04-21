"""P3 Extract — LLM distills a cluster into a new higher-abstraction Memory.

Contract:

- Input: a list of related ``Memory`` (one cluster) + an ``LLMClient``.
- Output: a new ``Memory`` with ``memory_type ∈ {FACT, ENTITY, PATTERN}``,
  ``abstraction_level ≈ 0.6``, ``derived_from`` linking back to the
  cluster members, and a fresh content string from the LLM. The new
  memory's embedding is computed by the caller (the runner has the
  embedder).

- Cross-agent promotion: when the cluster spans multiple agents (or
  mixes agents with ambient), the extracted memory is ambient
  (``agent_id=None``). Same goes for when only one agent is represented
  alongside ambient memories — ambient wins because the fact is already
  cross-agent reachable.

LLM failures never kill dream — on any exception (timeout, malformed
JSON, refusal) ``extract_from_cluster`` returns ``None`` and the
phase records the skip in its details.
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


def build_extract_prompt(cluster_members: list[Memory]) -> str:
    lines = [
        "The following memories are semantically related (they've been "
        "clustered together during consolidation):",
        "",
    ]
    for i, m in enumerate(cluster_members, 1):
        role = m.role or "note"
        lines.append(f"{i}. [{role}] {m.content}")
    lines.extend(
        [
            "",
            "Distill these memories into a single concise, durable fact, "
            "entity, or pattern. Keep the content language-neutral (use the "
            "same language the memories are in).",
            "",
            "Respond with a single JSON object of this shape:",
            "{",
            '  "memory_type": "fact" | "entity" | "pattern",',
            '  "content": "one short sentence capturing what\'s common",',
            '  "abstraction_level": 0.6,',
            '  "aliases": []',
            "}",
            "",
            "Guidelines:",
            "- 'fact' is a propositional statement (e.g. \"user prefers dark mode\").",
            "- 'entity' is a person, place, or thing (e.g. \"Alice: the user's manager\").",
            "- 'pattern' is a recurring behavior (e.g. \"user commits on Fridays\").",
            "- abstraction_level ∈ [0.5, 0.9]: 0.5 = concrete, 0.9 = highly abstract.",
        ]
    )
    return "\n".join(lines)


def _cluster_agent_id(cluster_members: list[Memory]) -> str | None:
    """Cross-agent promotion: if the cluster spans >1 agent or mixes with
    ambient memories, the extracted memory is ambient."""

    agents = {m.agent_id for m in cluster_members}
    if len(agents) == 1:
        # All members share one scope (could be one agent or ambient).
        return next(iter(agents))
    return None


async def extract_from_cluster(
    cluster_members: list[Memory],
    llm: LLMClient,
    params: FormulaParams,
    *,
    now: datetime | None = None,
) -> Memory | None:
    """Call the LLM, parse the response, and build the derived Memory.

    Returns ``None`` if the cluster is empty, the LLM call fails, or the
    response doesn't parse into a valid extraction.
    """

    if not cluster_members:
        return None

    prompt = build_extract_prompt(cluster_members)
    try:
        response = await llm.complete_json(prompt)
    except Exception as e:  # pragma: no cover - LLM errors vary by provider
        log.warning("extract_from_cluster: LLM failure: %s", e)
        return None

    content = (response.get("content") or "").strip()
    if not content:
        return None

    memory_type_str = response.get("memory_type", "fact")
    try:
        memory_type = MemoryType(memory_type_str)
    except ValueError:
        memory_type = MemoryType.FACT

    try:
        abstraction = float(response.get("abstraction_level", 0.6))
    except (TypeError, ValueError):
        abstraction = 0.6
    abstraction = max(0.0, min(1.0, abstraction))

    agent_id = _cluster_agent_id(cluster_members)

    # Inherit session from the first member that has one.
    session_id: str | None = None
    for m in cluster_members:
        if m.session_id:
            session_id = m.session_id
            break

    # Salience for the derived memory: the max of its sources (a fact
    # that descends from a striking episode should stay striking).
    salience = max((m.salience for m in cluster_members), default=0.0)

    t = now if now is not None else datetime.now(UTC)
    ip = initial_idx_priority(params)

    return Memory(
        id=str(ulid.new()),
        workspace_id=cluster_members[0].workspace_id,
        agent_id=agent_id,
        session_id=session_id,
        created_at=t,
        content=content,
        content_embedding=None,
        role=None,
        memory_type=memory_type,
        abstraction_level=abstraction,
        access_history=[t],
        last_accessed_at=None,
        salience=salience,
        emotional_weight=0.0,
        reminisced_count=0,
        idx_priority=ip,
        index_tier=idx_priority_to_tier(ip),
        derived_from=[m.id for m in cluster_members],
        derived_to=[],
        source_message_ids=[],
        source_context=_build_source_context(response, cluster_members),
    )


def _build_source_context(
    response: dict[str, Any], cluster_members: list[Memory]
) -> dict[str, Any]:
    aliases = response.get("aliases")
    ctx: dict[str, Any] = {
        "extracted_by": "dream_p3",
        "cluster_size": len(cluster_members),
    }
    if isinstance(aliases, list) and aliases:
        ctx["aliases"] = [str(a) for a in aliases]
    return ctx
