"""Pure-function tool implementations.

Each function takes a ``backend`` (``Mnemoss`` or ``WorkspaceHandle``)
as the first argument so tests can pass a fake backend and exercise
the conversion logic without an MCP server. ``server.py`` re-exposes
these via ``FastMCP.tool`` decorators.

Return values are plain dicts / lists so MCP's JSON serialization
needs no custom codecs.
"""

from __future__ import annotations

from typing import Any

from mnemoss.core.types import Memory, Tombstone
from mnemoss.dream.dispose import DisposalStats
from mnemoss.dream.types import DreamReport
from mnemoss.formula.activation import ActivationBreakdown
from mnemoss.index import RebalanceStats
from mnemoss.recall import RecallResult

# ─── JSON-safe summaries ──────────────────────────────────────


def memory_summary(m: Memory) -> dict[str, Any]:
    """Lean Memory view suited to an agent's working-context budget.

    Skips embedding, access_history, derived links — the fields an MCP
    client asks about through ``explain`` or the REST API when it needs
    them.
    """

    return {
        "id": m.id,
        "content": m.content,
        "memory_type": m.memory_type.value,
        "created_at": m.created_at.isoformat(),
        "idx_priority": m.idx_priority,
        "index_tier": m.index_tier.value,
        "salience": m.salience,
        "agent_id": m.agent_id,
        "extracted_gist": m.extracted_gist,
    }


def recall_summary(r: RecallResult) -> dict[str, Any]:
    return {
        "memory": memory_summary(r.memory),
        "score": r.score,
        "source": r.source,
    }


def breakdown_summary(b: ActivationBreakdown) -> dict[str, float]:
    return {
        "base_level": b.base_level,
        "spreading": b.spreading,
        "matching": b.matching,
        "noise": b.noise,
        "total": b.total,
        "idx_priority": b.idx_priority,
        "w_f": b.w_f,
        "w_s": b.w_s,
        "query_bias": b.query_bias,
    }


def tombstone_summary(t: Tombstone) -> dict[str, Any]:
    return {
        "original_id": t.original_id,
        "agent_id": t.agent_id,
        "dropped_at": t.dropped_at.isoformat(),
        "reason": t.reason,
        "gist_snapshot": t.gist_snapshot,
        "b_at_drop": t.b_at_drop,
    }


def dream_summary(r: DreamReport) -> dict[str, Any]:
    return {
        "trigger": r.trigger.value,
        "started_at": r.started_at.isoformat(),
        "finished_at": r.finished_at.isoformat(),
        "duration_seconds": r.duration_seconds(),
        "agent_id": r.agent_id,
        "outcomes": [{"phase": o.phase.value, "status": o.status} for o in r.outcomes],
        "diary_path": str(r.diary_path) if r.diary_path is not None else None,
    }


def rebalance_summary(s: RebalanceStats) -> dict[str, Any]:
    return {
        "scanned": s.scanned,
        "migrated": s.migrated,
        "tier_before": {k.value: v for k, v in s.tier_before.items()},
        "tier_after": {k.value: v for k, v in s.tier_after.items()},
    }


def dispose_summary(s: DisposalStats) -> dict[str, Any]:
    return {
        "scanned": s.scanned,
        "disposed": s.disposed,
        "activation_dead": s.activation_dead,
        "redundant": s.redundant,
        "protected": s.protected,
        "disposed_ids": list(s.disposed_ids),
    }


# ─── tool functions ───────────────────────────────────────────


async def tool_observe(
    backend: Any,
    *,
    role: str,
    content: str,
    agent_id: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> dict[str, Any]:
    memory_id = await backend.observe(
        role=role,
        content=content,
        agent_id=agent_id,
        session_id=session_id,
        turn_id=turn_id,
    )
    return {"memory_id": memory_id, "encoded": memory_id is not None}


async def tool_recall(
    backend: Any,
    *,
    query: str,
    k: int = 5,
    agent_id: str | None = None,
    include_deep: bool = False,
    auto_expand: bool = True,
) -> list[dict[str, Any]]:
    results = await backend.recall(
        query,
        k=k,
        agent_id=agent_id,
        include_deep=include_deep,
        auto_expand=auto_expand,
    )
    return [recall_summary(r) for r in results]


async def tool_expand(
    backend: Any,
    *,
    memory_id: str,
    query: str | None = None,
    hops: int = 1,
    k: int = 5,
    agent_id: str | None = None,
) -> list[dict[str, Any]]:
    results = await backend.expand(
        memory_id,
        agent_id=agent_id,
        query=query,
        hops=hops,
        k=k,
    )
    return [recall_summary(r) for r in results]


async def tool_pin(
    backend: Any,
    *,
    memory_id: str,
    agent_id: str | None = None,
) -> dict[str, Any]:
    await backend.pin(memory_id, agent_id=agent_id)
    return {"ok": True, "memory_id": memory_id}


async def tool_explain_recall(
    backend: Any,
    *,
    query: str,
    memory_id: str,
    agent_id: str | None = None,
) -> dict[str, float]:
    breakdown = await backend.explain_recall(query, memory_id, agent_id=agent_id)
    return breakdown_summary(breakdown)


async def tool_dream(
    backend: Any,
    *,
    trigger: str = "session_end",
    agent_id: str | None = None,
) -> dict[str, Any]:
    report = await backend.dream(trigger=trigger, agent_id=agent_id)
    return dream_summary(report)


async def tool_rebalance(backend: Any) -> dict[str, Any]:
    stats = await backend.rebalance()
    return rebalance_summary(stats)


async def tool_dispose(backend: Any) -> dict[str, Any]:
    stats = await backend.dispose()
    return dispose_summary(stats)


async def tool_tombstones(
    backend: Any,
    *,
    agent_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    rows = await backend.tombstones(agent_id=agent_id, limit=limit)
    return [tombstone_summary(t) for t in rows]


async def tool_tier_counts(backend: Any) -> dict[str, int]:
    counts: dict[str, int] = await backend.tier_counts()
    return counts


async def tool_export_markdown(
    backend: Any,
    *,
    agent_id: str | None = None,
    min_idx_priority: float = 0.5,
) -> dict[str, str]:
    md = await backend.export_markdown(agent_id=agent_id, min_idx_priority=min_idx_priority)
    return {"markdown": md}


async def tool_flush_session(
    backend: Any,
    *,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, int]:
    n = await backend.flush_session(agent_id=agent_id, session_id=session_id)
    return {"flushed": n}


async def tool_status(backend: Any) -> dict[str, Any]:
    status: dict[str, Any] = await backend.status()
    return status
