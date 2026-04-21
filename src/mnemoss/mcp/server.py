"""Wire Mnemoss tool implementations to a FastMCP server.

Tool signatures intentionally do NOT include ``agent_id`` as a
parameter. Scoping happens at launch time via ``MNEMOSS_AGENT_ID`` —
one MCP instance serves one agent, which matches the way Claude Code
and similar clients launch an MCP server per-session.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from mnemoss.mcp import tools
from mnemoss.mcp.config import MCPConfig


def create_mcp_server(
    backend: Any,
    *,
    config: MCPConfig | None = None,
    name: str = "mnemoss",
) -> FastMCP:
    """Build a FastMCP server that proxies tool calls to ``backend``.

    ``backend`` is any object with the ``Mnemoss``/``WorkspaceHandle``
    method shape — ``open_backend()`` yields one.
    """

    cfg = config if config is not None else MCPConfig()
    mcp = FastMCP(
        name,
        instructions=(
            "Mnemoss — ACT-R based long-term memory for agents. "
            "Use observe() to record messages, recall() to retrieve them, "
            "pin() to protect important ones, and dream() to consolidate."
        ),
    )

    agent = cfg.agent_id

    @mcp.tool()
    async def observe(
        role: str,
        content: str,
        session_id: str | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a message. ``role`` must be one of user / assistant /
        tool_call / tool_result for the message to be encoded into a
        Memory (other roles are written to the Raw Log only)."""

        return await tools.tool_observe(
            backend,
            role=role,
            content=content,
            agent_id=agent,
            session_id=session_id,
            turn_id=turn_id,
        )

    @mcp.tool()
    async def recall(
        query: str,
        k: int = 5,
        include_deep: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve up to ``k`` memories ranked by ACT-R activation.

        Results come sorted by score descending. Include ``include_deep``
        only when the query is explicitly about something long ago —
        the DEEP tier otherwise scans lazily from cues like 'long ago'.
        """

        return await tools.tool_recall(
            backend,
            query=query,
            k=k,
            agent_id=agent,
            include_deep=include_deep,
        )

    @mcp.tool()
    async def pin(memory_id: str) -> dict[str, Any]:
        """Pin a memory so the disposal pass never drops it."""

        return await tools.tool_pin(backend, memory_id=memory_id, agent_id=agent)

    @mcp.tool()
    async def explain_recall(query: str, memory_id: str) -> dict[str, float]:
        """Return the full ActivationBreakdown for one memory given a query.

        Useful for debugging why a memory did or did not surface. All
        terms of the ACT-R formula are included.
        """

        return await tools.tool_explain_recall(
            backend,
            query=query,
            memory_id=memory_id,
            agent_id=agent,
        )

    @mcp.tool()
    async def expand(
        memory_id: str,
        query: str | None = None,
        hops: int = 1,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Explicitly expand from one memory via the relation graph.

        Dig deeper into a specific memory on demand — an LLM override of
        the automatic same-topic heuristic that ``recall`` uses. Returns
        memories reachable through the relation graph from the seed,
        ranked by spreading activation. If ``query`` is omitted, the
        seed's own content is used as the matching term.
        """

        return await tools.tool_expand(
            backend,
            memory_id=memory_id,
            query=query,
            hops=hops,
            k=k,
            agent_id=agent,
        )

    @mcp.tool()
    async def dream(trigger: str = "session_end") -> dict[str, Any]:
        """Run a dream cycle. Valid triggers: idle, session_end, surprise,
        cognitive_load, nightly. Each trigger runs a subset of the 6-phase
        pipeline; nightly runs everything.
        """

        return await tools.tool_dream(backend, trigger=trigger, agent_id=agent)

    @mcp.tool()
    async def rebalance() -> dict[str, Any]:
        """Recompute ``idx_priority`` and tier for every memory."""

        return await tools.tool_rebalance(backend)

    @mcp.tool()
    async def dispose() -> dict[str, Any]:
        """Run P8 Dispose standalone — prune dead-low-activation and
        redundant memories. Fresh (<30 days), pinned, highly salient, or
        emotionally heavy memories are protected."""

        return await tools.tool_dispose(backend)

    @mcp.tool()
    async def tombstones(limit: int = 100) -> list[dict[str, Any]]:
        """List recent disposal audit entries, newest first."""

        return await tools.tool_tombstones(backend, agent_id=agent, limit=limit)

    @mcp.tool()
    async def tier_counts() -> dict[str, int]:
        """Count of memories in each of the four index tiers."""

        return await tools.tool_tier_counts(backend)

    @mcp.tool()
    async def export_markdown(min_idx_priority: float = 0.5) -> dict[str, str]:
        """Render the Markdown view of high-priority memories."""

        return await tools.tool_export_markdown(
            backend,
            agent_id=agent,
            min_idx_priority=min_idx_priority,
        )

    @mcp.tool()
    async def flush_session(session_id: str | None = None) -> dict[str, int]:
        """Force-close in-flight event buffers (e.g. at session end)."""

        return await tools.tool_flush_session(
            backend,
            agent_id=agent,
            session_id=session_id,
        )

    @mcp.tool()
    async def status() -> dict[str, Any]:
        """Return the workspace's operational snapshot — counts, last
        dream / observe / rebalance / dispose timestamps, embedder info,
        schema version."""

        return await tools.tool_status(backend)

    return mcp
