"""HTTP client for the Mnemoss REST API.

Mirrors the shape of ``mnemoss.client.Mnemoss`` so code written against
the library can swap to the SDK by changing the import. Every method
returns library types (``Memory``, ``Tombstone``, …) — framework
plugins don't need to know about the wire format.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import httpx

from mnemoss.core.types import Tombstone
from mnemoss.dream.dispose import DisposalStats
from mnemoss.dream.types import DreamReport
from mnemoss.formula.activation import ActivationBreakdown
from mnemoss.index import RebalanceStats
from mnemoss.recall import RecallResult
from mnemoss.sdk._parse import (
    parse_breakdown,
    parse_disposal_stats,
    parse_dream_report,
    parse_rebalance_stats,
    parse_recall_result,
    parse_tombstone,
)


class MnemossClient:
    """Low-level HTTP connection to a Mnemoss server.

    Use ``client.workspace(id)`` to get a ``WorkspaceHandle`` scoped to
    one workspace. The client is an ``async`` context manager — either
    ``async with`` it or call ``await client.close()`` when done.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Accept": "application/json"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    # ─── lifecycle ──────────────────────────────────────────────

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> MnemossClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    # ─── factories ──────────────────────────────────────────────

    def workspace(self, workspace_id: str) -> WorkspaceHandle:
        return WorkspaceHandle(self, workspace_id)

    async def health(self) -> bool:
        resp = await self._http.get("/health")
        resp.raise_for_status()
        return bool(resp.json().get("ok"))

    # ─── internal helpers ───────────────────────────────────────

    async def _post(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resp = await self._http.post(path, json=json, params=_clean_params(params))
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resp = await self._http.get(path, params=_clean_params(params))
        resp.raise_for_status()
        return resp.json()


class WorkspaceHandle:
    """Per-workspace SDK surface. Mirrors ``Mnemoss`` method-for-method."""

    def __init__(self, client: MnemossClient, workspace_id: str) -> None:
        self._client = client
        self._workspace_id = workspace_id

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    def for_agent(self, agent_id: str) -> AgentHandle:
        return AgentHandle(self, agent_id)

    # ─── core ───────────────────────────────────────────────────

    async def observe(
        self,
        role: str,
        content: str,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        body = {
            "role": role,
            "content": content,
            "session_id": session_id,
            "turn_id": turn_id,
            "parent_id": parent_id,
            "metadata": metadata,
        }
        resp = await self._client._post(
            self._path("observe"),
            json=body,
            params={"agent_id": agent_id},
        )
        return resp["memory_id"]

    async def recall(
        self,
        query: str,
        *,
        k: int = 5,
        agent_id: str | None = None,
        include_deep: bool = False,
    ) -> list[RecallResult]:
        resp = await self._client._post(
            self._path("recall"),
            json={"query": query, "k": k, "include_deep": include_deep},
            params={"agent_id": agent_id},
        )
        return [parse_recall_result(r) for r in resp["results"]]

    async def pin(self, memory_id: str, *, agent_id: str | None = None) -> None:
        await self._client._post(
            self._path("pin"),
            json={"memory_id": memory_id},
            params={"agent_id": agent_id},
        )

    async def explain_recall(
        self,
        query: str,
        memory_id: str,
        *,
        agent_id: str | None = None,
    ) -> ActivationBreakdown:
        resp = await self._client._post(
            self._path("explain"),
            json={"query": query, "memory_id": memory_id},
            params={"agent_id": agent_id},
        )
        return parse_breakdown(resp["breakdown"])

    # ─── dream / housekeeping ────────────────────────────────────

    async def dream(
        self,
        trigger: str = "session_end",
        *,
        agent_id: str | None = None,
    ) -> DreamReport:
        resp = await self._client._post(
            self._path("dream"),
            json={"trigger": trigger},
            params={"agent_id": agent_id},
        )
        return parse_dream_report(resp)

    async def rebalance(self) -> RebalanceStats:
        resp = await self._client._post(self._path("rebalance"))
        return parse_rebalance_stats(resp)

    async def dispose(self) -> DisposalStats:
        resp = await self._client._post(self._path("dispose"))
        return parse_disposal_stats(resp)

    async def tombstones(
        self,
        *,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[Tombstone]:
        resp = await self._client._get(
            self._path("tombstones"),
            params={"agent_id": agent_id, "limit": limit},
        )
        return [parse_tombstone(t) for t in resp["tombstones"]]

    async def tier_counts(self) -> dict[str, int]:
        resp = await self._client._get(self._path("tiers"))
        return dict(resp["tiers"])

    async def export_markdown(
        self,
        *,
        agent_id: str | None = None,
        min_idx_priority: float = 0.5,
    ) -> str:
        resp = await self._client._post(
            self._path("export"),
            json={"min_idx_priority": min_idx_priority},
            params={"agent_id": agent_id},
        )
        return str(resp["markdown"])

    async def flush_session(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        resp = await self._client._post(
            self._path("flush"),
            json={"session_id": session_id},
            params={"agent_id": agent_id},
        )
        return int(resp["flushed"])

    async def status(self) -> dict[str, Any]:
        """Return the workspace's operational snapshot."""

        return await self._client._get(self._path("status"))

    # ─── internal ───────────────────────────────────────────────

    def _path(self, suffix: str) -> str:
        return f"/workspaces/{self._workspace_id}/{suffix}"


class AgentHandle:
    """Sugar over a ``WorkspaceHandle`` that binds ``agent_id``.

    Matches ``mnemoss.client.AgentHandle`` — same method names, same
    signatures (minus the bound ``agent_id``). Workspace-scoped
    operations without a sensible agent dimension (``rebalance``,
    ``dispose``, ``tier_counts``, ``tombstones``, ``flush_session``,
    ``dream``) live only on ``WorkspaceHandle``.
    """

    def __init__(self, workspace: WorkspaceHandle, agent_id: str) -> None:
        self._ws = workspace
        self._agent_id = agent_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    async def observe(
        self,
        role: str,
        content: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        return await self._ws.observe(
            role,
            content,
            agent_id=self._agent_id,
            session_id=session_id,
            turn_id=turn_id,
            parent_id=parent_id,
            metadata=metadata,
        )

    async def recall(
        self,
        query: str,
        *,
        k: int = 5,
        include_deep: bool = False,
    ) -> list[RecallResult]:
        return await self._ws.recall(
            query,
            k=k,
            agent_id=self._agent_id,
            include_deep=include_deep,
        )

    async def pin(self, memory_id: str) -> None:
        await self._ws.pin(memory_id, agent_id=self._agent_id)

    async def explain_recall(
        self, query: str, memory_id: str
    ) -> ActivationBreakdown:
        return await self._ws.explain_recall(
            query, memory_id, agent_id=self._agent_id
        )

    async def export_markdown(self, *, min_idx_priority: float = 0.5) -> str:
        return await self._ws.export_markdown(
            agent_id=self._agent_id, min_idx_priority=min_idx_priority
        )


# ─── helpers ─────────────────────────────────────────────────────


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop ``None`` values so the server sees ``?agent_id=alice`` not
    ``?agent_id=None`` when the caller didn't pass one."""

    if params is None:
        return None
    return {k: v for k, v in params.items() if v is not None}
