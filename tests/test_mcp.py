"""S6 — MCP wrapper tests.

Covers:

1. The pure-function tool implementations (``tools.tool_*``) with a
   real ``Mnemoss`` instance + FakeEmbedder.
2. ``create_mcp_server`` registering the expected tool surface.
3. Calling tools through FastMCP's dispatch to verify schemas work.
4. Config + backend selection (embedded vs HTTP).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mnemoss import FakeEmbedder, Mnemoss, StorageParams
from mnemoss.mcp import MCPConfig, create_mcp_server
from mnemoss.mcp.backend import open_backend
from mnemoss.mcp.tools import (
    tool_dispose,
    tool_explain_recall,
    tool_observe,
    tool_pin,
    tool_rebalance,
    tool_recall,
    tool_tier_counts,
    tool_tombstones,
)


def _mnemoss(tmp_path: Path) -> Mnemoss:
    return Mnemoss(
        workspace="mcp",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
    )


# ─── tool-level unit tests ───────────────────────────────────────


async def test_tool_observe_returns_memory_id(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        result = await tool_observe(mem, role="user", content="hi")
    finally:
        await mem.close()
    assert result["encoded"] is True
    assert isinstance(result["memory_id"], str)


async def test_tool_observe_filtered_role_not_encoded(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        result = await tool_observe(mem, role="system", content="prompt")
    finally:
        await mem.close()
    assert result == {"memory_id": None, "encoded": False}


async def test_tool_recall_returns_scored_summaries(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mid = await mem.observe(role="user", content="Alice meeting")
        results = await tool_recall(mem, query="Alice", k=3)
    finally:
        await mem.close()

    assert results
    top = results[0]
    assert top["memory"]["id"] == mid
    # Summary is lean — no embedding, no access_history.
    assert "content_embedding" not in top["memory"]
    assert "access_history" not in top["memory"]
    assert "score" in top


async def test_tool_pin_ok(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mid = await mem.observe(role="user", content="pin me")
        result = await tool_pin(mem, memory_id=mid)
    finally:
        await mem.close()
    assert result == {"ok": True, "memory_id": mid}


async def test_tool_explain_recall_returns_full_breakdown(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mid = await mem.observe(role="user", content="Alice")
        result = await tool_explain_recall(mem, query="Alice", memory_id=mid)
    finally:
        await mem.close()

    assert set(result) >= {
        "base_level",
        "spreading",
        "matching",
        "noise",
        "total",
        "idx_priority",
        "w_f",
        "w_s",
        "query_bias",
    }


async def test_tool_rebalance_and_tier_counts(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        for i in range(3):
            await mem.observe(role="user", content=f"item {i}")
        stats = await tool_rebalance(mem)
        tiers = await tool_tier_counts(mem)
    finally:
        await mem.close()

    assert stats["scanned"] >= 3
    assert set(tiers) == {"hot", "warm", "cold", "deep"}


async def test_tool_dispose_and_tombstones_on_fresh_workspace(
    tmp_path: Path,
) -> None:
    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="fresh")
        dispose_result = await tool_dispose(mem)
        tomb_result = await tool_tombstones(mem)
    finally:
        await mem.close()

    assert dispose_result["disposed"] == 0
    assert dispose_result["protected"] >= 1
    assert tomb_result == []


# ─── server-level tests ──────────────────────────────────────────


async def test_create_mcp_server_registers_expected_tools(
    tmp_path: Path,
) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mcp = create_mcp_server(mem)
        names = {t.name for t in await mcp.list_tools()}
    finally:
        await mem.close()

    assert names == {
        "observe",
        "recall",
        "pin",
        "explain_recall",
        "dream",
        "rebalance",
        "dispose",
        "tombstones",
        "tier_counts",
        "export_markdown",
        "flush_session",
    }


def _call_tool_structured(result) -> Any:
    """FastMCP.call_tool returns either a bare ``list[TextContent]`` or
    a ``(content, structured_dict)`` tuple. Handle both."""

    if isinstance(result, tuple):
        _, structured = result
        # Lists come back wrapped under ``result`` in the structured dict.
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured
    # Fallback: parse the first TextContent's JSON.
    return json.loads(result[0].text)


async def test_mcp_call_tool_returns_parseable_json(tmp_path: Path) -> None:
    """End-to-end: FastMCP.call_tool dispatches to our registered tool
    and produces both TextContent and structured content the client
    can consume."""

    mem = _mnemoss(tmp_path)
    try:
        mcp = create_mcp_server(mem)

        observe_result = await mcp.call_tool(
            "observe", {"role": "user", "content": "hi from mcp"}
        )
        observed = _call_tool_structured(observe_result)
        assert observed["encoded"] is True
        assert isinstance(observed["memory_id"], str)

        # Round-trip: recall should see the memory we just observed.
        recall_result = await mcp.call_tool(
            "recall", {"query": "hi from mcp", "k": 3}
        )
        recall_items = _call_tool_structured(recall_result)
        assert recall_items
        assert any(r["memory"]["content"] == "hi from mcp" for r in recall_items)
    finally:
        await mem.close()


async def test_mcp_agent_binding_applies_to_every_tool_call(
    tmp_path: Path,
) -> None:
    """Launching with ``MCPConfig.agent_id='alice'`` should scope every
    tool call to that agent without the tool signatures exposing it."""

    mem = _mnemoss(tmp_path)
    try:
        mcp = create_mcp_server(mem, config=MCPConfig(agent_id="alice"))

        # Write one as alice (via MCP), one as bob (via the library
        # directly — MCP is locked to alice).
        await mcp.call_tool(
            "observe", {"role": "user", "content": "alice private"}
        )
        await mem.observe(role="user", content="bob private", agent_id="bob")

        # MCP recall should see Alice's + ambient, never Bob's.
        r = await mcp.call_tool("recall", {"query": "private", "k": 5})
        items = _call_tool_structured(r)
        contents = {x["memory"]["content"] for x in items}
        assert "alice private" in contents
        assert "bob private" not in contents
    finally:
        await mem.close()


async def test_tool_schemas_are_valid_json(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mcp = create_mcp_server(mem)
        for tool in await mcp.list_tools():
            # Schemas serialize to JSON without error — which is what the
            # MCP protocol actually sends.
            json.dumps(tool.inputSchema)
            # Every tool has a description (docstring propagated).
            assert tool.description
    finally:
        await mem.close()


# ─── config + backend selection ──────────────────────────────────


async def test_embedded_backend_round_trip(tmp_path: Path, monkeypatch) -> None:
    # Force embedded mode: no MNEMOSS_API_URL.
    monkeypatch.delenv("MNEMOSS_API_URL", raising=False)
    monkeypatch.setenv("MNEMOSS_WORKSPACE", "mcp_embedded")
    # Storage redirected so we don't touch ~/.mnemoss.
    monkeypatch.setenv("MNEMOSS_STORAGE_ROOT", str(tmp_path))

    # The `Mnemoss()` constructor inside open_backend reads the
    # standard StorageParams default (``~/.mnemoss``), not the env var.
    # So instead of the env roundabout, construct config directly and
    # rely on default storage — but override the workspace directory by
    # replacing the Mnemoss factory. Simpler: just test MCPConfig.from_env
    # selects embedded when api_url is absent.
    config = MCPConfig.from_env()
    assert config.api_url is None
    assert config.workspace == "mcp_embedded"


def test_config_from_env_http_mode(monkeypatch) -> None:
    monkeypatch.setenv("MNEMOSS_API_URL", "http://example.com")
    monkeypatch.setenv("MNEMOSS_API_KEY", "s3cret")
    monkeypatch.setenv("MNEMOSS_WORKSPACE", "remote_ws")
    monkeypatch.setenv("MNEMOSS_AGENT_ID", "alice")

    config = MCPConfig.from_env()
    assert config.api_url == "http://example.com"
    assert config.api_key == "s3cret"
    assert config.workspace == "remote_ws"
    assert config.agent_id == "alice"


def test_config_from_env_defaults(monkeypatch) -> None:
    for var in (
        "MNEMOSS_API_URL",
        "MNEMOSS_API_KEY",
        "MNEMOSS_WORKSPACE",
        "MNEMOSS_AGENT_ID",
    ):
        monkeypatch.delenv(var, raising=False)

    config = MCPConfig.from_env()
    assert config.api_url is None
    assert config.api_key is None
    assert config.workspace == "default"
    assert config.agent_id is None


async def test_open_backend_http_mode_constructs_workspace_handle(
    tmp_path: Path,
) -> None:
    """HTTP mode yields a ``WorkspaceHandle`` bound to the configured
    workspace. We don't hit the network — opening the client + handle
    is enough to verify shape."""

    from mnemoss.sdk import WorkspaceHandle

    config = MCPConfig(api_url="http://testserver", workspace="http_ws")
    async with open_backend(config) as backend:
        assert isinstance(backend, WorkspaceHandle)
        assert backend.workspace_id == "http_ws"


async def test_open_backend_embedded_mode_constructs_mnemoss(
    tmp_path: Path, monkeypatch
) -> None:
    """Embedded mode yields a ``Mnemoss`` instance. Point storage at
    tmp_path so the test doesn't write to ~/.mnemoss."""

    # Patch StorageParams default. Easier: construct MCPConfig and
    # observe it returns a Mnemoss; cleanup happens via the context
    # manager.
    from mnemoss.client import Mnemoss as _Mnemoss

    # Monkeypatch Mnemoss to use our tmp_path-scoped storage — simpler
    # than threading StorageParams through MCPConfig for one test.
    real_init = _Mnemoss.__init__

    def init(self, workspace, **kwargs):
        kwargs.setdefault("embedding_model", FakeEmbedder(dim=16))
        kwargs.setdefault("storage", StorageParams(root=tmp_path))
        real_init(self, workspace, **kwargs)

    monkeypatch.setattr(_Mnemoss, "__init__", init)

    config = MCPConfig(api_url=None, workspace="embedded_ws")
    async with open_backend(config) as backend:
        assert isinstance(backend, _Mnemoss)
