"""T2 — Mnemoss.status() + structured logging.

Covers:
1. Library-level status() snapshot.
2. REST /status endpoint.
3. SDK client.status() parses the response.
4. MCP tool exposure.
5. Structured log lines emitted on key operations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from mnemoss import FakeEmbedder, Mnemoss, StorageParams
from mnemoss.core.config import SCHEMA_VERSION
from mnemoss.mcp import create_mcp_server
from mnemoss.sdk import MnemossClient
from mnemoss.server import ServerConfig, create_app


def _mnemoss(tmp_path: Path) -> Mnemoss:
    return Mnemoss(
        workspace="status_ws",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
    )


# ─── library-level ─────────────────────────────────────────────


async def test_status_fresh_workspace_has_expected_shape(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        snapshot = await mem.status()
    finally:
        await mem.close()

    assert snapshot["workspace"] == "status_ws"
    assert snapshot["schema_version"] == SCHEMA_VERSION
    assert snapshot["embedder"]["dim"] == 16
    assert snapshot["embedder"]["id"].startswith("fake")
    assert snapshot["memory_count"] == 0
    assert set(snapshot["tier_counts"]) == {"hot", "warm", "cold", "deep"}
    assert snapshot["tombstone_count"] == 0
    # Timestamps are None until the corresponding operation has run.
    for key in (
        "last_observe_at",
        "last_dream_at",
        "last_rebalance_at",
        "last_dispose_at",
    ):
        assert snapshot[key] is None


async def test_status_reflects_observe_and_dream(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="hi")
        await mem.dream(trigger="idle")
        await mem.rebalance()
        await mem.dispose()
        snapshot = await mem.status()
    finally:
        await mem.close()

    assert snapshot["memory_count"] == 1
    assert snapshot["last_observe_at"] is not None
    assert snapshot["last_dream_at"] is not None
    assert snapshot["last_dream_trigger"] == "idle"
    assert snapshot["last_rebalance_at"] is not None
    assert snapshot["last_dispose_at"] is not None


# ─── REST endpoint ──────────────────────────────────────────────


def test_rest_status_endpoint(tmp_path: Path) -> None:
    config = ServerConfig(
        embedder_override=FakeEmbedder(dim=16), storage_root=tmp_path
    )
    with TestClient(create_app(config)) as c:
        c.post(
            "/workspaces/rest_ws/observe",
            json={"role": "user", "content": "ping"},
        )
        r = c.get("/workspaces/rest_ws/status")

    assert r.status_code == 200
    body = r.json()
    assert body["workspace"] == "rest_ws"
    assert body["memory_count"] == 1
    assert body["last_observe_at"] is not None
    assert body["schema_version"] == SCHEMA_VERSION


# ─── SDK ────────────────────────────────────────────────────────


async def test_sdk_status_round_trip(tmp_path: Path) -> None:
    config = ServerConfig(
        embedder_override=FakeEmbedder(dim=16), storage_root=tmp_path
    )
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    try:
        async with MnemossClient(
            "http://testserver", transport=transport
        ) as client:
            ws = client.workspace("sdk_ws")
            await ws.observe(role="user", content="hi")
            snapshot = await ws.status()
    finally:
        await app.state.pool.close_all()

    assert snapshot["workspace"] == "sdk_ws"
    assert snapshot["memory_count"] == 1


# ─── MCP ─────────────────────────────────────────────────────────


async def test_mcp_status_tool_registered(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mcp = create_mcp_server(mem)
        names = {t.name for t in await mcp.list_tools()}
    finally:
        await mem.close()
    assert "status" in names


# ─── structured logging ─────────────────────────────────────────


async def test_observe_emits_structured_log(
    tmp_path: Path, caplog
) -> None:
    caplog.set_level(logging.INFO, logger="mnemoss.client")
    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="audit me")
    finally:
        await mem.close()

    records = [r for r in caplog.records if r.message == "observed"]
    assert records
    rec = records[0]
    assert rec.workspace == "status_ws"  # extra= field
    assert rec.role == "user"
    assert rec.encoded is True


async def test_recall_emits_structured_log(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="mnemoss.client")
    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="hi")
        caplog.clear()
        await mem.recall("hi", k=2)
    finally:
        await mem.close()

    records = [r for r in caplog.records if r.message == "recalled"]
    assert records
    rec = records[0]
    assert rec.workspace == "status_ws"
    assert rec.k == 2


async def test_dream_emits_structured_log(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="mnemoss.client")
    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="hi")
        caplog.clear()
        await mem.dream(trigger="idle")
    finally:
        await mem.close()

    records = [r for r in caplog.records if r.message == "dream"]
    assert records
    rec = records[0]
    assert rec.trigger == "idle"
    assert isinstance(rec.phases, list)
    assert rec.duration_seconds >= 0
