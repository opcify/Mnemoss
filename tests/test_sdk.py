"""S3 — Python SDK tests.

Uses ``httpx.MockTransport`` so the SDK is exercised end-to-end without
a real server. Focus: URLs, bodies, auth headers, and (de)serialization
parity with the server DTOs.

An E2E parity test against a live ``TestClient`` server lands in S4.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from mnemoss.core.types import IndexTier, MemoryType
from mnemoss.dream.types import PhaseName, TriggerType
from mnemoss.sdk import MnemossClient

UTC = timezone.utc


def _memory_dto(
    *,
    memory_id: str = "mem_1",
    workspace_id: str = "ws",
    content: str = "hello",
) -> dict:
    return {
        "id": memory_id,
        "workspace_id": workspace_id,
        "agent_id": None,
        "session_id": "default",
        "created_at": "2026-04-21T12:00:00+00:00",
        "content": content,
        "role": "user",
        "memory_type": "episode",
        "abstraction_level": 0.0,
        "access_history": [],
        "last_accessed_at": None,
        "rehearsal_count": 0,
        "salience": 0.0,
        "emotional_weight": 0.0,
        "reminisced_count": 0,
        "index_tier": "hot",
        "idx_priority": 0.5,
        "extracted_gist": None,
        "extracted_entities": None,
        "extracted_time": None,
        "extracted_location": None,
        "extracted_participants": None,
        "extraction_level": 0,
        "cluster_id": None,
        "cluster_similarity": None,
        "is_cluster_representative": False,
        "derived_from": [],
        "derived_to": [],
        "source_message_ids": [],
        "source_context": {},
    }


def _breakdown_dto() -> dict:
    return {
        "base_level": 1.0,
        "spreading": 0.0,
        "matching": 2.0,
        "noise": 0.0,
        "total": 3.0,
        "idx_priority": 0.73,
        "w_f": 0.5,
        "w_s": 0.5,
        "query_bias": 1.0,
    }


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def test_observe_sends_correct_request(tmp_path) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"memory_id": "mem_1"})

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        ws = client.workspace("ws")
        mid = await ws.observe(role="user", content="hi")

    assert mid == "mem_1"
    assert seen["url"] == "http://test/workspaces/ws/observe"
    assert seen["body"]["role"] == "user"
    assert seen["body"]["content"] == "hi"


async def test_observe_passes_agent_id_as_query_param() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"memory_id": "mem_1"})

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        ws = client.workspace("ws")
        await ws.observe(role="user", content="hi", agent_id="alice")

    assert "agent_id=alice" in seen["url"]


async def test_observe_omits_agent_id_when_none() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"memory_id": "mem_1"})

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        await client.workspace("ws").observe(role="user", content="hi")

    # ?agent_id=None in the URL would be a bug.
    assert "agent_id" not in seen["url"]


async def test_bearer_token_sent_when_configured() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    async with MnemossClient(
        "http://test", api_key="s3cret", transport=_transport(handler)
    ) as client:
        await client.health()

    assert seen["auth"] == "Bearer s3cret"


async def test_recall_parses_memory_and_breakdown() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "memory": _memory_dto(content="alpha"),
                        "score": 4.2,
                        "breakdown": _breakdown_dto(),
                    }
                ]
            },
        )

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        results = await client.workspace("ws").recall("alpha", k=1)

    assert len(results) == 1
    r = results[0]
    assert r.memory.content == "alpha"
    assert r.memory.memory_type is MemoryType.EPISODE
    assert r.memory.index_tier is IndexTier.HOT
    assert r.memory.created_at == datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    assert r.score == pytest.approx(4.2)
    assert r.breakdown.total == pytest.approx(3.0)
    # Embeddings never cross the wire.
    assert r.memory.content_embedding is None


async def test_agent_handle_binds_agent_id_on_every_call() -> None:
    observed_url = []

    def handler(req: httpx.Request) -> httpx.Response:
        observed_url.append(str(req.url))
        if req.url.path.endswith("/observe"):
            return httpx.Response(200, json={"memory_id": "m"})
        if req.url.path.endswith("/recall"):
            return httpx.Response(200, json={"results": []})
        if req.url.path.endswith("/pin"):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        alice = client.workspace("ws").for_agent("alice")
        await alice.observe(role="user", content="x")
        await alice.recall("y", k=3)
        await alice.pin("mem_1")

    for url in observed_url:
        assert "agent_id=alice" in url


async def test_explain_parses_breakdown() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"breakdown": _breakdown_dto()})

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        b = await client.workspace("ws").explain_recall("q", "mem_1")

    assert b.total == pytest.approx(3.0)
    assert b.idx_priority == pytest.approx(0.73)


async def test_dream_parses_report() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "trigger": "idle",
                "started_at": "2026-04-21T12:00:00+00:00",
                "finished_at": "2026-04-21T12:00:05+00:00",
                "duration_seconds": 5.0,
                "agent_id": None,
                "outcomes": [
                    {"phase": "replay", "status": "ok", "details": {"selected": 3}},
                    {"phase": "cluster", "status": "ok", "details": {}},
                ],
                "diary_path": "/tmp/diary.md",
            },
        )

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        report = await client.workspace("ws").dream(trigger="idle")

    assert report.trigger is TriggerType.IDLE
    assert len(report.outcomes) == 2
    assert report.outcomes[0].phase is PhaseName.REPLAY
    assert report.outcomes[0].details["selected"] == 3
    assert str(report.diary_path) == "/tmp/diary.md"


async def test_tombstones_parses_list() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "tombstones": [
                    {
                        "original_id": "mem_gone",
                        "workspace_id": "ws",
                        "agent_id": None,
                        "dropped_at": "2026-04-21T12:00:00+00:00",
                        "reason": "redundant",
                        "gist_snapshot": "was here",
                        "b_at_drop": -2.5,
                        "source_message_ids": ["msg_1"],
                    }
                ]
            },
        )

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        tombs = await client.workspace("ws").tombstones()

    assert len(tombs) == 1
    assert tombs[0].reason == "redundant"
    assert tombs[0].source_message_ids == ["msg_1"]


async def test_rebalance_parses_tier_enums() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "scanned": 10,
                "migrated": 3,
                "tier_before": {"hot": 10, "warm": 0, "cold": 0, "deep": 0},
                "tier_after": {"hot": 7, "warm": 2, "cold": 1, "deep": 0},
            },
        )

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        stats = await client.workspace("ws").rebalance()

    assert stats.scanned == 10
    assert stats.migrated == 3
    assert stats.tier_before[IndexTier.HOT] == 10
    assert stats.tier_after[IndexTier.WARM] == 2


async def test_dispose_parses_stats() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "scanned": 5,
                "disposed": 1,
                "activation_dead": 1,
                "redundant": 0,
                "protected": 4,
                "disposed_ids": ["mem_x"],
            },
        )

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        stats = await client.workspace("ws").dispose()

    assert stats.disposed == 1
    assert stats.activation_dead == 1
    assert stats.disposed_ids == ["mem_x"]


async def test_tier_counts_returns_plain_dict() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"tiers": {"hot": 3, "warm": 1, "cold": 0, "deep": 0}},
        )

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        counts = await client.workspace("ws").tier_counts()

    assert counts == {"hot": 3, "warm": 1, "cold": 0, "deep": 0}


async def test_export_markdown_returns_string() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"markdown": "## Facts\n- x"})

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        md = await client.workspace("ws").export_markdown()

    assert md == "## Facts\n- x"


async def test_flush_returns_count() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"flushed": 2})

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        n = await client.workspace("ws").flush_session()

    assert n == 2


async def test_http_errors_raise() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad key"})

    async with MnemossClient("http://test", transport=_transport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.workspace("ws").observe(role="user", content="hi")
