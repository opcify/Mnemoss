"""S4 — SDK ⇄ server end-to-end parity.

Uses ``httpx.ASGITransport`` so the SDK talks to a real FastAPI app
in-process — full request + response serialization runs without opening
a socket. This is stronger than the MockTransport suite in ``test_sdk``
because it catches any shape mismatch between DTO and parse helpers.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from mnemoss import (
    FakeEmbedder,
    FormulaParams,
    MemoryType,
    Mnemoss,
    MockLLMClient,
    StorageParams,
)
from mnemoss.sdk import MnemossClient
from mnemoss.server import ServerConfig, create_app


def _canned(_prompt: str) -> dict:
    return {
        "summary": {
            "memory_type": "fact",
            "content": "consolidated fact",
            "abstraction_level": 0.6,
            "aliases": [],
        },
        "refinements": [
            {
                "index": 1,
                "gist": "refined",
                "entities": [],
                "time": None,
                "location": None,
                "participants": [],
            }
        ],
        "patterns": [],
    }


def _asgi_client(tmp_path: Path, **overrides) -> tuple[MnemossClient, object]:
    config = ServerConfig(
        embedder_override=FakeEmbedder(dim=16),
        storage_root=tmp_path,
        **overrides,
    )
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    return MnemossClient("http://testserver", transport=transport), app


@pytest.fixture
async def asgi(tmp_path: Path):
    client, app = _asgi_client(tmp_path)
    try:
        yield client, app
    finally:
        await client.close()
        await app.state.pool.close_all()


async def test_observe_recall_pin_round_trip(asgi) -> None:
    client, _ = asgi
    ws = client.workspace("e2e")

    mid = await ws.observe(role="user", content="Alice meeting at 4:20")
    assert mid is not None

    results = await ws.recall("Alice", k=3)
    assert results
    assert any(r.memory.id == mid for r in results)

    # Breakdown survives the round-trip intact.
    assert results[0].breakdown.total == pytest.approx(results[0].score)

    # Pin doesn't raise.
    await ws.pin(mid)


async def test_explain_matches_library_shape(tmp_path: Path) -> None:
    """SDK and library return equivalent ActivationBreakdowns.

    Exact equality is not possible — ``base_level`` decays with the
    wall clock and several seconds pass between the two calls. Instead:
    pin down the time-invariant terms (matching, idx_priority, w_f/w_s,
    query_bias) and check base_level to loose tolerance.
    """

    lib = Mnemoss(
        workspace="parity",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        formula=FormulaParams(noise_scale=0.0),
    )
    try:
        mid = await lib.observe(role="user", content="Alice")
        lib_breakdown = await lib.explain_recall("Alice", mid)
    finally:
        await lib.close()

    client, app = _asgi_client(
        tmp_path, llm=None
    )  # same formula defaults → matching/idx_priority match
    try:
        sdk_breakdown = await client.workspace("parity").explain_recall("Alice", mid)
    finally:
        await client.close()
        await app.state.pool.close_all()

    # Only query_bias is purely input-derived and truly time-invariant.
    assert sdk_breakdown.query_bias == pytest.approx(lib_breakdown.query_bias)
    # Everything else drifts a hair with the wall clock (B_i → idx_priority
    # → matching → w_f/w_s), but clock drift over a one-second test run
    # bounds the error well under 1e-3.
    for field in ("base_level", "idx_priority", "matching", "w_f", "w_s"):
        assert getattr(sdk_breakdown, field) == pytest.approx(
            getattr(lib_breakdown, field), abs=1e-3
        ), f"{field} drifted too far"


async def test_agent_handle_isolation(asgi) -> None:
    client, _ = asgi
    ws = client.workspace("iso")
    alice = ws.for_agent("alice")
    bob = ws.for_agent("bob")

    await alice.observe(role="user", content="alice private plan")
    await bob.observe(role="user", content="bob grocery list")
    await ws.observe(role="user", content="shared schedule")

    # Bob must not see Alice's private content.
    bob_contents = {r.memory.content for r in await bob.recall("plan", k=5)}
    assert "alice private plan" not in bob_contents

    # Alice sees her own plus ambient.
    alice_contents = {r.memory.content for r in await alice.recall("plan", k=5)}
    assert "alice private plan" in alice_contents


async def test_dream_e2e_runs_phases_and_diary(tmp_path: Path) -> None:
    client, app = _asgi_client(tmp_path, llm=MockLLMClient(callback=_canned))
    try:
        ws = client.workspace("dream_ws")
        for i in range(3):
            await ws.observe(role="user", content=f"Alice note {i}")

        report = await ws.dream(trigger="idle")
        assert report.trigger.value == "idle"
        phases = [o.phase.value for o in report.outcomes]
        assert phases == ["replay", "cluster", "consolidate", "relations"]
        # Diary path comes back as a proper Path, not None.
        assert report.diary_path is not None
        assert str(report.diary_path).endswith(".md")
    finally:
        await client.close()
        await app.state.pool.close_all()


async def test_rebalance_and_tiers_round_trip(asgi) -> None:
    client, _ = asgi
    ws = client.workspace("rebal")
    for i in range(4):
        await ws.observe(role="user", content=f"item {i}")

    stats = await ws.rebalance()
    assert stats.scanned >= 4
    tiers = await ws.tier_counts()
    # Every tier key is present, values are ints.
    assert set(tiers) == {"hot", "warm", "cold", "deep"}
    assert sum(tiers.values()) == stats.scanned


async def test_export_markdown_returns_sensible_document(asgi) -> None:
    client, _ = asgi
    ws = client.workspace("export_ws")
    await ws.observe(role="user", content="Alice scheduled lunch")
    await ws.observe(role="user", content="We're meeting at 1pm")

    md = await ws.export_markdown(min_idx_priority=0.0)
    assert isinstance(md, str)
    # The exporter always emits at least one section heading.
    assert "##" in md or "#" in md


async def test_dispose_on_fresh_workspace_protects_everything(asgi) -> None:
    client, _ = asgi
    ws = client.workspace("dispose_ws")
    await ws.observe(role="user", content="fresh memory")

    stats = await ws.dispose()
    assert stats.disposed == 0
    assert stats.protected >= 1


async def test_tombstones_empty_list_on_fresh_workspace(asgi) -> None:
    client, _ = asgi
    ws = client.workspace("tomb_ws")
    await ws.observe(role="user", content="hi")

    tombs = await ws.tombstones()
    assert tombs == []


async def test_parity_memory_type_enum(asgi) -> None:
    """DTO → Memory conversion must preserve MemoryType as the enum,
    not the string — framework plugins rely on isinstance checks."""

    client, _ = asgi
    ws = client.workspace("enum_ws")
    await ws.observe(role="user", content="content")

    results = await ws.recall("content", k=1)
    assert results
    assert isinstance(results[0].memory.memory_type, MemoryType)
    assert results[0].memory.memory_type is MemoryType.EPISODE


async def test_auth_flows_end_to_end(tmp_path: Path) -> None:
    """Bearer token on the SDK must match the server config to pass."""

    config = ServerConfig(
        embedder_override=FakeEmbedder(dim=16),
        storage_root=tmp_path,
        api_key="s3cret",
    )
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    try:
        # Wrong key → 401.
        bad = MnemossClient("http://testserver", api_key="wrong", transport=transport)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await bad.workspace("ws").observe(role="user", content="x")
        finally:
            await bad.close()

        # Correct key → ok.
        good = MnemossClient(
            "http://testserver", api_key="s3cret", transport=transport
        )
        try:
            mid = await good.workspace("ws").observe(role="user", content="ok")
            assert mid is not None
        finally:
            await good.close()
    finally:
        await app.state.pool.close_all()
