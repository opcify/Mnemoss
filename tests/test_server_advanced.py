"""S2 — advanced server endpoints.

Covers explain / dream / rebalance / dispose / tombstones / tiers /
export / flush. Uses FakeEmbedder + a MockLLMClient with a canned
callback so dream exercises the LLM-dependent phases too.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mnemoss import FakeEmbedder, MockLLMClient
from mnemoss.server import ServerConfig, create_app


def _canned(_prompt: str) -> dict:
    """Shape-correct stand-in LLM response for the merged Consolidate phase."""

    return {
        "summary": {
            "memory_type": "fact",
            "content": "Alice fact",
            "abstraction_level": 0.6,
            "aliases": [],
        },
        "refinements": [
            {
                "index": 1,
                "gist": "refined gist",
                "entities": ["Alice"],
                "time": None,
                "location": None,
                "participants": ["Alice"],
            }
        ],
        "patterns": [{"content": "Alice-related pattern", "derived_from": [1, 2]}],
    }


def _client(tmp_path: Path, **overrides) -> TestClient:
    config = ServerConfig(
        embedder_override=FakeEmbedder(dim=16),
        storage_root=tmp_path,
        **overrides,
    )
    return TestClient(create_app(config))


def test_explain_returns_breakdown(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        mid = c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "Alice"},
        ).json()["memory_id"]

        r = c.post(
            "/workspaces/ws/explain",
            json={"query": "Alice", "memory_id": mid},
        )
    assert r.status_code == 200
    b = r.json()["breakdown"]
    assert set(b) >= {
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


def test_dream_idle_runs_without_llm(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "alpha"},
        )
        c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "beta"},
        )
        r = c.post("/workspaces/ws/dream", json={"trigger": "idle"})
    assert r.status_code == 200
    body = r.json()
    assert body["trigger"] == "idle"
    phases = [o["phase"] for o in body["outcomes"]]
    assert phases == ["replay", "cluster", "consolidate", "relations"]
    # Consolidate is skipped without an LLM.
    consolidate = next(o for o in body["outcomes"] if o["phase"] == "consolidate")
    assert consolidate["status"] == "skipped"


def test_dream_nightly_with_llm_runs_all_six_phases(tmp_path: Path) -> None:
    with _client(tmp_path, llm=MockLLMClient(callback=_canned)) as c:
        for i in range(5):
            c.post(
                "/workspaces/ws/observe",
                json={"role": "user", "content": f"Alice note {i}"},
            )
        r = c.post("/workspaces/ws/dream", json={"trigger": "nightly"})
    assert r.status_code == 200
    phases = [o["phase"] for o in r.json()["outcomes"]]
    assert phases == [
        "replay",
        "cluster",
        "consolidate",
        "relations",
        "rebalance",
        "dispose",
    ]


def test_dream_rejects_invalid_trigger(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "x"},
        )
        r = c.post("/workspaces/ws/dream", json={"trigger": "bogus"})
    assert r.status_code == 400


def test_rebalance_reports_counts(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "one"},
        )
        c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "two"},
        )
        r = c.post("/workspaces/ws/rebalance")
    assert r.status_code == 200
    body = r.json()
    assert body["scanned"] >= 2
    # Tier counts should cover all four tier names.
    assert set(body["tier_before"]) == {"hot", "warm", "cold", "deep"}
    assert set(body["tier_after"]) == {"hot", "warm", "cold", "deep"}


def test_dispose_on_fresh_memories_disposes_nothing(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "fresh"},
        )
        r = c.post("/workspaces/ws/dispose")
    assert r.status_code == 200
    body = r.json()
    assert body["disposed"] == 0
    assert body["protected"] >= 1


def test_tombstones_returns_empty_list_initially(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "hi"},
        )
        r = c.get("/workspaces/ws/tombstones")
    assert r.status_code == 200
    assert r.json() == {"tombstones": []}


def test_tiers_returns_all_four_keys(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "one"},
        )
        r = c.get("/workspaces/ws/tiers")
    assert r.status_code == 200
    assert set(r.json()["tiers"]) == {"hot", "warm", "cold", "deep"}


def test_export_returns_markdown(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "Alice meeting"},
        )
        r = c.post(
            "/workspaces/ws/export",
            json={"min_idx_priority": 0.0},
        )
    assert r.status_code == 200
    body = r.json()
    assert "markdown" in body
    assert isinstance(body["markdown"], str)


def test_flush_on_open_buffer_closes_event(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        # Explicit turn_id keeps the segmenter buffer open.
        c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "in flight", "turn_id": "open"},
        )
        r = c.post("/workspaces/ws/flush", json={})
    assert r.status_code == 200
    assert r.json()["flushed"] == 1


def test_auth_applied_to_every_advanced_endpoint(tmp_path: Path) -> None:
    with _client(tmp_path, api_key="k") as c:
        # Pin → explain → dream → rebalance → dispose → export → flush
        # → tombstones → tiers. Each should 401 without auth.
        for method, path, body in [
            ("POST", "/workspaces/ws/explain", {"query": "q", "memory_id": "m"}),
            ("POST", "/workspaces/ws/dream", {"trigger": "idle"}),
            ("POST", "/workspaces/ws/rebalance", None),
            ("POST", "/workspaces/ws/dispose", None),
            ("POST", "/workspaces/ws/export", {}),
            ("POST", "/workspaces/ws/flush", {}),
            ("GET", "/workspaces/ws/tombstones", None),
            ("GET", "/workspaces/ws/tiers", None),
        ]:
            r = c.request(method, path, json=body)
            assert r.status_code == 401, f"{method} {path} should be 401"
