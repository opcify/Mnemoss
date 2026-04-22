"""S1 — server core endpoints (observe / recall / pin) + auth.

Uses FastAPI's ``TestClient`` (sync) against a server configured with a
``FakeEmbedder`` so the suite runs offline. Each test uses ``tmp_path``
as the storage root so runs don't bleed into ``~/.mnemoss``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mnemoss import FakeEmbedder
from mnemoss.server import ServerConfig, create_app


def _client(tmp_path: Path, **overrides) -> TestClient:
    config = ServerConfig(
        embedder_override=FakeEmbedder(dim=16),
        storage_root=tmp_path,
        **overrides,
    )
    return TestClient(create_app(config))


def test_health_is_open_without_auth(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_observe_returns_memory_id(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        resp = c.post(
            "/workspaces/testws/observe",
            json={"role": "user", "content": "hello world"},
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert isinstance(payload["memory_id"], str)
    assert payload["memory_id"]


def test_observe_filtered_role_returns_null_memory_id(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        resp = c.post(
            "/workspaces/testws/observe",
            json={"role": "system", "content": "prompt"},
        )
    assert resp.status_code == 200
    # ``system`` is not in the default encoded_roles → Raw Log only.
    assert resp.json() == {"memory_id": None}


def test_recall_returns_observed_memory(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        r1 = c.post(
            "/workspaces/testws/observe",
            json={"role": "user", "content": "Alice meeting at 4:20"},
        )
        mid = r1.json()["memory_id"]

        r2 = c.post(
            "/workspaces/testws/recall",
            json={"query": "Alice", "k": 3},
        )
    assert r2.status_code == 200
    results = r2.json()["results"]
    assert results
    assert any(r["memory"]["id"] == mid for r in results)
    first = results[0]
    # Wire format must expose the full breakdown for explain_recall parity.
    assert set(first["breakdown"]) >= {
        "base_level",
        "spreading",
        "matching",
        "noise",
        "total",
        "idx_priority",
    }


def test_per_agent_scoping_isolates_private_memories(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.post(
            "/workspaces/testws/observe?agent_id=alice",
            json={"role": "user", "content": "alice secret plan"},
        )
        c.post(
            "/workspaces/testws/observe?agent_id=bob",
            json={"role": "user", "content": "bob grocery list"},
        )

        # Bob recalls — should NOT see Alice's content.
        r = c.post(
            "/workspaces/testws/recall?agent_id=bob",
            json={"query": "secret plan", "k": 5},
        )
    contents = [res["memory"]["content"] for res in r.json()["results"]]
    assert "alice secret plan" not in contents


def test_pin_accepts_memory_id(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        mid = c.post(
            "/workspaces/testws/observe",
            json={"role": "user", "content": "pin me"},
        ).json()["memory_id"]

        resp = c.post(
            "/workspaces/testws/pin",
            json={"memory_id": mid},
        )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_auth_required_when_api_key_configured(tmp_path: Path) -> None:
    with _client(tmp_path, api_key="s3cret") as c:
        # No Authorization header → 401.
        r = c.post(
            "/workspaces/testws/observe",
            json={"role": "user", "content": "hi"},
        )
        assert r.status_code == 401

        # Wrong token → 401.
        r = c.post(
            "/workspaces/testws/observe",
            json={"role": "user", "content": "hi"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 401

        # Correct token → 200.
        r = c.post(
            "/workspaces/testws/observe",
            json={"role": "user", "content": "hi"},
            headers={"Authorization": "Bearer s3cret"},
        )
        assert r.status_code == 200


def test_health_is_always_open_even_when_auth_configured(tmp_path: Path) -> None:
    with _client(tmp_path, api_key="s3cret") as c:
        resp = c.get("/health")
    assert resp.status_code == 200


def test_allowed_workspaces_blocks_unknown_ids(tmp_path: Path) -> None:
    with _client(tmp_path, allowed_workspaces={"ok_ws"}) as c:
        r = c.post(
            "/workspaces/other_ws/observe",
            json={"role": "user", "content": "hi"},
        )
        assert r.status_code == 403
        r2 = c.post(
            "/workspaces/ok_ws/observe",
            json={"role": "user", "content": "hi"},
        )
        assert r2.status_code == 200


def test_workspace_lifespan_closes_instances(tmp_path: Path) -> None:
    """After the TestClient context manager exits, the pool should be
    empty — no leaked SQLite connections."""

    config = ServerConfig(embedder_override=FakeEmbedder(dim=16), storage_root=tmp_path)
    app = create_app(config)
    with TestClient(app) as c:
        c.post(
            "/workspaces/testws/observe",
            json={"role": "user", "content": "hi"},
        )
        assert "testws" in app.state.pool._instances

    assert app.state.pool._instances == {}


@pytest.mark.parametrize("bad_body", [{}, {"role": "user"}, {"content": "x"}])
def test_observe_validates_payload(tmp_path: Path, bad_body: dict) -> None:
    with _client(tmp_path) as c:
        r = c.post("/workspaces/testws/observe", json=bad_body)
    assert r.status_code == 422
