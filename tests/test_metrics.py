"""T3 — Prometheus metrics.

Each test uses a unique workspace name so label-keyed counter values
don't bleed across test runs. The server's global registry lives for
the whole test session; reads use the proper ``get_sample_value`` API
rather than poking at counter internals.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("prometheus_client")

from mnemoss import FakeEmbedder  # noqa: E402
from mnemoss.server import ServerConfig, create_app  # noqa: E402
from mnemoss.server.metrics import REGISTRY  # noqa: E402


def _client(tmp_path: Path, **overrides) -> TestClient:
    config = ServerConfig(
        embedder_override=FakeEmbedder(dim=16),
        storage_root=tmp_path,
        **overrides,
    )
    return TestClient(create_app(config))


def _count(name: str, **labels) -> float:
    """``get_sample_value`` returns the current value for a
    name+labels tuple, or ``None`` if that series hasn't been touched."""

    value = REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


# ─── counters ────────────────────────────────────────────────────


def test_observe_increments_observes_total(tmp_path: Path) -> None:
    ws = "metrics_obs_ws"
    before = _count("mnemoss_observes_total", workspace=ws, encoded="true")

    with _client(tmp_path) as c:
        c.post(
            f"/workspaces/{ws}/observe",
            json={"role": "user", "content": "a"},
        )
        c.post(
            f"/workspaces/{ws}/observe",
            json={"role": "user", "content": "b"},
        )

    after = _count("mnemoss_observes_total", workspace=ws, encoded="true")
    assert after - before == 2


def test_observe_labels_encoded_false_on_filtered_role(tmp_path: Path) -> None:
    ws = "metrics_filtered_ws"
    before = _count("mnemoss_observes_total", workspace=ws, encoded="false")

    with _client(tmp_path) as c:
        c.post(
            f"/workspaces/{ws}/observe",
            json={"role": "system", "content": "prompt"},
        )

    after = _count("mnemoss_observes_total", workspace=ws, encoded="false")
    assert after - before == 1


def test_recall_increments_recalls_total(tmp_path: Path) -> None:
    ws = "metrics_recall_ws"
    before = _count("mnemoss_recalls_total", workspace=ws)

    with _client(tmp_path) as c:
        c.post(
            f"/workspaces/{ws}/observe",
            json={"role": "user", "content": "hi"},
        )
        c.post(
            f"/workspaces/{ws}/recall",
            json={"query": "hi", "k": 3},
        )

    after = _count("mnemoss_recalls_total", workspace=ws)
    assert after - before == 1


def test_dream_increments_dreams_total_with_trigger_label(
    tmp_path: Path,
) -> None:
    ws = "metrics_dream_ws"
    before = _count(
        "mnemoss_dreams_total", workspace=ws, trigger="idle"
    )

    with _client(tmp_path) as c:
        c.post(
            f"/workspaces/{ws}/observe",
            json={"role": "user", "content": "hi"},
        )
        c.post(f"/workspaces/{ws}/dream", json={"trigger": "idle"})

    after = _count("mnemoss_dreams_total", workspace=ws, trigger="idle")
    assert after - before == 1


def test_dispose_counter_stays_zero_on_fresh_workspace(tmp_path: Path) -> None:
    """Fresh memories are age-protected; the disposal counter shouldn't
    move even though the endpoint ran."""

    ws = "metrics_dispose_ws"
    before_dead = _count(
        "mnemoss_disposals_total", workspace=ws, reason="activation_dead"
    )
    before_redundant = _count(
        "mnemoss_disposals_total", workspace=ws, reason="redundant"
    )

    with _client(tmp_path) as c:
        c.post(
            f"/workspaces/{ws}/observe",
            json={"role": "user", "content": "hi"},
        )
        c.post(f"/workspaces/{ws}/dispose")

    after_dead = _count(
        "mnemoss_disposals_total", workspace=ws, reason="activation_dead"
    )
    after_redundant = _count(
        "mnemoss_disposals_total", workspace=ws, reason="redundant"
    )
    assert after_dead == before_dead
    assert after_redundant == before_redundant


# ─── /metrics endpoint ──────────────────────────────────────────


def test_metrics_endpoint_returns_prometheus_text(tmp_path: Path) -> None:
    ws = "metrics_endpoint_ws"
    with _client(tmp_path) as c:
        c.post(
            f"/workspaces/{ws}/observe",
            json={"role": "user", "content": "hi"},
        )
        r = c.get("/metrics")

    assert r.status_code == 200
    # Prometheus text format starts each metric family with HELP/TYPE.
    body = r.text
    assert "mnemoss_observes_total" in body
    assert "mnemoss_observe_duration_seconds" in body
    # Content-Type must be the Prometheus exposition format.
    assert r.headers["content-type"].startswith("text/plain")


def test_metrics_endpoint_refreshes_memory_count_gauge(tmp_path: Path) -> None:
    ws = "metrics_gauge_ws"
    with _client(tmp_path) as c:
        c.post(
            f"/workspaces/{ws}/observe",
            json={"role": "user", "content": "x"},
        )
        c.post(
            f"/workspaces/{ws}/observe",
            json={"role": "user", "content": "y"},
        )
        # Scrape populates the gauge.
        c.get("/metrics")

    hot_count = _count("mnemoss_memory_count", workspace=ws, tier="hot")
    assert hot_count == 2


def test_metrics_endpoint_is_unauthenticated_by_default(tmp_path: Path) -> None:
    """The /metrics endpoint uses the standard Prometheus scrape pattern
    (no per-request auth). We verify it doesn't get caught by the bearer
    middleware when the server is configured with an API key."""

    with _client(tmp_path, api_key="s3cret") as c:
        r = c.get("/metrics")
    assert r.status_code == 200
