"""T5.D — input-hardening tests for the REST server.

Covers the three configurable caps set on ``ServerConfig``:

- ``max_content_length_bytes`` on observe → 413
- ``max_recall_k`` on recall         → 422
- ``max_metadata_bytes`` on observe  → 413

The defaults are lenient (64 KiB / k=100 / 16 KiB) so existing tests
don't trip them. These tests set tight caps per-case and prove the
enforcement fires before the memory pipeline sees the bad input.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mnemoss import FakeEmbedder
from mnemoss.server import ServerConfig, create_app


def _client(tmp_path: Path, **overrides: object) -> TestClient:
    config = ServerConfig(
        embedder_override=FakeEmbedder(dim=16),
        storage_root=tmp_path,
        **overrides,
    )
    return TestClient(create_app(config))


# ─── content length cap ────────────────────────────────────────────


def test_oversized_content_returns_413(tmp_path: Path) -> None:
    with _client(tmp_path, max_content_length_bytes=100) as c:
        resp = c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "a" * 500},
        )
    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert "content exceeds max size" in detail
    assert "500" in detail  # actual size surfaced


def test_content_at_cap_is_accepted(tmp_path: Path) -> None:
    with _client(tmp_path, max_content_length_bytes=100) as c:
        resp = c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "a" * 100},
        )
    assert resp.status_code == 200


def test_content_cap_is_utf8_byte_length_not_char_count(tmp_path: Path) -> None:
    """Emoji + CJK cost multiple UTF-8 bytes per code point. The cap
    is by byte so a 30-char message of 4-byte code points can exceed
    a 100-byte cap."""

    with _client(tmp_path, max_content_length_bytes=100) as c:
        # Each 🎉 is 4 bytes in UTF-8, so 30 of them is 120 bytes.
        resp = c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "🎉" * 30},
        )
    assert resp.status_code == 413


def test_content_cap_none_disables_check(tmp_path: Path) -> None:
    with _client(tmp_path, max_content_length_bytes=None) as c:
        resp = c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "a" * 10_000},
        )
    assert resp.status_code == 200


# ─── recall k cap ───────────────────────────────────────────────────


def test_oversized_k_returns_422(tmp_path: Path) -> None:
    with _client(tmp_path, max_recall_k=5) as c:
        resp = c.post(
            "/workspaces/ws/recall",
            json={"query": "hi", "k": 50},
        )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "k=50" in detail
    assert "5" in detail


def test_recall_k_at_cap_is_accepted(tmp_path: Path) -> None:
    with _client(tmp_path, max_recall_k=5) as c:
        # Seed something so recall has anything to do — it still must
        # not trip the cap even with empty state.
        c.post(
            "/workspaces/ws/observe",
            json={"role": "user", "content": "seed"},
        )
        resp = c.post(
            "/workspaces/ws/recall",
            json={"query": "seed", "k": 5},
        )
    assert resp.status_code == 200


def test_recall_k_cap_none_disables_check(tmp_path: Path) -> None:
    with _client(tmp_path, max_recall_k=None) as c:
        resp = c.post(
            "/workspaces/ws/recall",
            json={"query": "hi", "k": 10_000},
        )
    # No error from the cap. The pipeline may return an empty list,
    # but must not 422.
    assert resp.status_code == 200


# ─── metadata size cap ─────────────────────────────────────────────


def test_oversized_metadata_returns_413(tmp_path: Path) -> None:
    big_metadata = {"k": "v" * 500}  # JSON-serialized > 500 bytes
    with _client(tmp_path, max_metadata_bytes=100) as c:
        resp = c.post(
            "/workspaces/ws/observe",
            json={
                "role": "user",
                "content": "hi",
                "metadata": big_metadata,
            },
        )
    assert resp.status_code == 413
    assert "metadata exceeds max size" in resp.json()["detail"]


def test_metadata_under_cap_is_accepted(tmp_path: Path) -> None:
    with _client(tmp_path, max_metadata_bytes=100) as c:
        resp = c.post(
            "/workspaces/ws/observe",
            json={
                "role": "user",
                "content": "hi",
                "metadata": {"trace_id": "abc"},
            },
        )
    assert resp.status_code == 200


# ─── defaults ──────────────────────────────────────────────────────


def test_defaults_are_lenient_enough_for_normal_use(tmp_path: Path) -> None:
    """Default caps (64 KiB content / k=100 / 16 KiB metadata) must
    not trip on a realistic message. If a future code change makes
    the defaults tighter, this test fails loudly — we don't want to
    silently break existing callers."""

    with _client(tmp_path) as c:
        resp = c.post(
            "/workspaces/ws/observe",
            json={
                "role": "user",
                "content": "a typical agent message of modest length",
                "metadata": {"session": "s1", "trace_id": "abc"},
            },
        )
    assert resp.status_code == 200
