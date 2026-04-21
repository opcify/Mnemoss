"""Tests for ``MnemossMemoryProvider``.

These tests run without hermes-agent installed: the provider imports
``agent.memory_provider.MemoryProvider`` at runtime, and because that
package isn't on ``sys.path`` in this repo, the fallback stub in
``mnemoss_hermes._stub_memory_provider`` is used instead. The stub
mirrors Hermes's abstract surface, so the same concrete provider
class exercises both paths.

The backend is mocked — we don't run a real Mnemoss here. The point is
to verify the Hermes-provider contract: lifecycle methods are wired to
the right Mnemoss calls with the right arguments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from mnemoss_hermes import MnemossMemoryProvider

# ─── mock Mnemoss backend ─────────────────────────────────────────


@dataclass
class _MockMemory:
    id: str
    content: str


@dataclass
class _MockResult:
    memory: _MockMemory
    score: float
    source: str = "direct"


class MockBackend:
    """Async-compatible fake of the Mnemoss surface the provider uses.

    Records every call so assertions can inspect routing. Canned return
    values live on the instance — tests populate them before exercising
    the provider.
    """

    def __init__(self) -> None:
        self.observed: list[dict[str, Any]] = []
        self.recall_calls: list[dict[str, Any]] = []
        self.expand_calls: list[dict[str, Any]] = []
        self.pin_calls: list[dict[str, Any]] = []
        self.dream_calls: list[dict[str, Any]] = []
        self.export_calls: list[dict[str, Any]] = []
        self.canned_recall: list[_MockResult] = []
        self.canned_expand: list[_MockResult] = []
        self.canned_export: str = ""

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
        self.observed.append(
            {
                "role": role,
                "content": content,
                "agent_id": agent_id,
                "session_id": session_id,
                "metadata": metadata,
            }
        )
        return f"mem-{len(self.observed)}"

    async def recall(
        self,
        query: str,
        *,
        k: int = 5,
        agent_id: str | None = None,
        include_deep: bool = False,
        auto_expand: bool = True,
    ) -> list[_MockResult]:
        self.recall_calls.append(
            {
                "query": query,
                "k": k,
                "agent_id": agent_id,
                "include_deep": include_deep,
            }
        )
        return list(self.canned_recall)

    async def expand(
        self,
        memory_id: str,
        *,
        agent_id: str | None = None,
        query: str | None = None,
        hops: int = 1,
        k: int = 5,
    ) -> list[_MockResult]:
        self.expand_calls.append(
            {"memory_id": memory_id, "agent_id": agent_id, "hops": hops, "k": k}
        )
        return list(self.canned_expand)

    async def pin(self, memory_id: str, *, agent_id: str | None = None) -> None:
        self.pin_calls.append({"memory_id": memory_id, "agent_id": agent_id})

    async def dream(
        self, trigger: str = "session_end", *, agent_id: str | None = None
    ) -> dict[str, Any]:
        self.dream_calls.append({"trigger": trigger, "agent_id": agent_id})
        return {"trigger": trigger, "status": "ok"}

    async def export_markdown(
        self,
        *,
        agent_id: str | None = None,
        min_idx_priority: float = 0.5,
    ) -> str:
        self.export_calls.append(
            {"agent_id": agent_id, "min_idx_priority": min_idx_priority}
        )
        return self.canned_export


def _make_provider(backend: MockBackend) -> MnemossMemoryProvider:
    """Construct a provider with the mock wired in directly.

    ``initialize`` normally builds the backend itself; for these tests
    we short-circuit by setting ``_backend`` after a bare init.
    """

    provider = MnemossMemoryProvider()
    provider.initialize("sess-1", agent_context="primary", platform="cli")
    # Replace whatever initialize produced (likely None on import error)
    # with our mock.
    provider._backend = backend
    provider._skipped_reason = None
    return provider


# ─── identity + availability ──────────────────────────────────────


def test_name_is_mnemoss() -> None:
    assert MnemossMemoryProvider().name == "mnemoss"


def test_is_available_reflects_import() -> None:
    # Mnemoss is installed in this repo's venv, so availability is True.
    assert MnemossMemoryProvider().is_available() is True


def test_config_schema_includes_expected_fields() -> None:
    keys = {item["key"] for item in MnemossMemoryProvider().get_config_schema()}
    assert keys == {"baseUrl", "api_key", "workspace"}


# ─── cron guard ───────────────────────────────────────────────────


def test_initialize_skips_under_cron_context() -> None:
    provider = MnemossMemoryProvider()
    provider.initialize("s", agent_context="cron", platform="cli")
    assert provider._skipped_reason is not None
    # Plugin should look dormant — no prompt block, no tools.
    assert provider.system_prompt_block() == ""
    assert provider.get_tool_schemas() == []


def test_initialize_skips_under_flush_context() -> None:
    provider = MnemossMemoryProvider()
    provider.initialize("s", agent_context="flush", platform="cli")
    assert provider._skipped_reason is not None


def test_initialize_skips_under_cron_platform() -> None:
    provider = MnemossMemoryProvider()
    provider.initialize("s", agent_context="primary", platform="cron")
    assert provider._skipped_reason is not None


# ─── per-turn lifecycle ──────────────────────────────────────────


def test_system_prompt_block_present_when_active() -> None:
    provider = _make_provider(MockBackend())
    block = provider.system_prompt_block()
    assert "Mnemoss" in block
    assert "mnemoss_recall" in block


def test_queue_prefetch_caches_recall_result_for_next_turn() -> None:
    backend = MockBackend()
    backend.canned_recall = [
        _MockResult(_MockMemory("m1", "alice prefers concise"), 2.0),
        _MockResult(
            _MockMemory("m2", "alice timezone UTC"), 1.5, source="expanded"
        ),
    ]
    provider = _make_provider(backend)

    provider.queue_prefetch("what does alice like?")

    # Recall was issued with the expected kwargs.
    assert len(backend.recall_calls) == 1
    assert backend.recall_calls[0]["query"] == "what does alice like?"
    assert backend.recall_calls[0]["k"] == 5

    # The next prefetch returns the cached block, then clears it.
    first = provider.prefetch("unused")
    assert "alice prefers concise" in first
    assert "alice timezone UTC" in first
    assert "associated" in first  # source=expanded gets the tag

    # Second prefetch without a new queue: falls back to synchronous
    # recall (cache was cleared).
    provider.prefetch("ok")
    assert len(backend.recall_calls) == 2  # fallback triggered


def test_prefetch_with_empty_recall_returns_empty_string() -> None:
    backend = MockBackend()
    backend.canned_recall = []
    provider = _make_provider(backend)

    provider.queue_prefetch("q")
    assert provider.prefetch("q") == ""


def test_sync_turn_observes_both_sides() -> None:
    backend = MockBackend()
    provider = _make_provider(backend)

    provider.sync_turn(
        "Hello, my name is Alice.",
        "Nice to meet you, Alice!",
    )
    roles = [entry["role"] for entry in backend.observed]
    assert roles == ["user", "assistant"]
    assert backend.observed[0]["session_id"] == "sess-1"
    assert backend.observed[1]["content"].startswith("Nice to meet you")


def test_sync_turn_skips_empty_sides() -> None:
    backend = MockBackend()
    provider = _make_provider(backend)

    provider.sync_turn("", "assistant only")
    assert len(backend.observed) == 1
    assert backend.observed[0]["role"] == "assistant"


# ─── tools ────────────────────────────────────────────────────────


def test_tool_schemas_returned_when_active() -> None:
    provider = _make_provider(MockBackend())
    names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert names == {"mnemoss_recall", "mnemoss_expand", "mnemoss_pin"}


def test_handle_mnemoss_recall_dispatches() -> None:
    backend = MockBackend()
    backend.canned_recall = [
        _MockResult(_MockMemory("m1", "hit"), 3.0),
    ]
    provider = _make_provider(backend)

    raw = provider.handle_tool_call(
        "mnemoss_recall", {"query": "search me", "k": 2, "include_deep": True}
    )
    payload = json.loads(raw)

    assert "results" in payload
    assert len(payload["results"]) == 1
    assert payload["results"][0]["id"] == "m1"
    assert payload["results"][0]["source"] == "direct"
    # Arguments routed through.
    assert backend.recall_calls[-1]["k"] == 2
    assert backend.recall_calls[-1]["include_deep"] is True


def test_handle_mnemoss_recall_missing_query_returns_error() -> None:
    provider = _make_provider(MockBackend())
    raw = provider.handle_tool_call("mnemoss_recall", {})
    assert "error" in json.loads(raw)


def test_handle_mnemoss_expand_dispatches() -> None:
    backend = MockBackend()
    backend.canned_expand = [
        _MockResult(_MockMemory("m2", "related"), 1.1, source="expanded"),
    ]
    provider = _make_provider(backend)

    raw = provider.handle_tool_call(
        "mnemoss_expand", {"memory_id": "m1", "hops": 2, "k": 3}
    )
    payload = json.loads(raw)

    assert payload["results"][0]["id"] == "m2"
    assert payload["results"][0]["source"] == "expanded"
    assert backend.expand_calls[-1]["hops"] == 2
    assert backend.expand_calls[-1]["k"] == 3


def test_handle_mnemoss_pin_dispatches() -> None:
    backend = MockBackend()
    provider = _make_provider(backend)

    raw = provider.handle_tool_call("mnemoss_pin", {"memory_id": "m1"})
    payload = json.loads(raw)

    assert "Pinned" in payload["result"]
    assert backend.pin_calls[-1]["memory_id"] == "m1"


def test_handle_unknown_tool_returns_error() -> None:
    provider = _make_provider(MockBackend())
    raw = provider.handle_tool_call("mnemoss_unknown", {})
    assert "Unknown tool" in json.loads(raw)["error"]


def test_handle_tool_when_skipped_returns_inactive_error() -> None:
    provider = MnemossMemoryProvider()
    provider.initialize("s", agent_context="cron", platform="cli")
    raw = provider.handle_tool_call("mnemoss_recall", {"query": "q"})
    assert "not active" in json.loads(raw)["error"]


# ─── session boundaries ──────────────────────────────────────────


def test_on_session_end_triggers_dream() -> None:
    backend = MockBackend()
    provider = _make_provider(backend)

    provider.on_session_end([])
    assert backend.dream_calls == [
        {"trigger": "session_end", "agent_id": None}
    ]


def test_on_pre_compress_returns_export_markdown() -> None:
    backend = MockBackend()
    backend.canned_export = "# Memory\n## Facts\n- user prefers SQL"
    provider = _make_provider(backend)

    md = provider.on_pre_compress([{"role": "user", "content": "..."}])
    assert "user prefers SQL" in md
    assert backend.export_calls[-1]["min_idx_priority"] == 0.7


def test_on_memory_write_mirrors_adds_only() -> None:
    backend = MockBackend()
    provider = _make_provider(backend)

    provider.on_memory_write("add", "user", "User name is Alice.")
    provider.on_memory_write("remove", "user", "User name is Alice.")
    provider.on_memory_write("add", "memory", "Project: X.")

    # Only two "add" calls landed.
    assert len(backend.observed) == 2
    roles = [entry["role"] for entry in backend.observed]
    assert roles == ["user", "assistant"]  # target=user→role=user, target=memory→role=assistant
    assert backend.observed[0]["metadata"] == {"mirror_of": "builtin.user"}


# ─── shutdown ────────────────────────────────────────────────────


def test_shutdown_is_idempotent_and_safe() -> None:
    backend = MockBackend()
    provider = _make_provider(backend)
    provider.shutdown()
    # Second call must not raise.
    provider.shutdown()


# ─── register() entry point ──────────────────────────────────────


def test_register_calls_register_memory_provider_on_ctx() -> None:
    captured: list[Any] = []

    class Ctx:
        def register_memory_provider(self, provider: Any) -> None:
            captured.append(provider)

    from mnemoss_hermes import register

    register(Ctx())
    assert len(captured) == 1
    assert isinstance(captured[0], MnemossMemoryProvider)


# Silence unused-import warnings.
_ = pytest
