"""Tests for the T2.3 observability extensions.

Covers:

- ``Mnemoss(cost_limits=...)`` plumbs the governor into ``dream()``.
- ``status()`` gains ``llm_cost`` (today / month / total / limits) and
  ``dreams`` (recent run summaries, degraded count) blocks.
- ``ActivationBreakdown.to_dict()`` produces a JSON-safe view.
"""

from __future__ import annotations

import json
from pathlib import Path

from mnemoss import (
    CostLimits,
    FakeEmbedder,
    FormulaParams,
    Mnemoss,
    MockLLMClient,
    StorageParams,
)
from mnemoss.formula.activation import ActivationBreakdown


def _make_mem(
    tmp_path: Path,
    *,
    llm: MockLLMClient | None = None,
    cost_limits: CostLimits | None = None,
) -> Mnemoss:
    return Mnemoss(
        workspace="obs",
        embedding_model=FakeEmbedder(dim=16),
        llm=llm,
        cost_limits=cost_limits,
        formula=FormulaParams(noise_scale=0.0),
        storage=StorageParams(root=tmp_path),
    )


def _consolidate_response(n: int) -> dict:
    return {
        "summary": {
            "memory_type": "fact",
            "content": "cluster summary",
            "abstraction_level": 0.6,
        },
        "refinements": [
            {"index": i + 1, "gist": f"g-{i}", "time": None}
            for i in range(n)
        ],
        "patterns": [],
    }


# ─── status() extensions ─────────────────────────────────────────


async def test_status_includes_llm_cost_block_on_fresh_workspace(
    tmp_path: Path,
) -> None:
    """The cost block is present even on a fresh workspace that's
    never dreamed. All counts start at zero; limits surface whatever
    the caller configured."""

    limits = CostLimits(max_llm_calls_per_day=42, max_llm_calls_per_run=5)
    mem = _make_mem(tmp_path, cost_limits=limits)
    try:
        await mem.observe(role="user", content="hello")
        status = await mem.status()

        assert "llm_cost" in status
        cost = status["llm_cost"]
        assert cost["today_calls"] == 0
        assert cost["month_calls"] == 0
        assert cost["total_calls"] == 0
        assert cost["limits"]["max_llm_calls_per_day"] == 42
        assert cost["limits"]["max_llm_calls_per_run"] == 5
        assert cost["limits"]["max_llm_calls_per_month"] is None
    finally:
        await mem.close()


async def test_status_includes_dreams_block(tmp_path: Path) -> None:
    """After a dream run, ``status().dreams`` shows a summary entry
    with the trigger, phase statuses, and degraded flag."""

    llm = MockLLMClient(responses=[_consolidate_response(2) for _ in range(5)])
    mem = _make_mem(tmp_path, llm=llm)
    try:
        for i in range(3):
            await mem.observe(role="user", content=f"note {i} on the topic")
        await mem.dream(trigger="idle")

        status = await mem.status()
        dreams = status["dreams"]
        assert dreams["recent_count"] == 1
        assert dreams["recent_degraded_count"] == 0
        entry = dreams["recent"][0]
        assert entry["trigger"] == "idle"
        assert entry["degraded"] is False
        assert "phase_statuses" in entry
        # Duration is a positive number, not a string.
        assert isinstance(entry["duration_seconds"], float)
        assert entry["duration_seconds"] >= 0
    finally:
        await mem.close()


async def test_status_dreams_is_bounded(tmp_path: Path) -> None:
    """Recent-dream history caps at the in-memory bound — a 24/7
    scheduler shouldn't accumulate unlimited entries in ``status()``."""

    llm = MockLLMClient(responses=[_consolidate_response(1) for _ in range(100)])
    mem = _make_mem(tmp_path, llm=llm)
    try:
        await mem.observe(role="user", content="seed")
        # Force more runs than the cap.
        for _ in range(15):
            await mem.dream(trigger="idle")

        status = await mem.status()
        assert status["dreams"]["recent_count"] <= 10
    finally:
        await mem.close()


async def test_cost_ledger_persists_across_reopen(tmp_path: Path) -> None:
    """The LLM call count survives a close/reopen because it lives in
    ``workspace_meta`` on disk, not in process memory.

    We drive the ledger directly rather than going through
    ``dream()`` because the clustering phase isn't guaranteed to
    produce any clusters on tiny workspaces — this test is about the
    persistence mechanism, not the dream pipeline.
    """

    mem = _make_mem(tmp_path)
    try:
        await mem.observe(role="user", content="seed so the store is real")
        # Simulate three past LLM calls by calling the ledger directly.
        assert mem._cost_ledger is not None
        mem._cost_ledger.record_call()
        mem._cost_ledger.record_call()
        mem._cost_ledger.record_call()

        status = await mem.status()
        assert status["llm_cost"]["total_calls"] == 3
        assert status["llm_cost"]["today_calls"] == 3
    finally:
        await mem.close()

    # Fresh instance, same workspace directory — ledger should re-read.
    mem2 = _make_mem(tmp_path)
    try:
        status = await mem2.status()
        assert status["llm_cost"]["total_calls"] == 3
        assert status["llm_cost"]["today_calls"] == 3
    finally:
        await mem2.close()


async def test_status_reports_degraded_dream(tmp_path: Path) -> None:
    """A dream with a crashed phase shows up as ``degraded`` in the
    dreams summary, even though the run as a whole completed."""

    from unittest.mock import patch

    import mnemoss.dream.runner as runner_mod

    llm = MockLLMClient(responses=[_consolidate_response(1) for _ in range(3)])
    mem = _make_mem(tmp_path, llm=llm)
    try:
        await mem.observe(role="user", content="anything")

        async def boom(*_a, **_kw):
            raise RuntimeError("injected")

        with patch.object(
            runner_mod, "select_replay_candidates", side_effect=boom
        ):
            await mem.dream(trigger="nightly")

        status = await mem.status()
        dreams = status["dreams"]
        assert dreams["recent_count"] == 1
        assert dreams["recent_degraded_count"] == 1
        entry = dreams["recent"][0]
        assert entry["degraded"] is True
        assert any("injected" in e["error"] for e in entry["errors"])
    finally:
        await mem.close()


async def test_status_is_json_serializable(tmp_path: Path) -> None:
    """The whole status payload must round-trip through ``json.dumps``
    — we ship it over REST and Prometheus-ingest it elsewhere."""

    llm = MockLLMClient(responses=[_consolidate_response(1)])
    mem = _make_mem(tmp_path, llm=llm)
    try:
        await mem.observe(role="user", content="status-json")
        await mem.dream(trigger="idle")
        status = await mem.status()
        # This will raise if anything non-JSON (datetime, dataclass,
        # set) leaked into the payload.
        encoded = json.dumps(status)
        assert "llm_cost" in encoded
        assert "dreams" in encoded
    finally:
        await mem.close()


# ─── ActivationBreakdown.to_dict() ────────────────────────────────


def test_activation_breakdown_to_dict_round_trip() -> None:
    """``to_dict`` is the canonical JSON-safe view of the scoring
    breakdown. The numbers must survive JSON encoding intact."""

    b = ActivationBreakdown(
        base_level=-0.5,
        spreading=0.1,
        matching=0.3,
        noise=0.02,
        total=-0.08,
        idx_priority=0.6,
        w_f=0.7,
        w_s=0.3,
        query_bias=1.3,
    )
    d = b.to_dict()
    # Every field present.
    assert set(d.keys()) == {
        "base_level", "spreading", "matching", "noise", "total",
        "idx_priority", "w_f", "w_s", "query_bias",
    }
    # JSON round-trip.
    decoded = json.loads(json.dumps(d))
    for k, v in d.items():
        assert decoded[k] == v


async def test_explain_recall_output_is_exportable(tmp_path: Path) -> None:
    """End-to-end: ``explain_recall`` returns an ``ActivationBreakdown``
    whose ``to_dict`` is serializable — so REST / MCP / logs can ship
    it without custom encoders."""

    mem = _make_mem(tmp_path)
    try:
        mid = await mem.observe(role="user", content="explainable content")
        breakdown = await mem.explain_recall("explainable", mid)
        assert breakdown is not None
        payload = json.dumps(breakdown.to_dict())
        assert "base_level" in payload
    finally:
        await mem.close()


# ─── cost_limits plumbing ──────────────────────────────────────────


async def test_constructor_cost_limits_actually_gates_dream(
    tmp_path: Path,
) -> None:
    """Passing ``cost_limits`` to the ``Mnemoss`` constructor should
    flow through to ``dream()`` — not just sit inert in status()."""

    # Cap at zero calls — no LLM work should happen.
    limits = CostLimits(max_llm_calls_per_run=0)
    llm = MockLLMClient(responses=[_consolidate_response(2) for _ in range(5)])
    mem = _make_mem(tmp_path, llm=llm, cost_limits=limits)
    try:
        for i in range(4):
            await mem.observe(role="user", content=f"clusterable note {i}")

        await mem.dream(trigger="nightly")

        # The cap of zero means MockLLMClient should never be called.
        assert llm.calls == []

        # Cost ledger also stays at zero.
        status = await mem.status()
        assert status["llm_cost"]["today_calls"] == 0
        assert status["llm_cost"]["total_calls"] == 0
    finally:
        await mem.close()
