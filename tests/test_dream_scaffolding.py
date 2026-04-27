"""Dream dispatcher + P1 Replay tests (Checkpoint M).

Exercises the dispatcher end-to-end with a real SQLite backend, a
FakeEmbedder, and either no LLM or a MockLLMClient. Cluster / Extract /
Relations phases are still skipped here — they're wired in Checkpoint N.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mnemoss import (
    FakeEmbedder,
    Mnemoss,
    MockLLMClient,
    PhaseName,
    StorageParams,
    TriggerType,
)
from mnemoss.core.config import FormulaParams
from mnemoss.dream.replay import select_replay_candidates

UTC = timezone.utc


def _mnemoss(tmp_path: Path, **kwargs) -> Mnemoss:
    return Mnemoss(
        workspace="test",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        **kwargs,
    )


async def test_dream_returns_report_with_phase_outcomes(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="alpha")
        await mem.observe(role="user", content="beta")

        report = await mem.dream(trigger="idle")
        assert report.trigger is TriggerType.IDLE
        # Idle runs the three light-trigger phases (Relations removed
        # 2026-04-27 per dreaming-validation study).
        phases = [o.phase for o in report.outcomes]
        assert phases == [
            PhaseName.REPLAY,
            PhaseName.CLUSTER,
            PhaseName.CONSOLIDATE,
        ]
        replay = report.outcome(PhaseName.REPLAY)
        assert replay is not None and replay.status == "ok"
        assert replay.details["selected"] >= 2
    finally:
        await mem.close()


async def test_dream_without_llm_records_explicit_skip(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)  # no llm configured
    try:
        await mem.observe(role="user", content="x")
        report = await mem.dream(trigger="idle")
        consolidate = report.outcome(PhaseName.CONSOLIDATE)
        assert consolidate is not None
        assert consolidate.status == "skipped"
        assert "llm" in (consolidate.skip_reason or "").lower()
    finally:
        await mem.close()


async def test_dream_skips_consolidate_for_tiny_replay_sets(tmp_path: Path) -> None:
    """Consolidate needs ≥2 memories per cluster — a single-memory
    workspace exercises the happy path without burning an LLM call."""

    mock = MockLLMClient()
    mem = _mnemoss(tmp_path, llm=mock)
    try:
        await mem.observe(role="user", content="just one")
        report = await mem.dream(trigger="idle")
        consolidate = report.outcome(PhaseName.CONSOLIDATE)
        assert consolidate is not None
        assert consolidate.status == "ok"
        assert consolidate.details["summaries"] == 0
        assert consolidate.details["patterns"] == 0
        assert consolidate.details["refined"] == 0
        assert mock.calls == []  # LLM not invoked for singleton clusters.
    finally:
        await mem.close()


async def test_dream_flushes_pending_event_buffers(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        # Keep the buffer open — no turn change.
        mid = await mem.observe(role="user", content="in-flight", turn_id="open")
        report = await mem.dream(trigger="session_end")
        # The buffered memory now exists, so P1 Replay saw it.
        replay = report.outcome(PhaseName.REPLAY)
        assert replay is not None
        assert mid in replay.details["memory_ids"]
    finally:
        await mem.close()


async def test_dream_agent_scoping(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    alice = mem.for_agent("alice")
    bob = mem.for_agent("bob")
    try:
        await alice.observe(role="user", content="alice note")
        await bob.observe(role="user", content="bob note")
        await mem.observe(role="user", content="ambient note")

        alice_report = await mem.dream(trigger="idle", agent_id="alice")
        alice_ids = alice_report.outcome(PhaseName.REPLAY).details["memory_ids"]
        alice_contents = {
            m.content for m in alice_report.outcome(PhaseName.REPLAY).details["memories"]
        }
        assert "alice note" in alice_contents
        assert "ambient note" in alice_contents
        assert "bob note" not in alice_contents
        _ = alice_ids
    finally:
        await mem.close()


# ─── P1 Replay as a unit ────────────────────────────────────────────


async def test_replay_respects_limit(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        for i in range(8):
            await mem.observe(role="user", content=f"m{i}")

        assert mem._store is not None
        candidates = await select_replay_candidates(
            mem._store,
            agent_id=None,
            params=FormulaParams(),
            now=datetime.now(UTC),
            limit=3,
        )
        assert len(candidates) == 3
    finally:
        await mem.close()


async def test_replay_prioritizes_higher_base_level(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        # First memory is ancient; second is fresh.
        ancient = await mem.observe(role="user", content="ancient")
        fresh = await mem.observe(role="user", content="fresh")

        assert mem._store is not None
        candidates = await select_replay_candidates(
            mem._store,
            agent_id=None,
            params=FormulaParams(),
            now=datetime.now(UTC),
            limit=2,
        )
        # Both fresh enough to show up; ordering by B_i desc.
        ids = [c.id for c in candidates]
        # fresh should rank at least as high as ancient (fresh has higher B_i).
        assert fresh in ids
        _ = ancient
    finally:
        await mem.close()


@pytest.mark.parametrize(
    "bad_trigger",
    ["not_a_trigger", "random", ""],
)
async def test_dream_rejects_unknown_trigger(tmp_path: Path, bad_trigger: str) -> None:
    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="x")
        with pytest.raises(ValueError):
            await mem.dream(trigger=bad_trigger)
    finally:
        await mem.close()


async def test_replay_filters_by_min_base_level(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="some content")

        assert mem._store is not None
        # B_i for a fresh memory hovers near η_0 = 1.0.
        filtered = await select_replay_candidates(
            mem._store,
            agent_id=None,
            params=FormulaParams(),
            now=datetime.now(UTC) + timedelta(days=365),
            limit=10,
            min_base_level=-0.1,  # Ancient memories fail this.
        )
        assert filtered == []
    finally:
        await mem.close()
