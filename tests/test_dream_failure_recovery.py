"""Partial-failure recovery tests for the Dream runner.

Verifies that:

1. A phase that raises does not crash the whole run — it's recorded
   as ``status="error"`` and downstream phases still execute.
2. ``DreamReport.degraded_mode`` flags that at least one phase
   errored; skipped phases alone do not flag it.
3. Empty upstream state produces ``status="skipped"`` with a clear
   ``skip_reason`` rather than an error.
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
from unittest.mock import patch

from mnemoss.core.config import FormulaParams
from mnemoss.dream.runner import DreamRunner
from mnemoss.dream.types import PhaseName, TriggerType
from mnemoss.encoder.embedder import FakeEmbedder
from mnemoss.llm.mock import MockLLMClient
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


async def _open_store(tmp_path: Path) -> tuple[SQLiteBackend, FakeEmbedder]:
    embedder = FakeEmbedder(dim=16)
    store = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=embedder.dim,
        embedder_id=embedder.embedder_id,
    )
    await store.open()
    return store, embedder


async def test_empty_workspace_nightly_skips_but_does_not_error(
    tmp_path: Path,
) -> None:
    """Nightly on a blank workspace should finish cleanly with
    everything either ``ok`` (replay with 0 memories) or skipped
    downstream. Nothing errored."""

    store, embedder = await _open_store(tmp_path)
    try:
        runner = DreamRunner(
            store=store,
            params=FormulaParams(),
            llm=MockLLMClient(),
            embedder=embedder,
        )
        report = await runner.run(trigger=TriggerType.NIGHTLY)
        assert not report.degraded_mode
        assert report.errors() == []
        cluster = report.outcome(PhaseName.CLUSTER)
        assert cluster is not None
        assert cluster.status == "skipped"
        assert cluster.skip_reason == "empty replay set"
    finally:
        await store.close()


async def test_phase_exception_is_caught_and_reported(tmp_path: Path) -> None:
    """If a phase raises, the runner records it as ``error`` and
    continues with downstream phases on whatever state survives."""

    store, embedder = await _open_store(tmp_path)
    try:
        runner = DreamRunner(
            store=store,
            params=FormulaParams(),
            llm=MockLLMClient(),
            embedder=embedder,
        )

        # Sabotage REPLAY with a patched selector that throws.
        import mnemoss.dream.runner as runner_mod

        async def boom(*_args, **_kwargs):
            raise RuntimeError("synthetic replay failure")

        with patch.object(runner_mod, "select_replay_candidates", side_effect=boom):
            report = await runner.run(trigger=TriggerType.NIGHTLY)

        # The run completed — no exception bubbled up.
        replay = report.outcome(PhaseName.REPLAY)
        assert replay is not None
        assert replay.status == "error"
        assert "synthetic replay failure" in (replay.error or "")

        # Downstream phases ran with empty state and either skipped
        # cleanly or returned ok with zeros.
        cluster = report.outcome(PhaseName.CLUSTER)
        assert cluster is not None
        assert cluster.status == "skipped"  # empty replay skip

        assert report.degraded_mode
        assert len(report.errors()) == 1
    finally:
        await store.close()


async def test_no_llm_configured_skips_consolidate(tmp_path: Path) -> None:
    """A nightly without an LLM client should skip CONSOLIDATE with a
    named reason, not error."""

    store, embedder = await _open_store(tmp_path)
    try:
        runner = DreamRunner(
            store=store,
            params=FormulaParams(),
            llm=None,
            embedder=embedder,
        )
        report = await runner.run(trigger=TriggerType.NIGHTLY)
        consolidate = report.outcome(PhaseName.CONSOLIDATE)
        assert consolidate is not None
        assert consolidate.status == "skipped"
        assert consolidate.skip_reason == "no llm configured"
        assert not report.degraded_mode
    finally:
        await store.close()


async def test_consolidate_skip_when_upstream_empty(tmp_path: Path) -> None:
    """If REPLAY + CLUSTER both produce nothing, CONSOLIDATE skips
    with an upstream-empty reason rather than running the LLM on
    nothing."""

    store, embedder = await _open_store(tmp_path)
    try:
        runner = DreamRunner(
            store=store,
            params=FormulaParams(),
            llm=MockLLMClient(),
            embedder=embedder,
        )

        # SURPRISE trigger skips REPLAY + CLUSTER entirely — goes
        # straight to CONSOLIDATE with empty state.
        report = await runner.run(trigger=TriggerType.SURPRISE)
        consolidate = report.outcome(PhaseName.CONSOLIDATE)
        assert consolidate is not None
        assert consolidate.status == "skipped"
        assert consolidate.skip_reason == "no replay set or clusters from upstream"
    finally:
        await store.close()


async def test_multiple_phase_errors_all_recorded(tmp_path: Path) -> None:
    """When two phases fail, both errors land in the report; the run
    still completes."""

    store, embedder = await _open_store(tmp_path)
    try:
        runner = DreamRunner(
            store=store,
            params=FormulaParams(),
            llm=MockLLMClient(),
            embedder=embedder,
        )

        import mnemoss.dream.runner as runner_mod

        async def boom_replay(*_args, **_kwargs):
            raise ValueError("replay kaboom")

        async def boom_rebalance(*_args, **_kwargs):
            raise ValueError("rebalance kaboom")

        # Patch the runner's bound references (not the source modules)
        # since ``from mnemoss.index import rebalance as _rebalance``
        # snapshots the name at import time.
        with (
            patch.object(runner_mod, "select_replay_candidates", side_effect=boom_replay),
            patch.object(runner_mod, "_rebalance", side_effect=boom_rebalance),
        ):
            report = await runner.run(trigger=TriggerType.NIGHTLY)

        errors = report.errors()
        error_phases = {o.phase for o in errors}
        assert PhaseName.REPLAY in error_phases
        assert PhaseName.REBALANCE in error_phases
        assert report.degraded_mode
    finally:
        await store.close()
