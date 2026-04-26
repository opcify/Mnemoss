"""Tests for the ``phases=`` ablation mask on ``DreamRunner.run``.

The mask filters the iteration loop. Phases excluded by the mask record
``status="excluded_by_mask"``. Phases that run with empty input continue
to record their existing skip reasons (``"empty replay set"``, etc.) —
the mask layer does NOT enforce dependency rules; downstream phases
degrade naturally on empty state.

These tests use ``MockLLMClient`` and ``FakeEmbedder`` so they never
hit the network and run fast in the default suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemoss import (
    DreamerParams,
    FakeEmbedder,
    Mnemoss,
    MockLLMClient,
    StorageParams,
)


def _canned_response(_prompt: str) -> dict:
    return {
        "summary": {
            "memory_type": "fact",
            "content": "shared fact",
            "abstraction_level": 0.7,
            "aliases": [],
        },
        "refinements": [],
        "patterns": [],
    }


async def _seed(mem: Mnemoss, n: int = 4) -> None:
    for i in range(n):
        await mem.observe(role="user", content=f"Alice note {i} about the project")


# ─── full pipeline (regression baseline) ───────────────────────────


async def test_phases_none_runs_full_trigger_pipeline(tmp_path: Path) -> None:
    """``phases=None`` is the historical default: every phase the trigger
    normally runs runs."""

    mem = Mnemoss(
        workspace="ablate_baseline",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        llm=MockLLMClient(callback=_canned_response),
    )
    try:
        await _seed(mem)
        report = await mem.dream(trigger="idle")
        statuses = {o.phase.value: o.status for o in report.outcomes}
        # idle trigger = replay + cluster + consolidate + relations
        assert set(statuses.keys()) == {"replay", "cluster", "consolidate", "relations"}
        # None of them are excluded_by_mask in the baseline run.
        assert "excluded_by_mask" not in statuses.values()
    finally:
        await mem.close()


# ─── single-phase masks ────────────────────────────────────────────


@pytest.mark.parametrize(
    "kept_phase",
    ["replay", "cluster", "consolidate", "relations"],
)
async def test_single_phase_mask_excludes_others(tmp_path: Path, kept_phase: str) -> None:
    """``phases={X}`` runs X (or skips it on empty input) and marks every
    other phase ``excluded_by_mask``."""

    mem = Mnemoss(
        workspace=f"ablate_{kept_phase}",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        llm=MockLLMClient(callback=_canned_response),
    )
    try:
        await _seed(mem)
        report = await mem.dream(trigger="idle", phases={kept_phase})

        for outcome in report.outcomes:
            if outcome.phase.value == kept_phase:
                # Kept phase ran (either ok or naturally skipped on
                # empty upstream — never excluded_by_mask).
                assert outcome.status != "excluded_by_mask", (
                    f"{kept_phase} was masked OUT — it should have been kept"
                )
            else:
                assert outcome.status == "excluded_by_mask"
                assert outcome.skip_reason == "excluded by phases mask"
    finally:
        await mem.close()


# ─── empty mask ────────────────────────────────────────────────────


async def test_empty_phase_mask_excludes_everything(tmp_path: Path) -> None:
    """``phases=set()`` is a legal no-op: every phase records
    ``excluded_by_mask``, no LLM calls happen, no embeddings get
    computed."""

    mem = Mnemoss(
        workspace="ablate_empty",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        llm=MockLLMClient(callback=_canned_response),
    )
    try:
        await _seed(mem)
        report = await mem.dream(trigger="idle", phases=set())

        for outcome in report.outcomes:
            assert outcome.status == "excluded_by_mask"
        assert not report.degraded_mode  # Excluded != errored.
    finally:
        await mem.close()


# ─── downstream phases skip naturally on empty state ──────────────


async def test_consolidate_only_mask_skips_naturally_with_no_clusters(
    tmp_path: Path,
) -> None:
    """``phases={"consolidate"}`` runs Consolidate but its upstream
    (Cluster) was excluded, so ``state.cluster_assignments`` and
    ``state.replay_set`` are both empty. Consolidate must record its
    EXISTING ``status="skipped"`` skip reason — NOT
    ``excluded_by_mask`` and NOT a new ``cluster_excluded`` reason.

    This is the load-bearing test for the simplified phase-mask design:
    we don't enforce dependencies in the mask layer; downstream phases
    handle empty input via their existing guards.
    """

    mem = Mnemoss(
        workspace="ablate_consolidate_only",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        llm=MockLLMClient(callback=_canned_response),
    )
    try:
        await _seed(mem)
        report = await mem.dream(trigger="idle", phases={"consolidate"})

        consolidate = next(o for o in report.outcomes if o.phase.value == "consolidate")
        # Consolidate runs but skips because upstream state is empty.
        assert consolidate.status == "skipped"
        assert consolidate.skip_reason is not None
        # Replay + cluster + relations are excluded by the mask.
        for phase_value in ("replay", "cluster", "relations"):
            outcome = next(o for o in report.outcomes if o.phase.value == phase_value)
            assert outcome.status == "excluded_by_mask"
    finally:
        await mem.close()


# ─── DreamerParams plumbing ────────────────────────────────────────


async def test_dreamer_params_reach_runner(tmp_path: Path) -> None:
    """``Mnemoss(dreamer=DreamerParams(cluster_min_size=5))`` must
    actually plumb through to the ``DreamRunner`` so the harness can
    pin config without monkey-patching."""

    mem = Mnemoss(
        workspace="ablate_dreamer_plumbing",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        llm=MockLLMClient(callback=_canned_response),
        dreamer=DreamerParams(
            cluster_min_size=5,
            replay_limit=42,
            replay_min_base_level=-1.5,
        ),
    )
    try:
        # The plumbing is verified at the config level — the runner is
        # constructed inside dream() and isn't directly inspectable
        # without hooks. Verify the config carries the values.
        assert mem._config.dreamer.cluster_min_size == 5
        assert mem._config.dreamer.replay_limit == 42
        assert mem._config.dreamer.replay_min_base_level == -1.5
        # End-to-end: dream() doesn't crash with non-default values.
        await _seed(mem)
        report = await mem.dream(trigger="idle")
        # min_size=5 with 4 seeded memories means cluster will likely
        # produce no clusters — that's fine, the test is plumbing not
        # cluster-quality.
        assert any(o.phase.value == "cluster" for o in report.outcomes)
    finally:
        await mem.close()
