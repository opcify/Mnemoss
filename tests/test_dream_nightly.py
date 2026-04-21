"""Nightly dream cycle + new triggers (Checkpoint R).

Validates that every phase dispatches correctly under the ``nightly``
trigger, and spot-checks the ``surprise`` / ``cognitive_load`` triggers'
phase lists. Uses MockLLMClient so no network traffic.
"""

from __future__ import annotations

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


def _canned(prompt: str) -> dict:
    """MockLLMClient callback — returns shape-correct responses based on
    which prompt type we're looking at."""

    if "improve or correct" in prompt:
        # P4 Refine prompt.
        return {
            "gist": "refined gist",
            "entities": ["Alice"],
            "time": None,
            "location": None,
            "participants": ["Alice"],
        }
    if "higher-level patterns that span multiple" in prompt:
        # P6 Generalize prompt.
        return {
            "patterns": [
                {"content": "Alice-related pattern", "derived_from": [1, 2]},
            ]
        }
    # Fall through: P3 Extract prompt.
    return {
        "memory_type": "fact",
        "content": "Alice fact",
        "abstraction_level": 0.6,
        "aliases": [],
    }


def _mnemoss(tmp_path: Path, llm: MockLLMClient | None = None) -> Mnemoss:
    return Mnemoss(
        workspace="nightly",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        llm=llm,
    )


async def test_nightly_runs_all_eight_phases(tmp_path: Path) -> None:
    mock = MockLLMClient(callback=_canned)
    mem = _mnemoss(tmp_path, llm=mock)
    try:
        for i in range(5):
            await mem.observe(role="user", content=f"Alice note {i}")

        report = await mem.dream(trigger="nightly")

        phases = [o.phase for o in report.outcomes]
        assert phases == [
            PhaseName.REPLAY,
            PhaseName.CLUSTER,
            PhaseName.EXTRACT,
            PhaseName.REFINE,
            PhaseName.RELATIONS,
            PhaseName.GENERALIZE,
            PhaseName.REBALANCE,
            PhaseName.DISPOSE,
        ]
        # Every phase should run (ok/skipped are both valid end-states).
        for outcome in report.outcomes:
            assert outcome.status in ("ok", "skipped")

        # Rebalance and dispose always run end-to-end (no LLM dependency).
        rebalance = report.outcome(PhaseName.REBALANCE)
        assert rebalance is not None and rebalance.status == "ok"
        assert rebalance.details["scanned"] >= 5

        dispose = report.outcome(PhaseName.DISPOSE)
        assert dispose is not None and dispose.status == "ok"
        # Fresh memories are age-protected → 0 disposals expected.
        assert dispose.details["disposed"] == 0
        assert dispose.details["protected"] >= 5
    finally:
        await mem.close()


async def test_surprise_trigger_runs_extract_and_relations(tmp_path: Path) -> None:
    mock = MockLLMClient(callback=_canned)
    mem = _mnemoss(tmp_path, llm=mock)
    try:
        await mem.observe(role="user", content="x")
        report = await mem.dream(trigger="surprise")
        phases = [o.phase for o in report.outcomes]
        assert phases == [PhaseName.EXTRACT, PhaseName.RELATIONS]
    finally:
        await mem.close()


async def test_cognitive_load_trigger_runs_refine_then_extract(tmp_path: Path) -> None:
    mock = MockLLMClient(callback=_canned)
    mem = _mnemoss(tmp_path, llm=mock)
    try:
        await mem.observe(role="user", content="x")
        report = await mem.dream(trigger="cognitive_load")
        phases = [o.phase for o in report.outcomes]
        assert phases == [PhaseName.REFINE, PhaseName.EXTRACT]
    finally:
        await mem.close()


async def test_nightly_diary_records_all_phases(tmp_path: Path) -> None:
    mock = MockLLMClient(callback=_canned)
    mem = _mnemoss(tmp_path, llm=mock)
    try:
        for i in range(5):
            await mem.observe(role="user", content=f"Alice note {i}")
        report = await mem.dream(trigger="nightly")

        assert report.diary_path is not None
        entry = report.diary_path.read_text()
        for section in (
            "REPLAY",
            "CLUSTER",
            "EXTRACT",
            "REFINE",
            "RELATIONS",
            "GENERALIZE",
            "REBALANCE",
            "DISPOSE",
        ):
            assert f"### {section}" in entry
    finally:
        await mem.close()


async def test_trigger_type_enum_has_all_five_values() -> None:
    values = {t.value for t in TriggerType}
    assert values == {
        "idle",
        "session_end",
        "surprise",
        "cognitive_load",
        "nightly",
    }


@pytest.mark.parametrize(
    "trigger",
    ["idle", "session_end", "surprise", "cognitive_load", "nightly"],
)
async def test_all_triggers_dispatch_without_error(
    tmp_path: Path, trigger: str
) -> None:
    mock = MockLLMClient(callback=_canned)
    mem = _mnemoss(tmp_path, llm=mock)
    try:
        await mem.observe(role="user", content="some memory")
        report = await mem.dream(trigger=trigger)
        assert report.trigger.value == trigger
        assert report.outcomes  # Non-empty; every trigger has at least one phase.
    finally:
        await mem.close()
