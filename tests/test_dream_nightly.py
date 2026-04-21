"""Nightly dream cycle + the five triggers.

Validates that every phase dispatches correctly under the ``nightly``
trigger, and spot-checks the ``surprise`` / ``cognitive_load`` triggers'
phase lists. Uses MockLLMClient so no network traffic.

After the P3/P4/P6 merger the pipeline is six phases —
Replay → Cluster → Consolidate → Relations → Rebalance → Dispose.
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


def _canned(_prompt: str) -> dict:
    """MockLLMClient callback — returns a fully-shaped Consolidate response.

    The merged phase asks for all three outputs in one JSON object, so
    one canned dict covers every call during a dream run.
    """

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
            },
            {
                "index": 2,
                "gist": "refined gist 2",
                "entities": ["Alice"],
                "time": None,
                "location": None,
                "participants": ["Alice"],
            },
        ],
        "patterns": [
            {"content": "Alice-related pattern", "derived_from": [1, 2]},
        ],
    }


def _mnemoss(tmp_path: Path, llm: MockLLMClient | None = None) -> Mnemoss:
    return Mnemoss(
        workspace="nightly",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        llm=llm,
    )


async def test_nightly_runs_all_six_phases(tmp_path: Path) -> None:
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
            PhaseName.CONSOLIDATE,
            PhaseName.RELATIONS,
            PhaseName.REBALANCE,
            PhaseName.DISPOSE,
        ]
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


async def test_surprise_trigger_runs_consolidate_and_relations(tmp_path: Path) -> None:
    mock = MockLLMClient(callback=_canned)
    mem = _mnemoss(tmp_path, llm=mock)
    try:
        await mem.observe(role="user", content="x")
        report = await mem.dream(trigger="surprise")
        phases = [o.phase for o in report.outcomes]
        assert phases == [PhaseName.CONSOLIDATE, PhaseName.RELATIONS]
    finally:
        await mem.close()


async def test_cognitive_load_trigger_runs_consolidate_only(tmp_path: Path) -> None:
    mock = MockLLMClient(callback=_canned)
    mem = _mnemoss(tmp_path, llm=mock)
    try:
        await mem.observe(role="user", content="x")
        report = await mem.dream(trigger="cognitive_load")
        phases = [o.phase for o in report.outcomes]
        assert phases == [PhaseName.CONSOLIDATE]
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
            "CONSOLIDATE",
            "RELATIONS",
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


async def test_phase_name_enum_has_the_six_post_merge_phases() -> None:
    values = {p.value for p in PhaseName}
    assert values == {
        "replay",
        "cluster",
        "consolidate",
        "relations",
        "rebalance",
        "dispose",
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
        assert report.outcomes  # Non-empty; every trigger has ≥1 phase.
    finally:
        await mem.close()
