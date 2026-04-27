"""Dream Diary tests (Checkpoint O)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from mnemoss import FakeEmbedder, Mnemoss, MockLLMClient, StorageParams
from mnemoss.dream.diary import (
    append_entry,
    dream_diary_path,
    render_dream_entry,
)
from mnemoss.dream.types import (
    DreamReport,
    PhaseName,
    PhaseOutcome,
    TriggerType,
)

UTC = timezone.utc


def _fake_report() -> DreamReport:
    t0 = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC)
    return DreamReport(
        trigger=TriggerType.IDLE,
        started_at=t0,
        finished_at=t0 + timedelta(seconds=2, milliseconds=300),
        agent_id=None,
        outcomes=[
            PhaseOutcome(
                phase=PhaseName.REPLAY,
                status="ok",
                details={"selected": 5, "memory_ids": ["a", "b", "c", "d", "e"]},
            ),
            PhaseOutcome(
                phase=PhaseName.CLUSTER,
                status="ok",
                details={"clusters": 1, "noise": 0, "total": 5},
            ),
            PhaseOutcome(
                phase=PhaseName.CONSOLIDATE,
                status="skipped",
                details={"reason": "no llm configured"},
            ),
        ],
    )


def test_render_dream_entry_has_expected_structure() -> None:
    md = render_dream_entry(_fake_report())
    assert md.startswith("## Dream run · 2026-04-21T12:00:00+00:00")
    assert "**Trigger:** `idle`" in md
    assert "**Agent:** (ambient)" in md
    assert "**Duration:** 2.300s" in md
    # Every phase section present.
    sections = (
        "### REPLAY · ok",
        "### CLUSTER · ok",
        "### CONSOLIDATE · skipped",
    )
    for section in sections:
        assert section in md
    # Noisy fields suppressed.
    assert "Memory(" not in md  # no dataclass dumps


def test_append_entry_creates_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "diary.md"
    append_entry(path, _fake_report())
    content1 = path.read_text()
    assert content1.startswith("## Dream run · ")

    append_entry(path, _fake_report())
    content2 = path.read_text()
    # Second entry is separated by the horizontal rule.
    assert "\n\n---\n\n" in content2
    assert content2.count("## Dream run") == 2


def test_dream_diary_path_derives_from_workspace(tmp_path: Path) -> None:
    p = dream_diary_path(tmp_path, "my_ws")
    assert p == tmp_path / "workspaces" / "my_ws" / "dreams" / "diary.md"


# ─── end-to-end: dream() writes to the diary ──────────────────────


def _mnemoss(tmp_path: Path, **kwargs) -> Mnemoss:
    return Mnemoss(
        workspace="t",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        **kwargs,
    )


async def test_dream_writes_diary_entry(tmp_path: Path) -> None:
    mock = MockLLMClient()
    mem = _mnemoss(tmp_path, llm=mock)
    try:
        await mem.observe(role="user", content="sample")
        report = await mem.dream(trigger="session_end")
        assert report.diary_path is not None
        assert report.diary_path.exists()
        content = report.diary_path.read_text()
        assert "## Dream run" in content
        assert "session_end" in content
    finally:
        await mem.close()
