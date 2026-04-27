"""Dream integration — full cycle with MockLLMClient.

Does not hit the network. Exercises the whole path: observe →
flush → replay → cluster → consolidate → relations → diary → export.
"""

from __future__ import annotations

from pathlib import Path

from mnemoss import FakeEmbedder, Mnemoss, MockLLMClient, StorageParams


async def test_full_dream_cycle_end_to_end(tmp_path: Path) -> None:
    def canned(_prompt: str) -> dict:
        return {
            "summary": {
                "memory_type": "fact",
                "content": "shared fact about Alice",
                "abstraction_level": 0.7,
                "aliases": ["A"],
            },
            "refinements": [],
            "patterns": [],
        }

    mock = MockLLMClient(callback=canned)
    mem = Mnemoss(
        workspace="integration",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        llm=mock,
    )
    try:
        for i in range(4):
            await mem.observe(role="user", content=f"Alice note {i}")

        report = await mem.dream(trigger="idle")

        phase_statuses = {o.phase.value: o.status for o in report.outcomes}
        # Relations removed 2026-04-27 per dreaming-validation study.
        assert phase_statuses == {
            "replay": "ok",
            "cluster": "ok",
            "consolidate": "ok",
        }

        # Diary entry written.
        assert report.diary_path is not None
        assert report.diary_path.exists()
        diary_text = report.diary_path.read_text()
        assert "Dream run" in diary_text
        assert "REPLAY · ok" in diary_text

        # memory.md includes the original observations; if the mock LLM
        # produced a consolidated fact, it should be in the facts section.
        md = await mem.export_markdown(min_idx_priority=0.5)
        assert "Memory — ambient" in md
    finally:
        await mem.close()
