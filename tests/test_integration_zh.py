"""Stage 1 success criterion — the canonical Chinese recall test."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemoss import FormulaParams, Mnemoss, StorageParams


@pytest.mark.integration
async def test_success_criterion_chinese(tmp_path: Path) -> None:
    mem = Mnemoss(
        workspace="test_zh",
        storage=StorageParams(root=tmp_path),
        # Silence the noise so ranking is reproducible. Formula correctness
        # is tested independently in the formula unit suite.
        formula=FormulaParams(noise_scale=0.0),
    )
    try:
        m1 = await mem.observe(role="user", content="我明天下午 4:20 和 Alice 见面")
        await mem.observe(role="user", content="见面地点在悉尼歌剧院旁边")

        results = await mem.recall("什么时候见 Alice?", k=3)
        assert results, "recall returned nothing"
        assert results[0].memory.id == m1, (
            f"expected m1 ('4:20 和 Alice') first; got: {[r.memory.content for r in results]}"
        )
    finally:
        await mem.close()
