"""Multilingual stress tests: non-Latin-script scripts beyond Chinese."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemoss import FormulaParams, Mnemoss, StorageParams


@pytest.mark.integration
async def test_japanese_recall(tmp_path: Path) -> None:
    mem = Mnemoss(
        workspace="test_ja",
        storage=StorageParams(root=tmp_path),
        formula=FormulaParams(noise_scale=0.0),
    )
    try:
        m_time = await mem.observe(
            role="user", content="明日の午後4時20分にアリスと会議"
        )
        await mem.observe(role="user", content="会議の場所はシドニーのオペラハウスの近く")

        results = await mem.recall("いつアリスに会う？", k=3)
        assert results, "recall returned nothing"
        assert results[0].memory.id == m_time
    finally:
        await mem.close()


@pytest.mark.integration
async def test_arabic_recall(tmp_path: Path) -> None:
    mem = Mnemoss(
        workspace="test_ar",
        storage=StorageParams(root=tmp_path),
        formula=FormulaParams(noise_scale=0.0),
    )
    try:
        m_time = await mem.observe(
            role="user", content="أقابل أليس غداً الساعة 4:20 مساءً"
        )
        await mem.observe(
            role="user", content="مكان الاجتماع بجوار دار أوبرا سيدني"
        )

        results = await mem.recall("متى ألتقي مع أليس؟", k=3)
        assert results, "recall returned nothing"
        assert results[0].memory.id == m_time
    finally:
        await mem.close()


@pytest.mark.integration
async def test_cross_language_semantic_bridge(tmp_path: Path) -> None:
    """Multilingual embedder bridges languages: English query over Chinese content."""

    mem = Mnemoss(
        workspace="test_xlang",
        storage=StorageParams(root=tmp_path),
        formula=FormulaParams(noise_scale=0.0),
    )
    try:
        m_time = await mem.observe(
            role="user", content="我明天下午 4:20 和 Alice 见面"
        )
        await mem.observe(role="user", content="见面地点在悉尼歌剧院旁边")

        results = await mem.recall("when is the meeting with Alice?", k=3)
        assert results, "recall returned nothing"
        # Semantic cosine should carry the English query to the Chinese memory.
        top_ids = [r.memory.id for r in results]
        assert m_time in top_ids
    finally:
        await mem.close()
