"""English-language integration test with the real multilingual embedder.

Sanity-checks that the pipeline works on ASCII/Latin text too — a
prerequisite for multilingual claims. This is the "easy" case that
should pass a fortiori if the non-Latin tests pass, but we run it to
catch any regression where English accidentally stops working because
of over-indexing on trigram/CJK behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemoss import FormulaParams, Mnemoss, StorageParams


@pytest.mark.integration
async def test_english_recall(tmp_path: Path) -> None:
    mem = Mnemoss(
        workspace="test_en",
        storage=StorageParams(root=tmp_path),
        formula=FormulaParams(noise_scale=0.0),
    )
    try:
        m_time = await mem.observe(role="user", content="Meeting with Alice tomorrow at 4:20 PM")
        await mem.observe(
            role="user",
            content="The meeting place is next to the Sydney Opera House",
        )

        results = await mem.recall("When is my meeting with Alice?", k=3)
        assert results, "recall returned nothing"
        assert results[0].memory.id == m_time, (
            f"expected the 4:20 memory first; got: {[r.memory.content for r in results]}"
        )
    finally:
        await mem.close()


@pytest.mark.integration
async def test_english_semantic_bridges_paraphrase(tmp_path: Path) -> None:
    """Semantic similarity should bridge a paraphrase without shared tokens."""

    mem = Mnemoss(
        workspace="test_en_semantic",
        storage=StorageParams(root=tmp_path),
        formula=FormulaParams(noise_scale=0.0),
    )
    try:
        m_target = await mem.observe(
            role="user",
            content="I need to buy groceries after work on Friday evening",
        )
        await mem.observe(
            role="user",
            content="The meeting with the architecture review board was canceled",
        )

        # No token overlap with the target; semantic cosine must carry it.
        results = await mem.recall("shopping list errand", k=3)
        assert results
        top_ids = [r.memory.id for r in results]
        assert m_target in top_ids
    finally:
        await mem.close()
