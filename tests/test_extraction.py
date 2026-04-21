"""Lazy heuristic extraction (Checkpoint L).

Unit tests cover ``extract_heuristic`` directly; integration tests
verify that recall() fires the extraction and that extraction_level=1
prevents re-extraction on subsequent recalls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from mnemoss import FakeEmbedder, Mnemoss, StorageParams
from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.encoder.extraction import ExtractionFields, extract_heuristic
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


# ─── pure heuristic unit tests ──────────────────────────────────────


def test_empty_content_returns_blank_fields() -> None:
    fields = extract_heuristic("")
    assert fields == ExtractionFields(level=1)


def test_gist_takes_first_sentence() -> None:
    content = "Alice arrived at 4:20. Then they went to dinner."
    fields = extract_heuristic(content)
    assert fields.gist == "Alice arrived at 4:20"


def test_gist_caps_at_100_chars() -> None:
    long = "a" * 150
    fields = extract_heuristic(long)
    assert fields.gist is not None
    assert len(fields.gist) == 100


def test_entities_include_latin_proper_nouns() -> None:
    fields = extract_heuristic("Alice and Bob met at the Sydney Opera House")
    assert fields.entities is not None
    assert "Alice" in fields.entities
    assert "Bob" in fields.entities
    assert "Sydney" in fields.entities
    assert "Opera" in fields.entities
    assert "House" in fields.entities


def test_entities_dedup() -> None:
    fields = extract_heuristic("Alice and Alice and Alice")
    assert fields.entities == ["Alice"]


def test_entities_skip_allcaps_and_stopwords() -> None:
    fields = extract_heuristic("USA is a country. The plan.")
    # "USA" skipped (allcaps); "The" skipped (stopword).
    assert fields.entities is None


def test_absolute_date_is_extracted() -> None:
    # A very explicit ISO date is deterministic across test runs.
    fields = extract_heuristic("meeting on 2026-04-22 at 10:00")
    assert fields.time is not None
    assert fields.time.year == 2026
    assert fields.time.month == 4
    assert fields.time.day == 22
    assert fields.time.tzinfo is not None


def test_no_date_returns_none_time() -> None:
    fields = extract_heuristic("just a plain sentence with no date")
    assert fields.time is None


def test_level_is_one_even_on_partial_fill() -> None:
    # No date, no proper noun, just a short sentence.
    fields = extract_heuristic("plain")
    assert fields.level == 1
    assert fields.time is None


def test_location_and_participants_stay_none_in_stage_3() -> None:
    fields = extract_heuristic("Alice went to Paris with Bob")
    # Stage 3 doesn't distinguish person vs place heuristically.
    assert fields.location is None
    assert fields.participants is None


# ─── store round-trip ─────────────────────────────────────────────


async def _backend(tmp_path: Path, dim: int = 4) -> SQLiteBackend:
    b = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        workspace_id="ws",
        embedding_dim=dim,
        embedder_id="fake:dim4",
    )
    await b.open()
    return b


def _memory(id: str, content: str) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=None,
        session_id="s",
        created_at=now,
        content=content,
        content_embedding=None,
        role="user",
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[now],
        index_tier=IndexTier.HOT,
    )


async def test_store_round_trips_extraction_fields(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    m = _memory("m1", "Alice arrived at 4:20")
    await b.write_memory(m, np.array([1, 0, 0, 0], dtype=np.float32))

    await b.update_extraction(
        "m1",
        gist="Alice arrived at 4:20",
        entities=["Alice"],
        time=datetime(2026, 4, 22, 16, 20, tzinfo=UTC),
        location=None,
        participants=None,
        level=1,
    )

    got = await b.get_memory("m1")
    assert got is not None
    assert got.extracted_gist == "Alice arrived at 4:20"
    assert got.extracted_entities == ["Alice"]
    assert got.extracted_time == datetime(2026, 4, 22, 16, 20, tzinfo=UTC)
    assert got.extraction_level == 1
    await b.close()


# ─── recall integration ──────────────────────────────────────────


def _mnemoss(tmp_path: Path) -> Mnemoss:
    return Mnemoss(
        workspace="test",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
    )


async def test_recall_fires_extraction_on_top_k(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mid = await mem.observe(role="user", content="Alice arrived at 2026-04-22")
        results = await mem.recall("Alice", k=3)
        hit = next((r for r in results if r.memory.id == mid), None)
        assert hit is not None
        # Heuristic fields now populated.
        assert hit.memory.extraction_level == 1
        assert hit.memory.extracted_entities is not None
        assert "Alice" in hit.memory.extracted_entities
        assert hit.memory.extracted_time is not None
        assert hit.memory.extracted_time.year == 2026
    finally:
        await mem.close()


async def test_extraction_is_idempotent_across_recalls(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        mid = await mem.observe(role="user", content="Bob said hi")
        await mem.recall("Bob", k=3)
        # Re-open and verify extraction_level stays at 1.
        assert mem._store is not None
        first = await mem._store.get_memory(mid)
        assert first is not None and first.extraction_level == 1

        await mem.recall("Bob", k=3)
        second = await mem._store.get_memory(mid)
        assert second is not None and second.extraction_level == 1
    finally:
        await mem.close()


@pytest.mark.parametrize(
    "content, expect_time",
    [
        ("See you on 2026-04-22", True),
        # Chinese date-ish phrase — dateparser handles simplified-Chinese dates.
        ("2026年4月22日下午4:20", True),
        # Completely dateless.
        ("hello world", False),
    ],
)
async def test_multilingual_time_extraction(
    tmp_path: Path, content: str, expect_time: bool
) -> None:
    fields = extract_heuristic(content)
    assert (fields.time is not None) == expect_time
