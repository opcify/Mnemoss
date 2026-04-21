"""P4 Refine tests (Checkpoint P) — MockLLMClient, no network."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mnemoss import FakeEmbedder, Mnemoss, MockLLMClient, StorageParams
from mnemoss.core.types import Memory, MemoryType
from mnemoss.dream.refine import build_refine_prompt, refine_memory_fields

UTC = timezone.utc


def _mem(
    content: str,
    *,
    level: int = 1,
    gist: str | None = None,
    entities: list[str] | None = None,
) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id="m1",
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
        extracted_gist=gist,
        extracted_entities=entities,
        extraction_level=level,
    )


def test_prompt_includes_content_and_existing_fields() -> None:
    memory = _mem(
        "Alice met Bob at noon",
        gist="Alice met Bob",
        entities=["Alice", "Bob"],
    )
    prompt = build_refine_prompt(memory)
    assert "Alice met Bob at noon" in prompt
    # Existing heuristic fields shown verbatim.
    assert "Alice met Bob" in prompt  # gist
    assert '"Alice"' in prompt
    assert '"Bob"' in prompt
    # Schema spelled out.
    assert "gist" in prompt
    assert "entities" in prompt
    assert "time" in prompt
    assert "location" in prompt
    assert "participants" in prompt


async def test_refine_upgrades_fields_to_level_2() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "gist": "Alice and Bob met at the office at noon",
                "entities": ["Alice", "Bob"],
                "time": "2026-04-22T12:00:00+00:00",
                "location": "office",
                "participants": ["Alice", "Bob"],
            }
        ]
    )
    memory = _mem("Alice met Bob at noon in the office")
    fields = await refine_memory_fields(memory, llm)
    assert fields is not None
    assert fields.level == 2
    assert fields.gist == "Alice and Bob met at the office at noon"
    assert fields.entities == ["Alice", "Bob"]
    assert fields.time is not None
    assert fields.time.year == 2026
    assert fields.time.tzinfo is not None
    assert fields.location == "office"
    assert fields.participants == ["Alice", "Bob"]


async def test_refine_parses_loose_time_value() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "gist": "x",
                "entities": [],
                "time": "2026-04-22",  # date-only, no time/tz
                "location": None,
                "participants": [],
            }
        ]
    )
    memory = _mem("x")
    fields = await refine_memory_fields(memory, llm)
    assert fields is not None
    assert fields.time is not None
    # Missing tzinfo should be treated as UTC.
    assert fields.time.tzinfo is not None


async def test_refine_handles_null_time() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "gist": "no time here",
                "entities": [],
                "time": None,
                "location": None,
                "participants": [],
            }
        ]
    )
    memory = _mem("x")
    fields = await refine_memory_fields(memory, llm)
    assert fields is not None
    assert fields.time is None


async def test_refine_normalizes_empty_lists_to_none() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "gist": "x",
                "entities": [],
                "time": None,
                "location": None,
                "participants": [],
            }
        ]
    )
    memory = _mem("x")
    fields = await refine_memory_fields(memory, llm)
    assert fields is not None
    assert fields.entities is None
    assert fields.participants is None


async def test_refine_returns_none_on_empty_content() -> None:
    llm = MockLLMClient(responses=[{"gist": "should never reach here"}])
    memory = _mem("   ")
    fields = await refine_memory_fields(memory, llm)
    assert fields is None
    assert llm.calls == []  # Empty content short-circuits before the LLM call.


async def test_refine_returns_none_on_malformed_time() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "gist": "x",
                "entities": [],
                "time": "not-a-date-at-all",
                "location": None,
                "participants": [],
            }
        ]
    )
    memory = _mem("x")
    fields = await refine_memory_fields(memory, llm)
    assert fields is not None
    # Malformed time → fields.time is None, not an exception.
    assert fields.time is None
    # Other fields still refined.
    assert fields.gist == "x"


# ─── end-to-end via session_end dream ────────────────────────────


def _mnemoss(tmp_path: Path, **kwargs) -> Mnemoss:
    return Mnemoss(
        workspace="t",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        **kwargs,
    )


async def test_session_end_dream_refines_level_1_memories(tmp_path: Path) -> None:
    def canned(prompt: str) -> dict:
        # Two different prompt shapes arrive: extract (contains "Distill")
        # vs refine (contains "improve or correct").
        if "improve or correct" in prompt:
            return {
                "gist": "refined gist",
                "entities": ["Alice"],
                "time": "2026-04-22T10:00:00+00:00",
                "location": None,
                "participants": ["Alice"],
            }
        return {
            "memory_type": "fact",
            "content": "extracted fact",
            "abstraction_level": 0.6,
            "aliases": [],
        }

    mock = MockLLMClient(callback=canned)
    mem = _mnemoss(tmp_path, llm=mock)
    try:
        # Observe + force a recall to trigger heuristic extraction → level=1.
        for i in range(3):
            await mem.observe(role="user", content=f"Alice note {i}")
        await mem.recall("Alice", k=3)

        # All three memories are now at extraction_level=1.
        assert mem._store is not None
        pre = await mem._store.iter_memory_ids()
        levels_before = {
            m.id: m.extraction_level
            for m in await mem._store.materialize_memories(pre)
        }
        assert all(lvl == 1 for lvl in levels_before.values())

        # session_end dream runs refine.
        report = await mem.dream(trigger="session_end")
        refine = next(o for o in report.outcomes if o.phase.value == "refine")
        assert refine.status == "ok"
        assert refine.details["refined"] >= 1

        levels_after = {
            m.id: m.extraction_level
            for m in await mem._store.materialize_memories(pre)
        }
        assert any(lvl == 2 for lvl in levels_after.values())
    finally:
        await mem.close()


async def test_session_end_refine_respects_batch_limit(tmp_path: Path) -> None:
    def canned(prompt: str) -> dict:
        if "improve or correct" in prompt:
            return {
                "gist": "r",
                "entities": [],
                "time": None,
                "location": None,
                "participants": [],
            }
        return {
            "memory_type": "fact",
            "content": "extracted",
            "abstraction_level": 0.6,
            "aliases": [],
        }

    mock = MockLLMClient(callback=canned)
    # Construct with a tight batch limit via segmenter config.
    from mnemoss.dream.runner import DreamRunner
    from mnemoss.dream.types import TriggerType

    mem = _mnemoss(tmp_path, llm=mock)
    try:
        for i in range(6):
            await mem.observe(role="user", content=f"note {i}")

        assert mem._store is not None
        runner = DreamRunner(
            mem._store,
            mem._config.formula,
            llm=mock,
            embedder=mem._embedder,
            refine_batch_size=2,
        )
        report = await runner.run(TriggerType.SESSION_END)
        refine = next(o for o in report.outcomes if o.phase.value == "refine")
        # Refined is capped at 2 even though 6 memories are eligible.
        assert refine.details["refined"] <= 2
        assert refine.details["batch_limit"] == 2
    finally:
        await mem.close()


async def test_session_end_refine_skipped_without_llm(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)
    try:
        await mem.observe(role="user", content="note")
        report = await mem.dream(trigger="session_end")
        refine = next(o for o in report.outcomes if o.phase.value == "refine")
        assert refine.status == "skipped"
        assert "llm" in refine.details.get("reason", "").lower()
    finally:
        await mem.close()
