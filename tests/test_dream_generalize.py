"""P6 Generalize tests (Checkpoint Q)."""

from __future__ import annotations

from datetime import datetime, timezone

from mnemoss import MockLLMClient
from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory, MemoryType
from mnemoss.dream.generalize import (
    DEFAULT_ABSTRACTION_LEVEL,
    build_generalize_prompt,
    generalize_facts,
)

UTC = timezone.utc


def _fact(
    id: str,
    content: str,
    *,
    agent_id: str | None = None,
    memory_type: MemoryType = MemoryType.FACT,
    salience: float = 0.2,
) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=agent_id,
        session_id=None,
        created_at=now,
        content=content,
        content_embedding=None,
        role=None,
        memory_type=memory_type,
        abstraction_level=0.6,
        access_history=[now],
        salience=salience,
    )


def test_prompt_lists_facts_with_indices_and_labels() -> None:
    facts = [
        _fact("f1", "Alice prefers lattes"),
        _fact("e1", "Alice: user's manager", memory_type=MemoryType.ENTITY),
    ]
    prompt = build_generalize_prompt(facts)
    assert "1. [fact] Alice prefers lattes" in prompt
    assert "2. [entity] Alice: user's manager" in prompt
    assert "derived_from" in prompt
    assert "1-indexed" in prompt


async def test_generalize_with_fewer_than_two_facts_returns_empty() -> None:
    llm = MockLLMClient()
    result = await generalize_facts([], llm, FormulaParams())
    assert result == []
    result = await generalize_facts(
        [_fact("f1", "only one")], llm, FormulaParams()
    )
    assert result == []
    # LLM was never invoked.
    assert llm.calls == []


async def test_generalize_builds_pattern_memories() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "patterns": [
                    {
                        "content": "User has a consistent morning coffee routine",
                        "derived_from": [1, 2],
                    }
                ]
            }
        ]
    )
    facts = [
        _fact("f1", "user ordered latte at 9am"),
        _fact("f2", "user ordered espresso at 9am"),
        _fact("f3", "unrelated fact about weather"),
    ]
    patterns = await generalize_facts(facts, llm, FormulaParams())
    assert len(patterns) == 1
    p = patterns[0]
    assert p.memory_type is MemoryType.PATTERN
    assert p.abstraction_level == DEFAULT_ABSTRACTION_LEVEL
    assert p.content == "User has a consistent morning coffee routine"
    assert p.derived_from == ["f1", "f2"]


async def test_generalize_skips_patterns_with_fewer_than_two_sources() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "patterns": [
                    {"content": "single-source pattern", "derived_from": [1]},
                    {"content": "ok pattern", "derived_from": [1, 2]},
                ]
            }
        ]
    )
    facts = [_fact("f1", "a"), _fact("f2", "b")]
    patterns = await generalize_facts(facts, llm, FormulaParams())
    assert len(patterns) == 1
    assert patterns[0].content == "ok pattern"


async def test_generalize_ignores_out_of_range_refs() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "patterns": [
                    {"content": "pattern", "derived_from": [1, 2, 999]}
                ]
            }
        ]
    )
    facts = [_fact("f1", "a"), _fact("f2", "b")]
    patterns = await generalize_facts(facts, llm, FormulaParams())
    assert len(patterns) == 1
    assert patterns[0].derived_from == ["f1", "f2"]


async def test_generalize_handles_empty_patterns_list() -> None:
    llm = MockLLMClient(responses=[{"patterns": []}])
    facts = [_fact("f1", "a"), _fact("f2", "b")]
    patterns = await generalize_facts(facts, llm, FormulaParams())
    assert patterns == []


async def test_generalize_cross_agent_promotion() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "patterns": [
                    {
                        "content": "cross-agent pattern",
                        "derived_from": [1, 2],
                    }
                ]
            }
        ]
    )
    facts = [
        _fact("f1", "alice fact", agent_id="alice"),
        _fact("f2", "bob fact", agent_id="bob"),
    ]
    patterns = await generalize_facts(facts, llm, FormulaParams())
    assert len(patterns) == 1
    assert patterns[0].agent_id is None  # ambient


async def test_generalize_preserves_single_agent_scope() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "patterns": [
                    {"content": "alice pattern", "derived_from": [1, 2]}
                ]
            }
        ]
    )
    facts = [
        _fact("f1", "a", agent_id="alice"),
        _fact("f2", "b", agent_id="alice"),
    ]
    patterns = await generalize_facts(facts, llm, FormulaParams())
    assert len(patterns) == 1
    assert patterns[0].agent_id == "alice"


async def test_generalize_returns_empty_on_malformed_response() -> None:
    # Response missing "patterns" key.
    llm = MockLLMClient(responses=[{"something_else": "nope"}])
    facts = [_fact("f1", "a"), _fact("f2", "b")]
    patterns = await generalize_facts(facts, llm, FormulaParams())
    assert patterns == []
