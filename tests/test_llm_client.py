"""LLM client tests (Checkpoint M)."""

from __future__ import annotations

import pytest

from mnemoss import AnthropicClient, LLMClient, MockLLMClient, OpenAIClient
from mnemoss.llm.client import _extract_first_json_object


def test_mock_protocol_match() -> None:
    mock = MockLLMClient()
    # The Protocol is runtime_checkable.
    assert isinstance(mock, LLMClient)


def test_openai_is_lazy_until_called() -> None:
    """Construction must not require openai to be installed."""

    client = OpenAIClient(model="gpt-4o-mini", api_key="sk-test")
    # No outgoing call yet → no SDK import.
    assert client.model == "gpt-4o-mini"


def test_anthropic_is_lazy_until_called() -> None:
    client = AnthropicClient(model="claude-haiku-4-5-20251001", api_key="sk-test")
    assert client.model == "claude-haiku-4-5-20251001"


async def test_mock_canned_responses_consumed_in_order() -> None:
    mock = MockLLMClient(
        responses=[
            {"fact": "Alice likes coffee"},
            "free-form continuation",
        ]
    )
    first = await mock.complete_json("any prompt")
    second = await mock.complete_text("another prompt")
    assert first == {"fact": "Alice likes coffee"}
    assert second == "free-form continuation"
    assert mock.calls == [("json", "any prompt"), ("text", "another prompt")]


async def test_mock_raises_when_exhausted() -> None:
    mock = MockLLMClient(responses=[])
    with pytest.raises(RuntimeError, match="exhausted"):
        await mock.complete_text("x")


async def test_mock_callback_receives_prompt() -> None:
    captured: list[str] = []

    def callback(prompt: str) -> dict[str, str]:
        captured.append(prompt)
        return {"echoed": prompt}

    mock = MockLLMClient(callback=callback)
    result = await mock.complete_json("hello")
    assert result == {"echoed": "hello"}
    assert captured == ["hello"]


async def test_mock_type_mismatch_raises() -> None:
    mock = MockLLMClient(responses=["string for a json call"])
    with pytest.raises(TypeError, match="dict"):
        await mock.complete_json("prompt")


def test_json_extractor_handles_plain_object() -> None:
    assert _extract_first_json_object('{"a": 1}') == {"a": 1}


def test_json_extractor_strips_fences() -> None:
    text = '```json\n{"a": 1, "b": "x"}\n```'
    assert _extract_first_json_object(text) == {"a": 1, "b": "x"}


def test_json_extractor_handles_nested() -> None:
    text = 'noise before {"outer": {"inner": 42}} noise after'
    assert _extract_first_json_object(text) == {"outer": {"inner": 42}}


def test_json_extractor_survives_braces_in_strings() -> None:
    text = '{"caption": "the { char can appear }"}'
    assert _extract_first_json_object(text) == {
        "caption": "the { char can appear }"
    }


def test_json_extractor_raises_on_no_object() -> None:
    with pytest.raises(ValueError):
        _extract_first_json_object("no json here at all")
