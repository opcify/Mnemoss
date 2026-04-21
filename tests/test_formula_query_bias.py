"""Unit tests for query bias."""

from __future__ import annotations

import pytest

from mnemoss.formula.query_bias import compute_query_bias, has_deep_cue


@pytest.mark.parametrize(
    "query, expected",
    [
        ('"4:20 PM"', 1.5),
        ('what time is "4:20"', 1.5),
        ("what time at 4:20?", 1.3),
        ("meeting on 2026-04-22", 1.3),
        ("budget was 42000", 1.3),
        ("what did we decide", 1.0),
        # Latin-script proper noun now bumps to 1.2.
        ("什么时候见 Alice?", 1.2),
        # Time beats proper noun.
        ("什么时候见 Alice 下午 4:20?", 1.3),
    ],
)
def test_query_bias_values(query: str, expected: float) -> None:
    assert compute_query_bias(query) == expected


def test_cjk_quote_chars_trigger_quote_bias() -> None:
    # Chinese corner brackets 「」 are also quotes.
    assert compute_query_bias("他说「我明天要去」") == 1.5


@pytest.mark.parametrize(
    "query",
    [
        "他说《红楼梦》",           # Chinese book-title brackets
        "参考【要点】",             # Chinese emphasis brackets
        "『こんにちは』と彼は言った",  # Japanese white corner brackets
    ],
)
def test_extended_cjk_brackets_trigger_quote_bias(query: str) -> None:
    assert compute_query_bias(query) == 1.5


def test_empty_query_defaults() -> None:
    assert compute_query_bias("") == 1.0
    assert compute_query_bias("   ") == 1.0


@pytest.mark.parametrize(
    "query, expected",
    [
        # Latin proper nouns → 1.2
        ("Alice came by yesterday", 1.2),
        ("I saw Señor García", 1.2),
        ("what did Müller say", 1.2),
        # All-lower or all-upper doesn't count.
        ("alice came by yesterday", 1.0),
        ("USA visit", 1.0),
        # Title-case stopwords alone don't trigger (just "I" is too noisy).
        ("I thought so", 1.0),
    ],
)
def test_proper_noun_detection(query: str, expected: float) -> None:
    assert compute_query_bias(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "what did we discuss long ago",
        "Years ago there was a plan",
        "originally the setup was simpler",
        "hace mucho que no lo veo",            # Spanish
        "il y a longtemps, avant la guerre",    # French
        "vor langer Zeit gab es",               # German
        "很久以前的事",                          # Chinese
        "昔々、ある村に",                         # Japanese
        "오래 전에 있었다",                        # Korean
    ],
)
def test_has_deep_cue_matches_multilingual_markers(query: str) -> None:
    assert has_deep_cue(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "what did we discuss today",
        "see you tomorrow at 4:20",
        "没什么特别的",          # Chinese: "nothing special"
        "",
    ],
)
def test_has_deep_cue_ignores_recent_queries(query: str) -> None:
    assert has_deep_cue(query) is False
