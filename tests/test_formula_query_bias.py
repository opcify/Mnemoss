"""Unit tests for query bias."""

from __future__ import annotations

import pytest

from mnemoss.formula.query_bias import compute_query_bias


@pytest.mark.parametrize(
    "query, expected",
    [
        ('"4:20 PM"', 1.5),
        ('what time is "4:20"', 1.5),
        ("what time at 4:20?", 1.3),
        ("meeting on 2026-04-22", 1.3),
        ("budget was 42000", 1.3),
        ("what did we decide", 1.0),
        ("什么时候见 Alice?", 1.0),
        ("什么时候见 Alice 下午 4:20?", 1.3),
    ],
)
def test_query_bias_values(query: str, expected: float) -> None:
    assert compute_query_bias(query) == expected


def test_cjk_quote_chars_trigger_quote_bias() -> None:
    # Chinese corner brackets 「」 are also quotes.
    assert compute_query_bias("他说「我明天要去」") == 1.5


def test_empty_query_defaults() -> None:
    assert compute_query_bias("") == 1.0
    assert compute_query_bias("   ") == 1.0
