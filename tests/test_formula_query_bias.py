"""Unit tests for b_F query-bias rules.

All rules are language-neutral / structural. Non-Latin scripts trigger
every cue that uses language-agnostic regex (quotes, URLs, digits,
hashtags, code identifiers). The acronym rule is the only Latin-only
rule and is silent on other scripts by construction.
"""

from __future__ import annotations

import pytest

from mnemoss.formula.query_bias import compute_query_bias, has_deep_cue

# ─── 1.5 — verbatim phrase (quotes + backticks) ─────────────────────


@pytest.mark.parametrize(
    "query",
    [
        '"4:20 PM"',
        'what time is "4:20"',
        "`userId`",
        "what is `handleClick` doing",
        # CJK quote characters
        "他说「我明天要去」",
        "他说《红楼梦》",
        "参考【要点】",
        "『こんにちは』と彼は言った",
        # Mixed-script: backtick beats everything else
        "查看 `userId` 的记录",
    ],
)
def test_verbatim_phrase_triggers_1_5(query: str) -> None:
    assert compute_query_bias(query) == 1.5


# ─── 1.4 — URL / email / file path ──────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "check https://example.com for details",
        "email me at foo@bar.com",
        "the bug is in src/main.py",
        "open ./docs/readme.md",
        # Works embedded in CJK
        "查看 https://example.com 里的信息",
        "报告在 reports/q4.pdf 里",
    ],
)
def test_exact_literal_triggers_1_4(query: str) -> None:
    assert compute_query_bias(query) == 1.4


# ─── 1.3 — concrete structural tokens ──────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        # Numeric / temporal
        "what time at 4:20?",
        "meeting on 2026-04-22",
        "budget was 42000",
        # Hashtag / @-mention
        "did we tag it #release",
        "ping @alice about it",
        # Code identifiers
        "review iPhone release notes",  # CamelCase
        "value of my_variable at exit",  # snake_case
        "the foo-bar config flag",  # kebab-case
        # Version
        "upgrading to v1.2.3 soon",
        # Works across scripts
        "预约 4:20 的会议",
        "看看 my_config 的值",
    ],
)
def test_concrete_structural_triggers_1_3(query: str) -> None:
    assert compute_query_bias(query) == 1.3


# ─── 1.2 — acronym (Latin ALL-CAPS ≥3) ──────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "trip to the USA",
        "NASA launch today",
        "working at IBM right now",
    ],
)
def test_acronym_triggers_1_2(query: str) -> None:
    assert compute_query_bias(query) == 1.2


# ─── 1.0 — neutral natural language, any script ────────────────────


@pytest.mark.parametrize(
    "query",
    [
        # English natural language
        "what did we decide",
        "how did the review go",
        # Latin proper nouns NO LONGER trigger a bias (we dropped the
        # NER-ish Title-Case heuristic — recall stays uniform across
        # scripts).
        "Alice came by yesterday",
        "I saw Señor García",
        "what did Müller say",
        # All-lowercase
        "alice came by yesterday",
        "i thought so",
        # Short non-Latin queries are trusted as conceptual
        "什么时候见 Alice",
        "今天的会议讨论了什么",
        "お元気ですか",
        "ما الذي حدث",
    ],
)
def test_neutral_returns_1_0(query: str) -> None:
    assert compute_query_bias(query) == 1.0


# ─── ordering: strongest cue wins ───────────────────────────────────


@pytest.mark.parametrize(
    "query, expected",
    [
        # Quote + time → quote wins (1.5)
        ('"meeting at 4:20"', 1.5),
        # URL + number → URL wins (1.4)
        ("see https://example.com at 4:20", 1.4),
        # CamelCase + acronym → CamelCase wins (1.3)
        ("iPhone vs USA market", 1.3),
    ],
)
def test_strongest_cue_wins(query: str, expected: float) -> None:
    assert compute_query_bias(query) == expected


def test_empty_query_defaults() -> None:
    assert compute_query_bias("") == 1.0
    assert compute_query_bias("   ") == 1.0


# ─── deep-cue detection (unchanged) ─────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "what did we discuss long ago",
        "Years ago there was a plan",
        "originally the setup was simpler",
        "hace mucho que no lo veo",  # Spanish
        "il y a longtemps, avant la guerre",  # French
        "vor langer Zeit gab es",  # German
        "很久以前的事",  # Chinese
        "昔々、ある村に",  # Japanese
        "오래 전에 있었다",  # Korean
    ],
)
def test_has_deep_cue_matches_multilingual_markers(query: str) -> None:
    assert has_deep_cue(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "what did we discuss today",
        "see you tomorrow at 4:20",
        "没什么特别的",  # Chinese: "nothing special"
        "",
    ],
)
def test_has_deep_cue_ignores_recent_queries(query: str) -> None:
    assert has_deep_cue(query) is False
