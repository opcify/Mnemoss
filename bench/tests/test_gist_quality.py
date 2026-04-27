"""Tests for the gist-quality LLM-as-judge.

Three pieces under test:

  - ``parse_judge_response``: defensively maps freeform replies
    ("A", "  B.", "tie", "neither", lowercase, with punctuation) to
    the canonical {A, B, tie} space.
  - ``_judge_pair``: order-randomization correctness — the judge's
    raw "A" vs "B" reply must be mapped back to the actual content
    (level-1 vs level-2) regardless of swap.
  - ``summarize``: verdict thresholds (CUT ≤ 55%, KEEP ≥ 65%,
    REBUILD otherwise) match docs/dreaming-decision.md.

These tests use ``MockLLMClient`` and don't hit the network.
"""

from __future__ import annotations

import random

import pytest

from bench.gist_quality import (
    Comparison,
    _judge_pair,
    build_judge_prompt,
    parse_judge_response,
    summarize,
)
from mnemoss import MockLLMClient

# ─── parse_judge_response ──────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("A", "A"),
        ("B", "B"),
        ("tie", "tie"),
        ("  A  ", "A"),
        ("a", "A"),
        ("A.", "A"),
        ('"B"', "B"),
        ("Tie.", "tie"),
        ("EQUAL", "tie"),
        ("neither", "tie"),
        ("both are good", "tie"),
        ("", "tie"),
        ("garbled response from a confused model", "tie"),
    ],
)
def test_parse_judge_handles_messy_replies(raw: str, expected: str) -> None:
    assert parse_judge_response(raw) == expected


# ─── prompt construction ───────────────────────────────────────────


def test_build_judge_prompt_includes_all_inputs() -> None:
    p = build_judge_prompt(
        query="when was the kickoff",
        gist_a="kickoff held tuesday morning",
        gist_b="alice attended a kickoff event",
    )
    assert "when was the kickoff" in p
    assert "kickoff held tuesday morning" in p
    assert "alice attended a kickoff event" in p
    # Pin the response format so a future prompt edit doesn't silently
    # break parse_judge_response.
    assert '"A", "B"' in p or "A, B" in p


# ─── order randomization correctness ───────────────────────────────


async def test_judge_pair_maps_a_back_to_level_1_when_not_swapped() -> None:
    """Without swap: position A holds level-1; judge says "A" → level-1 wins."""

    rng = random.Random(0)  # rng.random() < 0.5 is False on first call
    # Force no swap by patching: random(0) returns >=0.5 first.
    # Easier: use a callback judge that always returns "A".
    judge = MockLLMClient(callback=lambda _p: "A")

    # Drive RNG to a known state where the first random() < 0.5 is False
    # (no swap). random.Random(0).random() == 0.844... so swap=False.
    cmp = await _judge_pair(
        judge, "Q", "m1", gist_level_1="L1 text", gist_level_2="L2 text", rng=rng
    )
    assert cmp.swap is False
    # Position A held level-1; judge picked A → level-1 won.
    # consolidate_won = 0.0 means level-1 won.
    assert cmp.consolidate_won == 0.0
    assert cmp.parsed == "A"


async def test_judge_pair_maps_a_back_to_level_2_when_swapped() -> None:
    """With swap: position A holds level-2; judge says "A" → level-2 wins."""

    # random.Random(2).random() = 0.957... > 0.5 → swap=False; we need swap=True.
    # random.Random(1).random() = 0.134... < 0.5 → swap=True.
    rng = random.Random(1)
    judge = MockLLMClient(callback=lambda _p: "A")
    cmp = await _judge_pair(
        judge, "Q", "m1", gist_level_1="L1 text", gist_level_2="L2 text", rng=rng
    )
    assert cmp.swap is True
    assert cmp.consolidate_won == 1.0
    assert cmp.parsed == "A"


async def test_judge_pair_tie_gets_half_credit() -> None:
    rng = random.Random(0)
    judge = MockLLMClient(callback=lambda _p: "tie")
    cmp = await _judge_pair(judge, "Q", "m1", gist_level_1="L1", gist_level_2="L2", rng=rng)
    assert cmp.parsed == "tie"
    assert cmp.consolidate_won == 0.5


async def test_judge_pair_handles_messy_reply_as_tie() -> None:
    """Defensive: an unparseable reply maps to "tie", not exception."""

    rng = random.Random(0)
    judge = MockLLMClient(callback=lambda _p: "I'm not sure honestly")
    cmp = await _judge_pair(judge, "Q", "m1", gist_level_1="L1", gist_level_2="L2", rng=rng)
    assert cmp.parsed == "tie"
    assert cmp.consolidate_won == 0.5


# ─── summarize / verdict thresholds ────────────────────────────────


def _build_comparisons(consolidate_won_values: list[float]) -> list[Comparison]:
    return [
        Comparison(
            query="q",
            member_id=f"m{i}",
            gist_level_1="L1",
            gist_level_2="L2",
            swap=False,
            raw_judge_reply="A",
            parsed="A",
            consolidate_won=v,
        )
        for i, v in enumerate(consolidate_won_values)
    ]


def test_summarize_keep_when_lower_ci_above_065() -> None:
    """100 wins out of 100 → CI is essentially [1.0, 1.0]."""

    comps = _build_comparisons([1.0] * 100)
    s = summarize(comps)
    assert s["verdict"] == "KEEP"
    assert s["win_rate"] == 1.0


def test_summarize_cut_when_upper_ci_below_055() -> None:
    """0 wins out of 100 → CI is essentially [0.0, 0.0]."""

    comps = _build_comparisons([0.0] * 100)
    s = summarize(comps)
    assert s["verdict"] == "CUT"
    assert s["win_rate"] == 0.0


def test_summarize_rebuild_when_ci_straddles_50() -> None:
    """50/50 outcomes — CI brackets 0.5 → REBUILD verdict."""

    comps = _build_comparisons([1.0] * 50 + [0.0] * 50)
    s = summarize(comps)
    assert s["verdict"] == "REBUILD"
    assert s["win_rate"] == pytest.approx(0.5)


def test_summarize_counts_ties() -> None:
    comps = [
        Comparison("q", "m1", "L1", "L2", False, "tie", "tie", 0.5),
        Comparison("q", "m2", "L1", "L2", False, "A", "A", 0.0),
        Comparison("q", "m3", "L1", "L2", True, "tie", "tie", 0.5),
    ]
    s = summarize(comps)
    assert s["ties"] == 2
    assert s["num_comparisons"] == 3
