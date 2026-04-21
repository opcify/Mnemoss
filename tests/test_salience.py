"""Multi-dimensional salience tests (Checkpoint K)."""

from __future__ import annotations

import pytest

from mnemoss.encoder.salience import compute_salience


def test_empty_content_is_zero() -> None:
    assert compute_salience("") == 0.0
    assert compute_salience("   ") == 0.0


def test_bounded_in_unit_interval() -> None:
    # Extreme cases — should never go outside [0, 1].
    extreme = "WOW!!! ??? Alice Bob Carol URGENT URGENT URGENT 2026-04-22 4:20 !!!"
    val = compute_salience(extreme)
    assert 0.0 <= val <= 1.0


def test_short_trivial_is_low() -> None:
    low = compute_salience("ok")
    high = compute_salience(
        "Alice said the meeting on 2026-04-22 at 4:20 was important!"
    )
    assert low < 0.3
    assert high > low


def test_question_marks_raise_salience() -> None:
    plain = compute_salience("the meeting is tomorrow")
    asked = compute_salience("the meeting is tomorrow?")
    urgent = compute_salience("is the meeting tomorrow???")
    assert asked > plain
    assert urgent > asked


def test_proper_nouns_raise_salience() -> None:
    generic = compute_salience("the person asked about the place")
    named = compute_salience("the person Alice asked about the place Paris")
    assert named > generic


def test_numeric_temporal_content_raises_salience() -> None:
    plain = compute_salience("see you later at the usual place")
    numeric = compute_salience("see you later at 4:20 at the usual place")
    assert numeric > plain


def test_length_band_peaks_in_middle() -> None:
    short = compute_salience("ok")           # < 10 → 0.1
    brief = compute_salience("see you soon")  # 13 chars → 0.5
    medium = compute_salience(
        "Alice said the meeting moved to Tuesday afternoon at 4:20."
    )                                         # ~58 chars → 1.0 length
    # short < brief < medium on the length dimension alone.
    assert short < brief < medium


def test_very_long_gets_slight_penalty() -> None:
    medium = compute_salience("A" * 150)  # in peak band; no proper-noun / num
    very_long = compute_salience("A " * 400)  # > 500 → 0.6 band
    # Both use only the length signal meaningfully; caps-ratio zero because
    # single-letter tokens don't match _CAPS_WORD_RE (requires 3+ chars).
    assert medium > very_long


def test_caps_ratio_raises_salience() -> None:
    calm = compute_salience("this is an alert for you to review")
    loud = compute_salience("this is an URGENT ALERT for you to review")
    assert loud > calm


def test_cjk_content_uses_other_signals() -> None:
    # CJK can't hit proper-nouns (Latin-only), so it relies on length +
    # numerics + punctuation.
    base = compute_salience("今天有会议")
    with_time = compute_salience("今天下午 4:20 有会议")
    assert with_time > base


@pytest.mark.parametrize(
    "content",
    [
        "hello",
        "Alice came by",
        "the meeting is at 4:20 tomorrow",
        "WARNING: system compromised",
        "¿cuándo vienes?",
    ],
)
def test_always_in_range(content: str) -> None:
    val = compute_salience(content)
    assert 0.0 <= val <= 1.0
