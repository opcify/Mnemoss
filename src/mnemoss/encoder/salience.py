"""Heuristic salience scoring (Stage 3).

A rule-based multi-signal score in ``[0, 1]`` that fills
``Memory.salience`` at encode time. The formula uses salience via
``α · salience`` in ``idx_priority``, so a higher score keeps a memory
in a warmer tier longer.

Signals (each normalized to ``[0, 1]``):

- **punctuation**: ``?``, ``!``, ``…``, ``...`` indicate question
  density or emphasis.
- **proper_nouns**: 0, 1, or 2+ Latin-script Title-Case tokens.
- **numeric_temporal**: any time/date/multi-digit number present.
- **length_band**: short messages ("ok") are low-info; medium messages
  are peak; very long messages get a slight rambling penalty.
- **caps_ratio**: all-caps word ratio — shouting usually means the
  writer flagged it.

Signals are averaged (equal weights) and clamped. Each signal is pure
and deterministic; no language model, no stateful vocabulary. The whole
function runs in microseconds.

Non-Latin-script content (Chinese, Japanese, Arabic) currently scores
zero on the proper-nouns signal — it relies on the other four. A
proper multilingual NER pass is Stage 5+.
"""

from __future__ import annotations

import re

from mnemoss.formula.query_bias import (
    _DATE_RE,
    _EN_TITLECASE_STOPWORDS,
    _NUMBER_RE,
    _TIME_RE,
    _WORD_RE,
)

_EMPHASIS_CHARS = ("?", "!", "…")
_ELLIPSIS_DOTS = "..."


def compute_salience(content: str, role: str | None = None) -> float:
    """Compute salience ∈ ``[0, 1]`` for ``content``.

    ``role`` is currently unused but kept in the signature so future
    stages can give e.g. ``tool_result`` memories a different weighting
    without changing callers.
    """

    text = content.strip()
    if not text:
        return 0.0

    signals = (
        _punctuation_signal(text),
        _proper_noun_signal(text),
        _numeric_signal(text),
        _length_signal(text),
        _caps_signal(text),
    )
    avg = sum(signals) / len(signals)
    return max(0.0, min(1.0, avg))


# ─── signals ────────────────────────────────────────────────────────


def _punctuation_signal(text: str) -> float:
    count = sum(text.count(c) for c in _EMPHASIS_CHARS)
    # "..." is an ellipsis variant; count non-overlapping runs.
    count += text.count(_ELLIPSIS_DOTS)
    # Saturation: 3+ emphasis marks ≈ max signal.
    return min(1.0, count / 3.0)


def _proper_noun_signal(text: str) -> float:
    n = _count_latin_proper_nouns(text)
    if n <= 0:
        return 0.0
    if n == 1:
        return 0.5
    return 1.0


def _numeric_signal(text: str) -> float:
    if (
        _TIME_RE.search(text)
        or _DATE_RE.search(text)
        or _NUMBER_RE.search(text)
    ):
        return 1.0
    return 0.0


def _length_signal(text: str) -> float:
    n = len(text)
    if n < 10:
        return 0.1
    if n < 20:
        return 0.5
    if n <= 200:
        return 1.0
    if n <= 500:
        return 0.8
    return 0.6


_CAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")


def _caps_signal(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    caps = len(_CAPS_WORD_RE.findall(text))
    ratio = caps / len(tokens)
    # One caps-word in every three tokens = max signal.
    return min(1.0, ratio * 3.0)


def _count_latin_proper_nouns(text: str) -> int:
    """Count Latin-script Title-Case tokens that aren't stopwords."""

    n = 0
    for token in _WORD_RE.findall(text):
        if token.isupper():
            continue
        if not token[0].isupper():
            continue
        if token.lower() in _EN_TITLECASE_STOPWORDS:
            continue
        n += 1
    return n
